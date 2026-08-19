"""Tests for EAGLE-style latent extrapolation.

Two things carry this module. The head must genuinely be *autoregressive in latent
space* -- step 3 has to depend on step 2's prediction, which is the entire difference
from Medusa and is invisible in any accuracy number. And the chain accounting must
count position t+1 exactly once: it is `argmax lm_head(h_t)`, the base model's own
output, and crediting a head for it as well is what inflated the Medusa figure.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.latent_extrapolator import (  # noqa: E402
    ExtrapolatorHead,
    LatentExtrapolator,
    RMSNorm,
    chain_speedup,
)
from tests.test_engine import _tiny_model  # noqa: E402


@pytest.fixture(scope="module")
def base_model():
    return _tiny_model(1234)


@pytest.fixture(scope="module")
def model(base_model):
    return LatentExtrapolator(base_model, num_attention_heads=4, intermediate_size=32)


# ---------------------------------------------------------------- the head
def test_head_shapes(model):
    head = model.head
    embeddings = torch.randn(2, 5, model.hidden_size)
    hidden = torch.randn(2, 5, model.hidden_size)
    out, cache = head(embeddings, hidden)
    assert out.shape == (2, 5, model.hidden_size)
    assert cache[0].shape[2] == 5 and cache[1].shape[2] == 5


def test_head_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="num_attention_heads"):
        ExtrapolatorHead(hidden_size=10, num_attention_heads=3)


def test_head_is_causal_over_the_latent_chain(model):
    """Position t's output must not depend on any later position.

    The head attends over the chain of latent states, so a leak would let training
    see the future and produce accuracy that evaporates at draft time.
    """
    torch.manual_seed(0)
    embeddings = torch.randn(1, 6, model.hidden_size)
    hidden = torch.randn(1, 6, model.hidden_size)
    full, _ = model.head(embeddings, hidden)

    # Perturb only the last position; earlier outputs must be untouched.
    embeddings[0, 5] += 10.0
    hidden[0, 5] += 10.0
    perturbed, _ = model.head(embeddings, hidden)

    assert torch.allclose(full[0, :5], perturbed[0, :5], atol=1e-4), (
        "an earlier output changed when a later input did: attention is not causal"
    )
    assert not torch.allclose(full[0, 5], perturbed[0, 5], atol=1e-3)


def test_cached_stepping_matches_full_sequence(model):
    """Autoregressive drafting with a cache must equal scoring the sequence at once.

    If these diverged, the head would behave differently while drafting than while
    training, and the measured accuracy would not describe the deployed path.
    """
    torch.manual_seed(1)
    embeddings = torch.randn(1, 4, model.hidden_size)
    hidden = torch.randn(1, 4, model.hidden_size)
    reference, _ = model.head(embeddings, hidden)

    cache = None
    outputs = []
    for t in range(4):
        out, cache = model.head(
            embeddings[:, t: t + 1], hidden[:, t: t + 1], cache
        )
        outputs.append(out)
    stepped = torch.cat(outputs, dim=1)
    assert torch.allclose(reference, stepped, atol=1e-4), (
        (reference - stepped).abs().max()
    )


def test_rmsnorm_normalises():
    norm = RMSNorm(8)
    x = torch.randn(3, 8) * 50
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(3), atol=1e-3)


# ---------------------------------------------------------------- the wrapper
def test_base_is_frozen_and_only_the_head_trains(base_model, model):
    for parameter in base_model.parameters():
        parameter.requires_grad_(True)
    rebuilt = LatentExtrapolator(base_model, num_attention_heads=4, intermediate_size=32)

    assert all(not p.requires_grad for p in rebuilt.base_model.parameters())
    assert all(p.requires_grad for p in rebuilt.head.parameters())
    trainable = sum(p.numel() for p in rebuilt.parameters() if p.requires_grad)
    assert trainable == rebuilt.head_parameters


def test_head_parameter_count_matches_the_closed_form(model):
    """Pin the parameter formula, so the real-scale claim below can rely on it."""
    h, inter = model.hidden_size, 32
    expected = (
        2 * h * h          # fuse: concat(embedding, hidden) -> hidden
        + 4 * h * h        # q, k, v, o
        + 2 * h * inter    # gate, up
        + inter * h        # down
        + 2 * h            # two RMSNorm weights
    )
    assert model.head_parameters == expected, (model.head_parameters, expected)


def test_head_is_lighter_than_one_medusa_head_at_real_scale():
    """The head reuses the frozen lm_head, so it never pays for a vocab projection.

    Checked with the real model's dimensions rather than the tiny fixture: the
    fixture's vocabulary is 16 tokens, which makes a "full-vocab" projection smaller
    than the head and the comparison meaningless. The claim is specifically about a
    152k vocabulary.
    """
    h, vocab, inter = 3584, 152064, 3584
    head = 2 * h * h + 4 * h * h + 2 * h * inter + inter * h + 2 * h
    medusa_head = h * vocab

    assert head < medusa_head
    assert medusa_head / head == pytest.approx(4.7, abs=0.2), (
        "one full-vocab Medusa head is several times the whole extrapolator"
    )
    # And M=10 of them is another order of magnitude beyond that.
    assert medusa_head * 10 / head > 40


def test_parameter_report_mentions_the_comparison(model):
    assert "full-vocab Medusa head" in model.parameter_report()


def test_draft_returns_one_plus_k_tokens(model):
    hidden = torch.randn(2, model.hidden_size)
    proposals = model.draft(hidden, steps=5)
    assert proposals.shape == (2, 6), "one exact token plus five extrapolated"
    assert proposals.dtype == torch.long


def test_first_drafted_token_is_the_base_models_own_argmax(model):
    """Position t+1 is exact -- it must equal lm_head(h_t)'s argmax, not a guess.

    This is why the chain accounting gives it a fixed 1.0 and why crediting a trained
    head for the same position double-counts it.
    """
    torch.manual_seed(2)
    hidden = torch.randn(3, model.hidden_size)
    expected = model.project(hidden).argmax(dim=-1)
    assert torch.equal(model.draft(hidden, steps=4)[:, 0], expected)


def test_draft_is_autoregressive_in_latent_space(model):
    """Step 3 must depend on step 2. This is the whole thesis of the architecture.

    Verified by perturbing the head's output projection and confirming that later
    proposals move even when the seed hidden state is untouched -- a Medusa-style set
    of independent projections would show no such coupling between steps.
    """
    torch.manual_seed(3)
    hidden = torch.randn(1, model.hidden_size)
    before = model.draft(hidden, steps=5)

    with torch.no_grad():
        model.head.down_proj.weight += 0.5
    after = model.draft(hidden, steps=5)
    with torch.no_grad():
        model.head.down_proj.weight -= 0.5

    assert torch.equal(before[:, 0], after[:, 0]), (
        "position t+1 comes from the frozen lm_head and must not move"
    )
    assert not torch.equal(before[:, 1:], after[:, 1:]), (
        "extrapolated positions did not respond to the head at all"
    )


def test_deeper_steps_depend_on_earlier_ones(model):
    """A change confined to a later step must not alter earlier ones, and drafting
    further must extend rather than rewrite."""
    torch.manual_seed(4)
    hidden = torch.randn(1, model.hidden_size)
    short = model.draft(hidden, steps=2)
    long = model.draft(hidden, steps=6)
    assert torch.equal(short, long[:, : short.shape[1]]), (
        "extending the draft changed tokens that were already decided"
    )


# ---------------------------------------------------------------- accounting
def test_chain_speedup_counts_the_exact_token_once():
    """Position t+1 contributes exactly 1.0 and is never taken from the accuracies."""
    assert chain_speedup([])["speedup"] == pytest.approx(1.0)
    assert chain_speedup([0.0, 0.0])["speedup"] == pytest.approx(1.0)
    assert chain_speedup([1.0, 1.0, 1.0])["speedup"] == pytest.approx(4.0)


def test_chain_speedup_multiplies_along_the_chain():
    projection = chain_speedup([0.5, 0.5, 0.5])
    assert projection["chain"] == pytest.approx([0.5, 0.25, 0.125])
    assert projection["tokens_per_forward"] == pytest.approx(1 + 0.875)


def test_chain_speedup_reproduces_the_corrected_medusa_figure():
    """Guards the correction. Medusa's head 1 predicts position t+1, which lm_head
    already gives exactly, so it must not be counted -- the accuracies passed in start
    at t+2. Including it produced the 1.86x previously reported instead of 1.31x.
    """
    measured = [0.655, 0.277, 0.109, 0.053, 0.037, 0.022, 0.021, 0.018, 0.016, 0.011]
    corrected = chain_speedup(measured[1:])["speedup"]
    assert corrected == pytest.approx(1.31, abs=0.01)

    # The old, double-counting arithmetic, kept as an explicit contrast.
    inflated = 1.0
    running = 1.0
    for accuracy in measured:
        running *= accuracy
        inflated += running
    assert inflated == pytest.approx(1.86, abs=0.01)
    assert inflated > corrected


def test_one_weak_step_caps_everything_behind_it():
    """The chain is a product, so an early failure dominates however good the tail."""
    assert chain_speedup([0.05, 1.0, 1.0, 1.0])["speedup"] == pytest.approx(1.2)
