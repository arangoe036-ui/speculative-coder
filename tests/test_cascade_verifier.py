"""Tests for early-exit cascade verification.

The module's job is to *measure* whether an intermediate layer can be trusted, so the
tests concentrate on the measurement being honest: that the projection applies the
final norm (without it the mid-layer distribution looks far worse than it is), that
agreement is computed against the full model rather than against itself, and that the
weight-traffic accounting charges deferred KV work instead of counting a postponement
as a saving.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cascade_verifier import (  # noqa: E402
    CascadeBudget,
    decoder_layers,
    final_norm,
    measure_cascade,
    project_intermediate,
)
from tests.test_engine import _tiny_model  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return _tiny_model(1234)


# ---------------------------------------------------------------- plumbing
def test_locates_layers_and_norm(model):
    layers = decoder_layers(model)
    assert len(layers) == model.config.num_hidden_layers
    assert final_norm(model) is not None


def test_missing_layers_raises():
    class Bare(torch.nn.Module):
        pass

    with pytest.raises(AttributeError):
        decoder_layers(Bare())


def test_projection_applies_the_final_norm(model):
    """The norm is not optional: lm_head was fitted to normalised states.

    Verified by scaling the input hugely -- a normalising projection is nearly
    invariant to that, a raw one is not. Omitting the norm would make the mid-layer
    distribution look far worse than it is and would understate the idea being tested.
    """
    torch.manual_seed(0)
    hidden = torch.randn(4, model.config.hidden_size)
    logits = project_intermediate(model, hidden)
    scaled = project_intermediate(model, hidden * 50.0)

    assert logits.shape == (4, model.config.vocab_size)
    assert torch.allclose(logits, scaled, atol=1e-2), (
        "projection is not normalising its input"
    )

    raw_head = model.get_output_embeddings()
    unnormed = raw_head(hidden.to(raw_head.weight.dtype))
    assert not torch.allclose(logits, unnormed, atol=1e-2), (
        "projection appears to skip the norm entirely"
    )


def test_projection_of_the_final_state_matches_the_model(model):
    """Projecting the last hidden state must reproduce the model's own logits.

    This anchors the whole measurement: if the projection were wrong, mid-layer
    agreement would be understated everywhere and the conclusion would be an artefact.
    """
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    projected = project_intermediate(model, out.hidden_states[-1][0])
    assert torch.allclose(projected.float(), out.logits[0].float(), atol=1e-3)


# ---------------------------------------------------------------- measurement
def test_measure_cascade_at_the_final_layer_is_perfect(model):
    """A gate placed at the top must agree with the full model 100% of the time.

    The strongest available check on the measurement itself -- any disagreement here
    would mean agreement is being computed against the wrong reference.
    """
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    stats = measure_cascade(
        model, ids, early_layer_idx=model.config.num_hidden_layers, threshold=0.0
    )
    assert stats["agreement_all"] == pytest.approx(1.0)
    assert stats["fire_rate"] == pytest.approx(1.0)
    assert stats["wrong_accepts"] == 0


def test_measure_cascade_threshold_gates_the_fire_rate(model):
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    never = measure_cascade(model, ids, early_layer_idx=1, threshold=1.0)
    always = measure_cascade(model, ids, early_layer_idx=1, threshold=0.0)
    assert never["fire_rate"] <= always["fire_rate"]
    assert always["fire_rate"] == pytest.approx(1.0)
    assert never["fired"] == 0


def test_measure_cascade_reports_wrong_accepts(model):
    """Fired-but-disagreeing positions must be counted; they are the lossiness."""
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    stats = measure_cascade(model, ids, early_layer_idx=1, threshold=0.0)
    expected_wrong = round((1 - stats["agreement"]) * stats["fired"])
    assert stats["wrong_accepts"] == pytest.approx(expected_wrong, abs=1)


# ---------------------------------------------------------------- accounting
def test_budget_charges_deferred_work():
    """An early exit defers upper-layer KV work; it must not read as a saving.

    `layers_per_token` shows what ran, but `layers_per_token_with_debt` is the honest
    figure, and with every exit outstanding it must equal the full depth.
    """
    budget = CascadeBudget(total_layers=28, early_layer_idx=14)
    for _ in range(10):
        budget.record_early_exit()

    assert budget.layers_per_token == pytest.approx(14.0)
    assert budget.layers_per_token_with_debt == pytest.approx(28.0), (
        "deferring the top half is being counted as deleting it"
    )
    assert budget.early_exit_rate == pytest.approx(1.0)


def test_budget_full_pass_repays_the_backlog():
    """A batched full pass clears deferred debt, which is the only way this pays."""
    budget = CascadeBudget(total_layers=28, early_layer_idx=14)
    for _ in range(4):
        budget.record_early_exit()
    assert budget.layers_owed == 4 * 14

    budget.record_full_pass(positions=1, repaid=4)
    assert budget.layers_owed == 0
    # Five tokens: four cheap passes plus one full pass that also caught up the four.
    assert budget.tokens == 5
    assert budget.layers_per_token == budget.layers_per_token_with_debt


def test_budget_never_exits_early_is_exactly_baseline():
    budget = CascadeBudget(total_layers=28, early_layer_idx=14)
    for _ in range(6):
        budget.record_full_pass()
    assert budget.layers_per_token == pytest.approx(28.0)
    assert budget.early_exit_rate == 0.0
    assert "baseline 28" in budget.summary()


def test_budget_empty_is_zero_not_a_crash():
    budget = CascadeBudget(total_layers=28, early_layer_idx=14)
    assert budget.layers_per_token == 0.0
    assert budget.layers_per_token_with_debt == 0.0
    assert budget.early_exit_rate == 0.0
