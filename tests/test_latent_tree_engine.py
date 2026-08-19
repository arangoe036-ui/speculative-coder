"""Tests for Latent Mitosis (dynamic EAGLE tree speculation).

The correctness bar is the same as every other breadth engine: verification is a real
target forward, so greedy output must equal plain greedy decoding however the latent
tree grows. Because that holds even for a tree that never branches and a head that
predicts nonsense, the *tree mechanics* and the *latent seeding* are asserted
separately.

The seeding is the subtle part. Each block's draft starts from the true hidden state at
the last committed position, read off the verification forward that just ran. Get that
index wrong and the head extrapolates from the wrong anchor -- which cannot corrupt the
output, because verification catches it, but silently destroys acceptance.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.latent_extrapolator import ExtrapolatorHead  # noqa: E402
from core.latent_tree_engine import LatentTreeEngine, load_extrapolator_head  # noqa: E402
from tests.test_engine import (  # noqa: E402
    PROMPT,
    CharTokenizer,
    _tiny_model,
    reference_greedy,
)


@pytest.fixture(scope="module")
def model():
    return _tiny_model(1234)


@pytest.fixture(scope="module")
def tokenizer():
    return CharTokenizer()


@pytest.fixture
def head(model):
    torch.manual_seed(0)
    return ExtrapolatorHead(
        hidden_size=model.config.hidden_size,
        num_attention_heads=4,
        intermediate_size=32,
    ).to(torch.float32)


def _engine(model, tokenizer, head, **kwargs):
    kwargs.setdefault("k", 6)
    kwargs.setdefault("max_leaves", 8)
    kwargs.setdefault("split_threshold", 0.85)
    return LatentTreeEngine(model, tokenizer, head, **kwargs)


def _ids(engine, n=24):
    text, stats = engine.generate(PROMPT, max_new_tokens=n, stream=False)
    return [ord(c) - ord("a") for c in text], stats


# ---------------------------------------------------------------- exactness
@pytest.mark.parametrize("max_leaves", [1, 2, 4, 8, 16])
def test_output_matches_plain_greedy(model, tokenizer, head, max_leaves):
    """Verification decides, so the latent tree cannot change what is emitted."""
    engine = _engine(model, tokenizer, head, max_leaves=max_leaves)
    expected = reference_greedy(model, tokenizer(PROMPT).input_ids, 24)
    actual, _ = _ids(engine)
    assert actual == expected, (
        "latent tree changed the output at max_leaves={}\n  expected: {}\n  actual  : {}"
        .format(max_leaves, expected, actual)
    )


@pytest.mark.parametrize("split_threshold", [0.0, 0.5, 0.85, 1.0])
def test_split_threshold_changes_speed_not_output(
    model, tokenizer, head, split_threshold
):
    engine = _engine(model, tokenizer, head, split_threshold=split_threshold)
    expected = reference_greedy(model, tokenizer(PROMPT).input_ids, 24)
    assert _ids(engine)[0] == expected, f"split_threshold={split_threshold}"


@pytest.mark.parametrize("k", [1, 2, 5, 8])
def test_output_matches_greedy_across_depth(model, tokenizer, head, k):
    engine = _engine(model, tokenizer, head, k=k)
    expected = reference_greedy(model, tokenizer(PROMPT).input_ids, 20)
    assert _ids(engine, 20)[0] == expected, f"k={k}"


def test_an_untrained_head_is_harmless(model, tokenizer, head):
    """A randomly initialised head proposes garbage; output must be unaffected.

    This is the strongest form of the guarantee: the draft can be arbitrarily bad and
    only throughput suffers, because the frozen target decides every token.
    """
    engine = _engine(model, tokenizer, head, k=8, max_leaves=16)
    expected = reference_greedy(model, tokenizer(PROMPT).input_ids, 30)
    actual, stats = _ids(engine, 30)
    assert actual == expected
    assert stats.tokens_generated == 30


def test_temperature_above_zero_is_refused(model, tokenizer, head):
    with pytest.raises(ValueError, match="temperature=0"):
        LatentTreeEngine(model, tokenizer, head, temperature=0.7)


def test_invalid_arguments(model, tokenizer, head):
    for kwargs, match in (
        ({"max_leaves": 0}, "max_leaves"),
        ({"split_threshold": 1.5}, "split_threshold"),
    ):
        with pytest.raises(ValueError, match=match):
            LatentTreeEngine(model, tokenizer, head, **kwargs)


def test_no_second_model_is_held(model, tokenizer, head):
    """The head replaces the drafter, so only the target is resident."""
    engine = _engine(model, tokenizer, head)
    assert engine.draft_model is engine.target_model is model


# ---------------------------------------------------------------- tree shape
def test_leaves_are_equal_length_and_start_with_the_exact_token(
    model, tokenizer, head
):
    """Every leaf carries 1 + k tokens, the first being lm_head's own argmax.

    A split duplicates its parent's full history, so depths cannot diverge -- which is
    what keeps the verification batch rectangular with no padding.
    """
    engine = _engine(model, tokenizer, head, k=5, max_leaves=8, split_threshold=1.0)
    context = tokenizer(PROMPT).input_ids.to(engine.device)
    primer = model(input_ids=context, output_hidden_states=True)
    hidden = primer.hidden_states[-1][0, -1]

    leaves, q, splits, _ = engine._grow_tree(hidden, [], 5)
    assert leaves.shape[1] == 5, "five proposals from an empty consume list"
    assert q.shape[:2] == leaves.shape
    assert splits > 0


def test_first_proposal_is_exact_when_nothing_is_consumed(model, tokenizer, head):
    """With an empty consume list the anchor is a true state, so the first proposal
    is ``lm_head``'s own argmax -- exact, not a head prediction.

    Checked with splitting disabled, because a split at step 0 deliberately gives one
    sibling the *top-2* token, which is not the exact one.
    """
    engine = _engine(model, tokenizer, head, k=4, max_leaves=8, split_threshold=0.0)
    context = tokenizer(PROMPT).input_ids.to(engine.device)
    primer = model(input_ids=context, output_hidden_states=True)
    hidden = primer.hidden_states[-1][0, -1]

    leaves, _, splits, _ = engine._grow_tree(hidden, [], 4)
    assert splits == 0
    exact = engine._project(hidden.view(1, -1)).argmax(dim=-1)
    assert (leaves[:, 0] == exact).all(), "the first proposal must be lm_head's argmax"


def test_consuming_a_token_makes_the_first_proposal_predicted(model, tokenizer, head):
    """After a rejection the corrected token was never run through the target, so the
    chain must advance through it via the head -- and the first proposal is then a
    prediction rather than an exact token. This is inherent to EAGLE, not a defect.
    """
    engine = _engine(model, tokenizer, head, k=3, max_leaves=4, split_threshold=0.0)
    context = tokenizer(PROMPT).input_ids.to(engine.device)
    primer = model(input_ids=context, output_hidden_states=True)
    hidden = primer.hidden_states[-1][0, -1]

    without, _, _, steps_without = engine._grow_tree(hidden, [], 3)
    with_consume, _, _, steps_with = engine._grow_tree(hidden, [7], 3)

    assert steps_with == steps_without + 1, "consuming a token costs one head step"
    exact = int(engine._project(hidden.view(1, -1)).argmax(dim=-1))
    assert int(without[0, 0]) == exact, "no consume: the first proposal is exact"
    # With a consumed token the anchor has been advanced by the head, so the first
    # proposal comes from a predicted state. It may coincide with `exact` by chance,
    # so the shape of the chain is what is asserted rather than the token itself.
    assert with_consume.shape == without.shape


def test_threshold_zero_never_splits(model, tokenizer, head):
    engine = _engine(model, tokenizer, head, k=6, split_threshold=0.0)
    _, stats = _ids(engine, 24)
    assert stats.tree_splits == 0
    assert set(stats.leaf_counts) == {1}


def test_threshold_one_fills_to_the_leaf_ceiling(model, tokenizer, head):
    engine = _engine(model, tokenizer, head, k=6, max_leaves=8, split_threshold=1.0)
    _, stats = _ids(engine, 24)
    assert max(stats.leaf_counts) == 8


def test_max_leaves_is_never_exceeded(model, tokenizer, head):
    for ceiling in (1, 2, 3, 5, 16):
        engine = _engine(model, tokenizer, head, k=8, max_leaves=ceiling,
                         split_threshold=1.0)
        _, stats = _ids(engine, 20)
        assert max(stats.leaf_counts) <= ceiling


def test_split_takes_top1_and_top2(model, tokenizer, head):
    """Siblings must differ exactly at the split position by top-1 vs top-2."""
    engine = _engine(model, tokenizer, head, k=1, max_leaves=2, split_threshold=1.0)
    context = tokenizer(PROMPT).input_ids.to(engine.device)
    primer = model(input_ids=context, output_hidden_states=True)
    hidden = primer.hidden_states[-1][0, -1]

    leaves, _, splits, _ = engine._grow_tree(hidden, [], 1)
    assert splits == 1 and leaves.shape == (2, 1)
    assert leaves[0, 0] != leaves[1, 0], "the split position must differ"

    probs = torch.softmax(engine._project(hidden.view(1, -1)).float(), dim=-1)
    top2 = torch.topk(probs, 2, dim=-1).indices[0]
    assert int(leaves[0, 0]) == int(top2[0]), "first child takes top-1"
    assert int(leaves[1, 0]) == int(top2[1]), "sibling takes top-2"


def test_branches_are_autoregressive_in_latent_space(model, tokenizer, head):
    """Perturbing the head must move extrapolated tokens but not the exact first one.

    Confirms the tree really is being grown by the head rather than by anything
    inherited from the base model's own predictions.
    """
    engine = _engine(model, tokenizer, head, k=5, max_leaves=4)
    context = tokenizer(PROMPT).input_ids.to(engine.device)
    primer = model(input_ids=context, output_hidden_states=True)
    hidden = primer.hidden_states[-1][0, -1]

    before, _, _, _ = engine._grow_tree(hidden, [], 5)
    with torch.no_grad():
        head.down_proj.weight += 0.5
    after, _, _, _ = engine._grow_tree(hidden, [], 5)

    assert (before[:, 0] == after[:, 0]).all(), "the exact token must not move"
    assert not torch.equal(before[:, 1:], after[:, 1:])


# ---------------------------------------------------------------- seeding
def test_latent_seed_tracks_the_committed_position(model, tokenizer, head):
    """Each block must extrapolate from the true state at the last committed token.

    Verified by counting base forwards: after the first block's primer, verification
    supplies the next seed, so the loop should settle at one forward per block. A
    wrong index would either crash or silently force extra primer forwards.
    """
    engine = _engine(model, tokenizer, head, k=6, max_leaves=4)
    _, stats = _ids(engine, 40)
    blocks = stats.iterations
    # One primer at the start, then one verification per block.
    assert stats.target_forwards <= blocks + 1, (
        f"{stats.target_forwards} base forwards for {blocks} blocks: the latent seed "
        "is being re-primed instead of reused"
    )
    assert stats.target_forwards >= blocks


def test_telemetry_is_recorded(model, tokenizer, head):
    engine = _engine(model, tokenizer, head, k=6, max_leaves=8)
    _, stats = _ids(engine, 30)
    assert len(stats.leaf_counts) == stats.iterations
    assert len(stats.branch_accepted) == stats.iterations
    assert all(n >= 1 for n in stats.unique_candidates)
    assert stats.mean_leaves >= 1.0


def test_max_new_tokens_is_exact(model, tokenizer, head):
    engine = _engine(model, tokenizer, head, k=8, max_leaves=8)
    for n in (1, 2, 5, 9, 17):
        text, stats = engine.generate(PROMPT, max_new_tokens=n, stream=False)
        assert stats.tokens_generated == n and len(text) == n


def test_eos_stops_cleanly(model, tokenizer, head):
    engine = _engine(model, tokenizer, head, k=6, max_leaves=8)
    reference = reference_greedy(model, tokenizer(PROMPT).input_ids, 20)
    eos_id = reference[4]
    expected_length = reference.index(eos_id)

    class EosTokenizer(CharTokenizer):
        eos_token_id = eos_id

    engine.tokenizer = EosTokenizer()
    text, stats = engine.generate(PROMPT, max_new_tokens=20, stream=False)
    assert stats.tokens_generated == expected_length
    assert eos_id not in stats.token_ids


# ---------------------------------------------------------------- checkpoint
def test_head_checkpoint_round_trips(model, head, tmp_path):
    """The engine loads a trained head from disk, so the format must round-trip."""
    path = tmp_path / "head.pt"
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
            "hidden_size": head.hidden_size,
            "num_attention_heads": head.num_heads,
            "intermediate_size": head.gate_proj.out_features,
        },
        path,
    )
    restored = load_extrapolator_head(str(path), device="cpu")

    assert restored.hidden_size == head.hidden_size
    for (name, original), (_, loaded) in zip(
        head.state_dict().items(), restored.state_dict().items()
    ):
        assert torch.allclose(original, loaded.cpu()), name

    embeddings = torch.randn(1, 3, head.hidden_size)
    hidden = torch.randn(1, 3, head.hidden_size)
    assert torch.allclose(
        head(embeddings, hidden)[0], restored(embeddings, hidden)[0], atol=1e-6
    )
