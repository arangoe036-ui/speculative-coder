"""Tests for evolutionary tree speculation.

Correctness is the usual bar: greedy output must equal plain greedy decoding
whatever shape the tree grows into. Since that holds even for a tree that never
branches, the *branching mechanics* are asserted separately -- the split trigger,
the leaf ceiling, equal leaf depth, and the draft-row saving that is the whole
point of growing a tree instead of a fixed batch.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tree_engine import TreeSpeculativeEngine  # noqa: E402
from tests.test_engine import (  # noqa: E402
    PROMPT,
    CharTokenizer,
    _perturbed_model,
    _tiny_model,
    reference_greedy,
)


@pytest.fixture(scope="module")
def target_model():
    return _tiny_model(1234)


@pytest.fixture(scope="module")
def draft_model(target_model):
    return _perturbed_model(target_model, scale=0.02, seed=7)


@pytest.fixture(scope="module")
def tokenizer():
    return CharTokenizer()


def _ids(engine, n=24):
    text, stats = engine.generate(PROMPT, max_new_tokens=n, stream=False)
    return [ord(c) - ord("a") for c in text], stats


# ---------------------------------------------------------------- losslessness
@pytest.mark.parametrize("max_leaves", [1, 2, 4, 8, 16])
def test_output_matches_plain_greedy(draft_model, target_model, tokenizer, max_leaves):
    engine = TreeSpeculativeEngine(
        draft_model, target_model, tokenizer, k=8, max_leaves=max_leaves,
        split_threshold=0.8,
    )
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 24)
    actual, _ = _ids(engine)
    assert actual == expected, (
        "tree speculation changed the output at max_leaves={}\n"
        "  expected: {}\n  actual  : {}".format(max_leaves, expected, actual)
    )


@pytest.mark.parametrize("split_threshold", [0.0, 0.3, 0.8, 0.99, 1.0])
def test_output_is_independent_of_split_threshold(
    draft_model, target_model, tokenizer, split_threshold
):
    """How eagerly the tree branches may change speed, never content."""
    engine = TreeSpeculativeEngine(
        draft_model, target_model, tokenizer, k=8, max_leaves=8,
        split_threshold=split_threshold,
    )
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 24)
    assert _ids(engine)[0] == expected, f"split_threshold={split_threshold}"


@pytest.mark.parametrize("k", [1, 2, 5, 8])
def test_output_matches_greedy_across_k(draft_model, target_model, tokenizer, k):
    engine = TreeSpeculativeEngine(draft_model, target_model, tokenizer, k=k)
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 20)
    assert _ids(engine, 20)[0] == expected


def test_leaves_form_a_prefix_chain(draft_model, target_model, tokenizer):
    """Every leaf must emit a prefix of one string, or argmax is arbitrary."""
    from core.verifier import verify_tokens

    engine = TreeSpeculativeEngine(
        draft_model, target_model, tokenizer, k=8, max_leaves=8, split_threshold=0.99
    )
    context = tokenizer(PROMPT).input_ids
    leaves, q, _, _, _, _ = engine._draft_tree(context, None, 8)
    p, _ = engine._verify_leaves(context, leaves, None)

    emitted = []
    for i in range(leaves.shape[0]):
        seq, n_accepted = verify_tokens(q[i], p[i], leaves[i])
        assert len(seq) == n_accepted + 1
        emitted.append(seq.tolist())

    emitted.sort(key=len)
    for shorter, longer in zip(emitted, emitted[1:]):
        assert longer[: len(shorter)] == shorter, (
            f"leaves disagree on a shared position:\n  {shorter}\n  {longer}"
        )


def test_temperature_above_zero_is_refused(draft_model, target_model, tokenizer):
    with pytest.raises(ValueError, match="temperature=0"):
        TreeSpeculativeEngine(draft_model, target_model, tokenizer, temperature=0.7)


def test_invalid_arguments(draft_model, target_model, tokenizer):
    for kwargs, match in (
        ({"max_leaves": 0}, "max_leaves"),
        ({"split_threshold": -0.1}, "split_threshold"),
        ({"split_threshold": 1.5}, "split_threshold"),
    ):
        with pytest.raises(ValueError, match=match):
            TreeSpeculativeEngine(draft_model, target_model, tokenizer, **kwargs)


# ---------------------------------------------------------------- tree shape
def test_all_leaves_have_equal_depth(draft_model, target_model, tokenizer):
    """Every branch appends one token per step, so no padding is ever needed.

    This is what makes the verification batch rectangular for free. If a split
    ever failed to duplicate its parent's full history, leaf depths would diverge
    and the batched verify would silently compare misaligned positions.
    """
    engine = TreeSpeculativeEngine(
        draft_model, target_model, tokenizer, k=7, max_leaves=8, split_threshold=0.95
    )
    context = tokenizer(PROMPT).input_ids
    for depth in (1, 3, 7):
        leaves, q, _, _, _, _ = engine._draft_tree(context, None, depth)
        assert leaves.shape[1] == depth, "every leaf must be exactly k deep"
        assert q.shape[:2] == leaves.shape, "q must align with the tokens"


def test_threshold_zero_never_splits(draft_model, target_model, tokenizer):
    """A threshold of 0 cannot be undercut, so the tree stays a single trunk."""
    engine = TreeSpeculativeEngine(
        draft_model, target_model, tokenizer, k=8, max_leaves=8, split_threshold=0.0
    )
    _, stats = _ids(engine, 24)
    assert stats.tree_splits == 0
    assert set(stats.leaf_counts) == {1}
    # One trunk, k rows per iteration: the cheapest possible draft.
    assert stats.draft_row_forwards == sum(stats.draft_lengths)


def test_threshold_one_splits_at_every_opportunity(draft_model, target_model, tokenizer):
    """A threshold of 1 always splits, so the tree fills to the leaf ceiling."""
    engine = TreeSpeculativeEngine(
        draft_model, target_model, tokenizer, k=8, max_leaves=8, split_threshold=1.0
    )
    _, stats = _ids(engine, 24)
    assert stats.tree_splits > 0
    assert max(stats.leaf_counts) == 8, f"never reached the ceiling: {stats.leaf_counts}"


def test_max_leaves_is_never_exceeded(draft_model, target_model, tokenizer):
    for ceiling in (1, 2, 3, 5, 8):
        engine = TreeSpeculativeEngine(
            draft_model, target_model, tokenizer, k=8, max_leaves=ceiling,
            split_threshold=1.0,
        )
        _, stats = _ids(engine, 20)
        assert max(stats.leaf_counts) <= ceiling, (
            f"grew to {max(stats.leaf_counts)} leaves with ceiling {ceiling}"
        )


def test_split_takes_top1_and_top2(draft_model, target_model, tokenizer):
    """A split must produce siblings differing exactly at the split position.

    Checked by forcing a split on the first step and confirming the two leaves
    carry the draft's two most likely tokens.
    """
    engine = TreeSpeculativeEngine(
        draft_model, target_model, tokenizer, k=1, max_leaves=2, split_threshold=1.0
    )
    context = tokenizer(PROMPT).input_ids
    leaves, _, _, _, _, splits = engine._draft_tree(context, None, 1)

    assert splits == 1 and leaves.shape == (2, 1)
    # Recover the draft's own top-2 for that position and compare.
    with torch.no_grad():
        logits = draft_model(input_ids=context).logits[:, -1]
    top2 = torch.topk(engine._draft_probs(logits), 2, dim=-1).indices[0]
    assert leaves[0, 0].item() == int(top2[0]), "first child must take the top-1 token"
    assert leaves[1, 0].item() == int(top2[1]), "second child must take the top-2 token"


def test_split_threshold_read_from_unwarped_probabilities(
    draft_model, target_model, tokenizer
):
    """The target's temperature of 0 must not leak into the split decision.

    `_logits_to_probs` returns one-hot at temperature 0, whose top-1 probability is
    always exactly 1.0 -- so a split test built on it would never fire and the tree
    would silently never branch, looking identical to a correctly-implemented tree
    that simply found no uncertainty.
    """
    engine = TreeSpeculativeEngine(
        draft_model, target_model, tokenizer, k=8, max_leaves=8, split_threshold=0.9
    )
    context = tokenizer(PROMPT).input_ids
    with torch.no_grad():
        logits = draft_model(input_ids=context).logits[:, -1]
    probs = engine._draft_probs(logits)
    assert probs.max() < 1.0, "draft probabilities collapsed to one-hot"
    _, stats = _ids(engine, 24)
    assert stats.tree_splits > 0, "no split ever fired: threshold is reading one-hot q"


# ---------------------------------------------------------------- draft cost
def test_tree_uses_fewer_draft_rows_than_fixed_width(draft_model, target_model, tokenizer):
    """The central claim: a grown tree costs fewer draft rows than a fixed batch.

    Compared against Monte Carlo at the same B and K on the same prompt, since
    that is the configuration this design is meant to undercut.
    """
    from core.monte_carlo_engine import MonteCarloEngine

    torch.manual_seed(0)
    mc = MonteCarloEngine(draft_model, target_model, tokenizer, k=8, branches=8)
    _, mc_stats = _ids(mc, 40)

    tree = TreeSpeculativeEngine(
        draft_model, target_model, tokenizer, k=8, max_leaves=8, split_threshold=0.8
    )
    _, tree_stats = _ids(tree, 40)

    assert tree_stats.draft_row_forwards < mc_stats.draft_row_forwards, (
        f"tree spent {tree_stats.draft_row_forwards} draft rows vs Monte Carlo's "
        f"{mc_stats.draft_row_forwards}; growing the tree should cost less"
    )
    # Both make the same number of launches; only the widths differ.
    assert tree_stats.draft_forwards == sum(tree_stats.draft_lengths)


def test_row_count_matches_the_tree_shape(draft_model, target_model, tokenizer):
    """Row accounting must equal the sum of active widths, or the headline
    draft-compute claim is unverifiable."""
    engine = TreeSpeculativeEngine(
        draft_model, target_model, tokenizer, k=6, max_leaves=4, split_threshold=1.0
    )
    context = tokenizer(PROMPT).input_ids
    leaves, _, _, calls, rows, splits = engine._draft_tree(context, None, 6)

    assert calls == 6, "one forward launch per depth step"
    # Widths grow 1, 2, 4, 4, 4, 4 with a ceiling of 4 and always-split.
    assert rows == 1 + 2 + 4 + 4 + 4 + 4, rows
    assert leaves.shape[0] == 4 and splits == 3


def test_forward_accounting(draft_model, target_model, tokenizer):
    engine = TreeSpeculativeEngine(draft_model, target_model, tokenizer, k=8)
    _, stats = _ids(engine, 24)
    assert stats.target_forwards == stats.iterations
    assert stats.draft_row_forwards >= stats.draft_forwards
    assert stats.mean_leaves >= 1.0


# ---------------------------------------------------------------- bookkeeping
def test_max_new_tokens_is_exact(draft_model, target_model, tokenizer):
    engine = TreeSpeculativeEngine(draft_model, target_model, tokenizer, k=8, max_leaves=8)
    for n in (1, 2, 5, 9, 17):
        text, stats = engine.generate(PROMPT, max_new_tokens=n, stream=False)
        assert stats.tokens_generated == n and len(text) == n


def test_caches_collapse_to_one_row(draft_model, target_model, tokenizer):
    engine = TreeSpeculativeEngine(draft_model, target_model, tokenizer, k=4, max_leaves=8,
                                   split_threshold=1.0)
    seen = []
    original = engine._draft_tree

    def spy(context, cache, k):
        if cache is not None:
            seen.append(cache.layers[0].keys.shape[0])
        return original(context, cache, k)

    engine._draft_tree = spy
    engine.generate(PROMPT, max_new_tokens=30, stream=False)
    assert len(seen) > 4
    assert set(seen) == {1}, f"cache handed to the draft phase was batched: {seen}"


def test_eos_stops_cleanly(draft_model, target_model, tokenizer):
    engine = TreeSpeculativeEngine(draft_model, target_model, tokenizer, k=8, max_leaves=8)
    reference = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 20)
    eos_id = reference[4]
    expected_length = reference.index(eos_id)

    class EosTokenizer(CharTokenizer):
        eos_token_id = eos_id

    engine.tokenizer = EosTokenizer()
    text, stats = engine.generate(PROMPT, max_new_tokens=20, stream=False)
    assert stats.tokens_generated == expected_length
    assert eos_id not in stats.token_ids


def test_dual_gate_is_refused(draft_model, target_model, tokenizer):
    engine = TreeSpeculativeEngine(draft_model, target_model, tokenizer)
    engine.use_dual_gate = True
    with pytest.raises(NotImplementedError):
        engine.generate(PROMPT, max_new_tokens=4, stream=False)
