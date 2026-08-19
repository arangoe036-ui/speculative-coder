"""Tests for the hybrid cascade router.

The correctness bar is the same as everywhere else: routing changes which tokens
are proposed, never what is emitted. So greedy output must still match plain greedy
decoding whichever drafter served each block -- and because that holds no matter
how badly the n-gram drafter guesses, output tests alone cannot tell whether the
router is doing anything. The routing telemetry is therefore asserted directly.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hybrid_engine import HybridCascadeEngine  # noqa: E402
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


def _engine(draft, target, tokenizer, **kwargs):
    kwargs.setdefault("temperature", 0.0)
    kwargs.setdefault("use_dual_gate", False)
    return HybridCascadeEngine(draft, target, tokenizer, **kwargs)


def _ids(engine, n=24):
    text, stats = engine.generate(PROMPT, max_new_tokens=n, stream=False)
    return [ord(c) - ord("a") for c in text], stats


# ---------------------------------------------------------------- correctness
@pytest.mark.parametrize("min_match_len", [1, 2, 3, 5])
def test_routing_preserves_greedy_output(draft_model, target_model, tokenizer, min_match_len):
    """Whichever drafter serves a block, the emitted tokens are the target's.

    Swept over pattern length because it changes how often the n-gram path is
    taken; if any setting altered the output, the router would be biasing
    generation rather than accelerating it.
    """
    engine = _engine(draft_model, target_model, tokenizer, min_match_len=min_match_len)
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 24)
    actual, _ = _ids(engine)
    assert actual == expected, (
        "hybrid routing changed the output at min_match_len={}\n"
        "  expected: {}\n  actual  : {}".format(min_match_len, expected, actual)
    )


def test_matches_the_plain_engine_token_for_token(draft_model, target_model, tokenizer):
    """The cascade must agree with the non-routing engine it inherits from."""
    from core.engine import SpeculativeEngine

    plain = SpeculativeEngine(draft_model, target_model, tokenizer, k=5, temperature=0.0)
    hybrid = _engine(draft_model, target_model, tokenizer, k=5, min_match_len=2)
    plain_text, _ = plain.generate(PROMPT, max_new_tokens=24, stream=False)
    hybrid_text, _ = hybrid.generate(PROMPT, max_new_tokens=24, stream=False)
    assert hybrid_text == plain_text


def test_output_survives_a_deliberately_useless_router(draft_model, target_model, tokenizer):
    """A pathological fast path must cost throughput, never correctness.

    min_match_len=1 matches almost everywhere and predicts badly, which is the
    worst case for acceptance and therefore the best test of the guarantee.
    """
    engine = _engine(draft_model, target_model, tokenizer, min_match_len=1,
                     ngram_draft_len=5)
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 30)
    actual, stats = _ids(engine, 30)
    assert actual == expected
    assert stats.ngram_drafts_used > 0, "min_match_len=1 should route constantly"


# ---------------------------------------------------------------- routing
def test_router_actually_uses_both_paths(draft_model, target_model, tokenizer):
    """Both drafters must be exercised, or the telemetry proves nothing."""
    engine = _engine(draft_model, target_model, tokenizer, k=5, min_match_len=2)
    _, stats = _ids(engine, 40)

    assert stats.ngram_drafts_used > 0, "fast path never fired"
    assert stats.model_drafts_used > 0, "fallback never fired"
    assert stats.ngram_drafts_used + stats.model_drafts_used == stats.iterations
    assert 0.0 < stats.ngram_share < 1.0


def test_a_long_pattern_disables_the_fast_path(draft_model, target_model, tokenizer):
    """An unmatchable pattern length must degrade to pure model drafting."""
    engine = _engine(draft_model, target_model, tokenizer, k=5, min_match_len=500)
    _, stats = _ids(engine, 20)
    assert stats.ngram_drafts_used == 0
    assert stats.model_drafts_used == stats.iterations


def test_fast_path_costs_no_draft_forwards(draft_model, target_model, tokenizer):
    """The entire point: an n-gram hit skips the GPU draft model.

    Asserted as an inequality against what pure model drafting would have spent,
    since that is the saving being claimed.
    """
    engine = _engine(draft_model, target_model, tokenizer, k=5, min_match_len=2)
    _, stats = _ids(engine, 40)

    # Every model-drafted block costs up to k forwards; n-gram blocks cost zero.
    assert stats.draft_forwards <= 5 * stats.model_drafts_used
    assert stats.ngram_drafts_used > 0
    would_have_cost = 5 * stats.iterations
    assert stats.draft_forwards < would_have_cost, (
        f"{stats.draft_forwards} draft forwards is no better than the "
        f"{would_have_cost} pure model drafting would have needed"
    )


def test_token_counters_split_by_source_and_reconcile(draft_model, target_model, tokenizer):
    engine = _engine(draft_model, target_model, tokenizer, k=5, min_match_len=2)
    _, stats = _ids(engine, 40)

    assert (stats.ngram_tokens_proposed + stats.model_tokens_proposed
            == stats.draft_tokens_proposed)
    assert (stats.ngram_tokens_accepted + stats.model_tokens_accepted
            == stats.draft_tokens_accepted)
    assert 0.0 <= stats.ngram_acceptance_rate <= 1.0
    assert 0.0 <= stats.model_acceptance_rate <= 1.0
    assert "n-gram" in stats.routing_summary()


def test_ngram_blocks_respect_the_token_budget(draft_model, target_model, tokenizer):
    """A fast-path proposal must not overrun the caller's remaining budget."""
    engine = _engine(draft_model, target_model, tokenizer, k=5, ngram_draft_len=5,
                     min_match_len=1)
    for n in (1, 2, 3, 7, 11):
        text, stats = engine.generate(PROMPT, max_new_tokens=n, stream=False)
        assert stats.tokens_generated == n and len(text) == n
        assert max(stats.draft_lengths) <= max(5, n)


def test_ngram_draft_len_caps_the_fast_path(draft_model, target_model, tokenizer):
    engine = _engine(draft_model, target_model, tokenizer, k=8, ngram_draft_len=2,
                     min_match_len=1)
    _, stats = _ids(engine, 30)
    assert stats.ngram_drafts_used > 0
    # n-gram blocks are capped at 2; model blocks may be longer, so check the mean
    # is pulled below the model's k.
    assert stats.ngram_tokens_proposed <= 2 * stats.ngram_drafts_used


def test_works_with_the_dual_gate_enabled(draft_model, target_model, tokenizer):
    """The cascade and the gate must compose: the gate governs only the fallback."""
    engine = HybridCascadeEngine(
        draft_model, target_model, tokenizer, temperature=0.0,
        use_dual_gate=True, max_draft_len=8, entropy_threshold=0.65,
        min_match_len=2,
    )
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 30)
    actual, stats = _ids(engine, 30)
    assert actual == expected
    # Gate triggers can only be attributed to model-drafted blocks.
    assert stats.statistical_gate_triggers <= stats.model_drafts_used


def test_draft_cache_recovers_after_a_fast_path_gap(draft_model, target_model, tokenizer):
    """Skipping the draft model leaves its cache behind; it must self-heal.

    The fallback feeds whatever the cache has not seen, so a run of n-gram hits is
    absorbed by re-feeding a longer tail rather than by corrupting state. Verified
    by generating well past several routing switches and checking the output is
    still exactly right.
    """
    engine = _engine(draft_model, target_model, tokenizer, k=5, min_match_len=2)
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 60)
    actual, stats = _ids(engine, 60)
    assert actual == expected
    assert stats.ngram_drafts_used > 0 and stats.model_drafts_used > 0


def test_invalid_arguments(draft_model, target_model, tokenizer):
    for kwargs in ({"ngram_draft_len": 0}, {"min_match_len": 0}):
        with pytest.raises(ValueError):
            HybridCascadeEngine(draft_model, target_model, tokenizer, **kwargs)
