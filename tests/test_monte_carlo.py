"""Tests for Monte Carlo (parallel-branch) speculative decoding.

Two claims carry this engine and both are tested directly.

Losslessness: at target temperature 0 every branch emits a prefix of the same
greedy string, so committing the longest cannot change the content. If that
reasoning were wrong, output would drift from plain greedy decoding -- which is
exactly what the equivalence tests check, across B and draft temperature.

Divergence: if the branches did not actually explore different continuations, B
would be pure overhead and every acceptance count would be identical. That is
invisible in the output, so it is asserted on the telemetry instead.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import _cache_length  # noqa: E402
from core.monte_carlo_engine import (  # noqa: E402
    MonteCarloEngine,
    _expand_cache,
    _select_cache_row,
)
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
@pytest.mark.parametrize("branches", [1, 2, 4, 8])
def test_output_matches_plain_greedy_across_b(draft_model, target_model, tokenizer, branches):
    """Committing the longest-accepted branch must not change what is emitted."""
    torch.manual_seed(0)
    engine = MonteCarloEngine(
        draft_model, target_model, tokenizer, k=5, branches=branches,
        draft_temperature=1.2,
    )
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 24)
    actual, _ = _ids(engine)
    assert actual == expected, (
        "branch selection changed the output at B={}\n"
        "  expected: {}\n  actual  : {}".format(branches, expected, actual)
    )


@pytest.mark.parametrize("draft_temperature", [0.3, 1.0, 1.2, 2.5])
def test_output_is_independent_of_draft_temperature(
    draft_model, target_model, tokenizer, draft_temperature
):
    """The draft may be arbitrarily wild; the target still decides.

    This is the load-bearing consequence of `q` being the honest proposal density:
    the draft's temperature affects acceptance and therefore speed, never content.
    """
    torch.manual_seed(1)
    engine = MonteCarloEngine(
        draft_model, target_model, tokenizer, k=5, branches=4,
        draft_temperature=draft_temperature,
    )
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 24)
    assert _ids(engine)[0] == expected, f"draft_temperature={draft_temperature}"


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
def test_output_matches_greedy_across_k(draft_model, target_model, tokenizer, k):
    torch.manual_seed(2)
    engine = MonteCarloEngine(draft_model, target_model, tokenizer, k=k, branches=4)
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 20)
    assert _ids(engine, 20)[0] == expected


def test_every_branch_emits_a_prefix_of_the_same_string(draft_model, target_model, tokenizer):
    """The claim the whole design rests on, checked at the block level.

    At temperature 0, verifying branch b must yield the greedy continuation
    truncated to n_accepted + 1. So sorting the branches by acceptance count must
    give a chain of prefixes -- if two branches disagreed on a shared position,
    selecting by length would be picking between different strings and the engine
    would be biased.
    """
    from core.verifier import verify_tokens

    torch.manual_seed(3)
    engine = MonteCarloEngine(
        draft_model, target_model, tokenizer, k=5, branches=6, draft_temperature=1.5
    )
    context = tokenizer(PROMPT).input_ids

    draft_tokens, q, draft_cache, _ = engine._draft_branches(context, None, 5)
    p, _ = engine._verify_branches(context, draft_tokens, None)

    emitted = []
    for b in range(draft_tokens.shape[0]):
        tokens, n_accepted = verify_tokens(q[b], p[b], draft_tokens[b])
        emitted.append(tokens.tolist())
        assert len(tokens) == n_accepted + 1, "block length must be n_accepted + 1"

    emitted.sort(key=len)
    for shorter, longer in zip(emitted, emitted[1:]):
        assert longer[: len(shorter)] == shorter, (
            "branches disagree on a shared position, so choosing the longest is "
            f"choosing between different strings:\n  {shorter}\n  {longer}"
        )


def test_temperature_above_zero_is_refused(draft_model, target_model, tokenizer):
    """Selection biases the distribution when the target is sampled, so refuse.

    Silently biasing would be far worse than failing: the damage is statistical and
    invisible in any single generation.
    """
    with pytest.raises(ValueError, match="temperature=0"):
        MonteCarloEngine(draft_model, target_model, tokenizer, temperature=0.7)


def test_degenerate_draft_temperature_is_refused(draft_model, target_model, tokenizer):
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="draft_temperature"):
            MonteCarloEngine(draft_model, target_model, tokenizer, draft_temperature=bad)
    with pytest.raises(ValueError, match="branches"):
        MonteCarloEngine(draft_model, target_model, tokenizer, branches=0)


# ---------------------------------------------------------------- divergence
def test_branches_actually_diverge(draft_model, target_model, tokenizer):
    """B is worthless unless the branches explore different continuations."""
    torch.manual_seed(4)
    engine = MonteCarloEngine(
        draft_model, target_model, tokenizer, k=5, branches=4, draft_temperature=1.5
    )
    context = tokenizer(PROMPT).input_ids
    draft_tokens, _, _, _ = engine._draft_branches(context, None, 5)

    rows = {tuple(row) for row in draft_tokens.tolist()}
    assert len(rows) > 1, f"all {draft_tokens.shape[0]} branches proposed the same tokens"


def test_breadth_buys_acceptance(draft_model, target_model, tokenizer):
    """The winning branch must beat an average branch, or B is pure overhead."""
    torch.manual_seed(5)
    engine = MonteCarloEngine(
        draft_model, target_model, tokenizer, k=5, branches=6, draft_temperature=1.5
    )
    _, stats = _ids(engine, 40)

    assert stats.branch_accepted, "no branch telemetry recorded"
    assert all(len(row) == 6 for row in stats.branch_accepted)
    assert stats.mean_best_branch >= stats.mean_single_branch
    assert stats.branch_gain > 0.0, (
        "taking the max over 6 branches bought nothing: "
        f"best {stats.mean_best_branch:.2f} vs single {stats.mean_single_branch:.2f}"
    )


def test_single_branch_has_no_gain(draft_model, target_model, tokenizer):
    """With B=1 the max and the mean coincide, so the gain must be exactly zero.

    A non-zero gain here would mean the telemetry is measuring something other
    than what it claims.
    """
    torch.manual_seed(6)
    engine = MonteCarloEngine(draft_model, target_model, tokenizer, k=5, branches=1)
    _, stats = _ids(engine, 20)
    assert stats.branch_gain == pytest.approx(0.0)
    assert set(stats.branch_wins) == {0}


def test_wins_are_spread_across_branches(draft_model, target_model, tokenizer):
    """No single branch should dominate; that would indicate broken sampling."""
    torch.manual_seed(7)
    engine = MonteCarloEngine(
        draft_model, target_model, tokenizer, k=5, branches=4, draft_temperature=1.5
    )
    _, stats = _ids(engine, 60)
    assert len(stats.branch_win_spread) > 1, (
        f"only one branch ever won: {stats.branch_win_spread}"
    )
    assert "best branch" in stats.branch_summary()


# ---------------------------------------------------------------- bookkeeping
def test_max_new_tokens_is_exact(draft_model, target_model, tokenizer):
    torch.manual_seed(8)
    engine = MonteCarloEngine(draft_model, target_model, tokenizer, k=5, branches=4)
    for n in (1, 2, 5, 6, 7, 13):
        text, stats = engine.generate(PROMPT, max_new_tokens=n, stream=False)
        assert stats.tokens_generated == n and len(text) == n


def test_caches_return_to_batch_one_each_iteration(draft_model, target_model, tokenizer):
    """The master caches must be collapsed, or memory grows by B every iteration."""
    torch.manual_seed(9)
    engine = MonteCarloEngine(draft_model, target_model, tokenizer, k=3, branches=4)

    seen = []
    original = engine._draft_branches

    def spy(context, cache, k):
        if cache is not None:
            seen.append(cache.layers[0].keys.shape[0])
        return original(context, cache, k)

    engine._draft_branches = spy
    engine.generate(PROMPT, max_new_tokens=30, stream=False)

    assert len(seen) > 4
    assert set(seen) == {1}, f"cache handed to draft phase was batched: {seen}"


def test_forward_accounting(draft_model, target_model, tokenizer):
    """One batched target forward per iteration, k batched draft forwards."""
    torch.manual_seed(10)
    engine = MonteCarloEngine(draft_model, target_model, tokenizer, k=5, branches=4)
    _, stats = _ids(engine, 24)
    assert stats.target_forwards == stats.iterations
    assert stats.draft_forwards == sum(stats.draft_lengths)


def test_eos_stops_cleanly(draft_model, target_model, tokenizer):
    torch.manual_seed(11)
    engine = MonteCarloEngine(draft_model, target_model, tokenizer, k=5, branches=4)
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
    engine = MonteCarloEngine(draft_model, target_model, tokenizer)
    engine.use_dual_gate = True
    with pytest.raises(NotImplementedError):
        engine.generate(PROMPT, max_new_tokens=4, stream=False)


# ---------------------------------------------------------------- cache helpers
def _dummy_cache(length=5, batch=1):
    from transformers import DynamicCache
    cache = DynamicCache()
    for layer in range(2):
        keys = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1).expand(
            batch, 2, length, 4
        ).contiguous()
        cache.update(keys, keys.clone(), layer)
    return cache


def test_expand_then_select_round_trips():
    cache = _dummy_cache(length=5)
    assert cache.layers[0].keys.shape[0] == 1

    cache = _expand_cache(cache, 4)
    assert cache.layers[0].keys.shape[0] == 4
    assert _cache_length(cache) == 5, "expanding must not change sequence length"

    # Mark row 2 so selection is observable.
    cache.layers[0].keys[2] += 99.0
    cache = _select_cache_row(cache, 2)
    assert cache.layers[0].keys.shape[0] == 1
    assert float(cache.layers[0].keys[0, 0, 0, 0]) == 99.0, "selected the wrong row"


def test_expand_is_a_noop_for_one_or_none():
    assert _expand_cache(None, 4) is None
    cache = _dummy_cache()
    assert _expand_cache(cache, 1) is cache
    assert _select_cache_row(None, 0) is None
