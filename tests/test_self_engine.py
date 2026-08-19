"""Tests for single-model self-speculative decoding (Twin-Cache).

The load-bearing concern is RoPE positioning. The draft cache holds a *suffix* of
history, so its indices and the tokens' true positions diverge. Getting that wrong
does not raise -- it silently ruins the draft, showing up only as an acceptance
rate near zero. So the tests assert on acceptance, not just on output validity:
correctness is guaranteed by verification regardless, which means a positioning
bug would pass any output-only test.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import _cache_length  # noqa: E402
from core.self_engine import SelfSpeculativeEngine, _evict_to_window  # noqa: E402
from tests.test_engine import (  # noqa: E402
    PROMPT,
    VOCAB,
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


def _ids(engine, prompt, n):
    text, stats = engine.generate(prompt, max_new_tokens=n, stream=False)
    return [ord(c) - ord("a") for c in text], stats


# ---------------------------------------------------------------- correctness
@pytest.mark.parametrize("window", [1, 2, 4, 8, 64, 4096])
def test_output_always_matches_plain_greedy(model, tokenizer, window):
    """Verification runs on the full cache, so output is the target's own.

    Swept from a degenerate 1-token window to one larger than the whole context.
    However lossy the draft, the emitted tokens must be identical to plain greedy
    decoding -- that is the entire point of verifying against the full history.
    """
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=5, draft_window_size=window, temperature=0.0
    )
    expected = reference_greedy(model, tokenizer(PROMPT).input_ids, 24)
    actual, _ = _ids(engine, PROMPT, 24)
    assert actual == expected, (
        "self-speculation changed the output at window={}\n"
        "  expected: {}\n  actual  : {}".format(window, expected, actual)
    )


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
def test_output_matches_greedy_across_k(model, tokenizer, k):
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=k, draft_window_size=16, temperature=0.0
    )
    expected = reference_greedy(model, tokenizer(PROMPT).input_ids, 20)
    assert _ids(engine, PROMPT, 20)[0] == expected


def test_unbounded_window_accepts_everything(model, tokenizer):
    """With a window larger than the context, draft and target are identical.

    Nothing can be rejected, which is the degenerate case: perfect acceptance and
    no speculation benefit whatsoever, since each drafted token cost a full
    forward of the same model that then verifies it.
    """
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=5, draft_window_size=100_000, temperature=0.0
    )
    _, stats = _ids(engine, PROMPT, 24)
    assert stats.acceptance_rate == 1.0, stats.summary()


def test_draft_receives_true_absolute_positions(model, tokenizer):
    """The RoPE regression test, asserted white-box on the positions themselves.

    Cached keys carry rotations from their original absolute positions and
    eviction does not re-rotate them, so a new query must also carry its true
    absolute position or every relative offset RoPE encodes comes out wrong.

    Checked directly rather than via acceptance rate. Two reasons: the bug cannot
    show up in the *output* at all, since verification against the full cache
    cleans up after any draft; and on a randomly-initialised model, truncating
    context legitimately moves predictions a lot, so an acceptance threshold
    cannot separate "positions are wrong" from "the draft is simply weak".
    """
    window = 6
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=4, draft_window_size=window, temperature=0.0
    )
    prompt_len = tokenizer(PROMPT).input_ids.shape[1]

    blocks = []          # (context_len, cache_start, cache_len) per draft phase
    positions = []       # position_ids of every draft forward, per block
    original_draft = engine._draft_windowed
    original_forward = engine._forward

    def draft_spy(context, cache, cache_start, k):
        blocks.append((context.shape[1], cache_start, _cache_length(cache)))
        positions.append([])
        return original_draft(context, cache, cache_start, k)

    def forward_spy(m, input_ids, cache, position_ids=None):
        if position_ids is not None and positions:
            positions[-1].append(position_ids.flatten().tolist())
        return original_forward(m, input_ids, cache, position_ids)

    engine._draft_windowed = draft_spy
    engine._forward = forward_spy
    engine.generate(PROMPT, max_new_tokens=32, stream=False)

    assert len(blocks) > 6, "not enough draft phases observed"

    for i, (context_len, cache_start, cache_len) in enumerate(blocks):
        covered = cache_start + cache_len
        assert covered <= context_len, (
            f"block {i}: draft cache claims to cover position {covered} of a "
            f"{context_len}-token context"
        )
        assert cache_len <= window, f"block {i}: cache {cache_len} exceeds window"

        # The block's first forward must resume at the true absolute position
        # just past what the cache covers -- not at the cache index.
        first = positions[i][0]
        assert first[0] == covered, (
            f"block {i}: drafting resumed at position {first[0]} but the cache "
            f"covers up to {covered}; positions are being numbered by cache index"
        )

        # Within a block positions advance contiguously. Across blocks they may
        # rewind, which is correct: rejected speculation is discarded and
        # redrafted from the accepted point.
        flat = [p for forward in positions[i] for p in forward]
        assert flat == list(range(flat[0], flat[0] + len(flat))), flat

    assert positions[0][0] == list(range(prompt_len)), "first forward is the prefill"

    # Eviction must actually make the two numberings diverge, or this test would
    # pass just as well against a cache-index implementation.
    assert any(start > 0 for _, start, _ in blocks), (
        "the window never slid, so absolute and cache-relative numbering never "
        "differed and this test proves nothing"
    )


def test_acceptance_is_nonzero_with_an_active_window(model, tokenizer):
    """Sanity floor only. A random tiny model makes a genuinely poor windowed
    draft, so this deliberately does not assert a high rate -- the real-model
    acceptance is measured in the benchmark, where the number means something."""
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=4, draft_window_size=6, temperature=0.0
    )
    _, stats = _ids(engine, PROMPT, 40)
    assert stats.acceptance_rate > 0.0


def test_tiny_window_still_drafts_something_useful(model, tokenizer):
    """Even a 1-token window is a legal, if very weak, draft."""
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=3, draft_window_size=1, temperature=0.0
    )
    _, stats = _ids(engine, PROMPT, 18)
    assert stats.tokens_generated == 18
    assert 0.0 <= stats.acceptance_rate <= 1.0


# ---------------------------------------------------------------- accounting
def test_forward_accounting_reflects_the_real_cost(model, tokenizer):
    """One verification forward per iteration, k draft forwards per iteration.

    Worth asserting because it is the crux of the throughput argument: the draft
    forwards are full forwards of the same model, so they are not free the way a
    1.5B draft is.
    """
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=5, draft_window_size=16, temperature=0.0
    )
    _, stats = _ids(engine, PROMPT, 24)
    assert stats.target_forwards == stats.iterations
    assert stats.draft_forwards == sum(stats.draft_lengths)
    # The crux: drafting costs many forwards per verification forward, and here
    # every one of them is a full forward of the 7B-equivalent target.
    assert stats.draft_forwards > 3 * stats.target_forwards


def test_max_new_tokens_is_exact(model, tokenizer):
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=5, draft_window_size=16, temperature=0.0
    )
    for n in (1, 2, 5, 6, 7, 13):
        text, stats = engine.generate(PROMPT, max_new_tokens=n, stream=False)
        assert stats.tokens_generated == n and len(text) == n


def test_no_second_model_is_held(model, tokenizer):
    """The point of the exercise: draft and target are the same object."""
    engine = SelfSpeculativeEngine(model, tokenizer)
    assert engine.draft_model is engine.target_model is model


def test_streaming_matches_returned_text(model, tokenizer, capsys):
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=5, draft_window_size=16, temperature=0.0
    )
    text, _ = engine.generate(PROMPT, max_new_tokens=18, stream=True)
    assert capsys.readouterr().out.strip() == text


def test_eos_stops_cleanly(model, tokenizer):
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=5, draft_window_size=16, temperature=0.0
    )
    reference = reference_greedy(model, tokenizer(PROMPT).input_ids, 20)
    eos_id = reference[4]
    expected_length = reference.index(eos_id)

    class EosTokenizer(CharTokenizer):
        eos_token_id = eos_id

    engine.tokenizer = EosTokenizer()
    text, stats = engine.generate(PROMPT, max_new_tokens=20, stream=False)
    assert stats.tokens_generated == expected_length
    assert eos_id not in stats.token_ids


def test_dual_gate_is_refused_rather_than_ignored(model, tokenizer):
    """Silently ignoring a requested feature would be worse than failing."""
    engine = SelfSpeculativeEngine(model, tokenizer)
    engine.use_dual_gate = True
    with pytest.raises(NotImplementedError):
        engine.generate(PROMPT, max_new_tokens=4, stream=False)


def test_invalid_window(model, tokenizer):
    for bad in (0, -1):
        with pytest.raises(ValueError):
            SelfSpeculativeEngine(model, tokenizer, draft_window_size=bad)


# ---------------------------------------------------------------- eviction
def _dummy_cache(length: int, layers: int = 2):
    from transformers import DynamicCache
    cache = DynamicCache()
    for layer in range(layers):
        keys = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1).expand(1, 2, length, 4)
        cache.update(keys.contiguous(), keys.clone().contiguous(), layer)
    return cache


def test_evict_drops_the_oldest_entries():
    """Windowing must drop the *front*; rollback drops the back."""
    cache = _dummy_cache(10)
    assert _cache_length(cache) == 10
    cache = _evict_to_window(cache, 4)
    assert _cache_length(cache) == 4
    # Values were seeded with their position index, so the survivors identify
    # themselves: the last four, not the first four.
    kept = cache.layers[0].keys[0, 0, :, 0].tolist()
    assert kept == [6.0, 7.0, 8.0, 9.0], kept


def test_evict_is_a_noop_when_within_the_window():
    cache = _evict_to_window(_dummy_cache(3), 8)
    assert _cache_length(cache) == 3
    assert _evict_to_window(None, 8) is None


def test_evicted_slices_are_contiguous_not_views():
    """A view keeps the original tensor alive, so the cache would never shrink.

    The bounded-memory property is the only reason to window at all; if eviction
    left views behind it would be silently false.
    """
    cache = _evict_to_window(_dummy_cache(64), 8)
    keys = cache.layers[0].keys
    assert keys.is_contiguous()
    assert keys.untyped_storage().size() == keys.numel() * keys.element_size(), (
        "slice still references the original allocation"
    )


def test_eviction_keeps_the_window_bounded_over_a_long_run(model, tokenizer):
    """The draft cache must stay bounded across many iterations.

    Asserted by observing the cache the draft phase is actually handed, since a
    steadily growing draft cache would defeat the entire design while still
    producing correct output.
    """
    window = 6
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=3, draft_window_size=window, temperature=0.0
    )

    seen = []
    original = engine._draft_windowed

    def spy(context, cache, cache_start, k):
        seen.append(_cache_length(cache))
        return original(context, cache, cache_start, k)

    engine._draft_windowed = spy
    engine.generate(PROMPT, max_new_tokens=48, stream=False)

    assert len(seen) > 5
    assert max(seen) <= window, f"draft cache grew past the window: {seen}"
    # And it should actually reach the window, or the test proves nothing.
    assert max(seen) == window, f"window never filled: {seen}"


def test_full_cache_keeps_everything(model, tokenizer):
    """The verification cache must NOT be windowed -- it is the authority."""
    engine = SelfSpeculativeEngine(
        model, tokenizer, k=3, draft_window_size=4, temperature=0.0
    )
    seen = []
    original = engine._verify_full

    def spy(context, draft_tokens, cache):
        seen.append((context.shape[1], _cache_length(cache)))
        return original(context, draft_tokens, cache)

    engine._verify_full = spy
    engine.generate(PROMPT, max_new_tokens=36, stream=False)

    for i, (context_len, cache_len) in enumerate(seen):
        expected = 0 if i == 0 else context_len - 1
        assert cache_len == expected, (
            f"iteration {i}: full cache holds {cache_len}, expected {expected} "
            "-- it must retain the whole history, not a window"
        )
