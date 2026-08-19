"""Tests for Jacobi (fixed-point relaxation) decoding.

This engine is exact by construction rather than in distribution, so the equivalence
test is the whole correctness story: output must equal plain greedy decoding for any
block size, iteration cap, or initial guess.

The other thing worth testing hard is *progress*. The natural commit rule -- the
longest common prefix between consecutive sweeps -- can commit zero tokens when the
guess is wrong at position 1, which would hang the loop rather than produce a wrong
answer. The iteration-count floor is what prevents that, and it is tested directly.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.jacobi_engine import JacobiEngine, longest_common_prefix  # noqa: E402
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


def _ids(engine, n=24):
    text, stats = engine.generate(PROMPT, max_new_tokens=n, stream=False)
    return [ord(c) - ord("a") for c in text], stats


# ---------------------------------------------------------------- exactness
@pytest.mark.parametrize("block_size", [1, 2, 5, 10, 16])
def test_output_matches_plain_greedy(model, tokenizer, block_size):
    """Relaxation must land on exactly the autoregressive greedy sequence."""
    engine = JacobiEngine(model, tokenizer, block_size=block_size, max_iterations=block_size)
    expected = reference_greedy(model, tokenizer(PROMPT).input_ids, 24)
    actual, _ = _ids(engine)
    assert actual == expected, (
        "Jacobi diverged from greedy at block_size={}\n  expected: {}\n  actual  : {}"
        .format(block_size, expected, actual)
    )


@pytest.mark.parametrize("max_iterations", [1, 2, 3, 5, 10])
def test_iteration_cap_changes_speed_not_output(model, tokenizer, max_iterations):
    """Stopping early may commit fewer tokens per block; it cannot commit wrong ones.

    A cap of 1 is the extreme: one sweep, one guaranteed token, exactly
    autoregressive cost. The output must be identical to a cap of 10.
    """
    engine = JacobiEngine(model, tokenizer, block_size=10, max_iterations=max_iterations)
    expected = reference_greedy(model, tokenizer(PROMPT).input_ids, 24)
    assert _ids(engine)[0] == expected, f"max_iterations={max_iterations}"


@pytest.mark.parametrize("min_match_len", [1, 2, 5, 500])
def test_initial_guess_quality_does_not_affect_output(model, tokenizer, min_match_len):
    """The seed only affects convergence speed. min_match_len=500 never matches, so
    every block starts from the deliberately poor repeated-token guess."""
    engine = JacobiEngine(model, tokenizer, block_size=8, max_iterations=8,
                          min_match_len=min_match_len)
    expected = reference_greedy(model, tokenizer(PROMPT).input_ids, 24)
    assert _ids(engine)[0] == expected, f"min_match_len={min_match_len}"


def test_temperature_above_zero_is_refused(model, tokenizer):
    """The fixed-point argument is argmax-specific; sampling needs more machinery."""
    with pytest.raises(ValueError, match="temperature=0"):
        JacobiEngine(model, tokenizer, temperature=0.7)


def test_invalid_arguments(model, tokenizer):
    for kwargs, match in (
        ({"block_size": 0}, "block_size"),
        ({"max_iterations": 0}, "max_iterations"),
    ):
        with pytest.raises(ValueError, match=match):
            JacobiEngine(model, tokenizer, **kwargs)


def test_no_second_model_is_held(model, tokenizer):
    """Jacobi needs no drafter, so it runs at plain-decoding VRAM."""
    engine = JacobiEngine(model, tokenizer)
    assert engine.draft_model is engine.target_model is model


# ---------------------------------------------------------------- progress
def test_progress_is_guaranteed_even_with_a_useless_guess(model, tokenizer):
    """The liveness property. A guess that is wrong at position 1 gives an empty
    common prefix, so a prefix-only commit rule would commit nothing and hang.

    min_match_len=500 guarantees the n-gram lookup always misses, so every block
    starts from the repeated-token guess -- the worst realistic seed.
    """
    engine = JacobiEngine(model, tokenizer, block_size=10, max_iterations=3,
                          min_match_len=500)
    _, stats = _ids(engine, 30)
    assert stats.tokens_generated == 30
    assert all(n >= 1 for n in stats.accepted_per_iteration), (
        f"a block committed nothing: {stats.accepted_per_iteration}"
    )


def test_commits_at_least_the_iteration_count(model, tokenizer):
    """After k sweeps the first k tokens are provably correct, so k is a valid floor."""
    engine = JacobiEngine(model, tokenizer, block_size=10, max_iterations=4,
                          min_match_len=500)
    _, stats = _ids(engine, 40)
    for committed, sweeps in zip(stats.accepted_per_iteration, stats.jacobi_iterations):
        assert committed >= min(sweeps, 10), (
            f"committed {committed} after {sweeps} sweeps, below the guaranteed floor"
        )


def test_single_iteration_is_at_least_autoregressive(model, tokenizer):
    """With one sweep per block, Jacobi is at worst one token per forward.

    Autoregressive parity is the *floor*, not the value. Even a repeated-token guess
    sometimes matches the first few predictions, and the common-prefix rule commits
    those too -- so a single sweep can already beat one token per forward. That floor
    is why the method can never be slower in forward count than the decoder it
    replaces.
    """
    engine = JacobiEngine(model, tokenizer, block_size=10, max_iterations=1,
                          min_match_len=500)
    _, stats = _ids(engine, 20)
    assert stats.target_forwards == len(stats.jacobi_iterations)
    assert all(n == 1 for n in stats.jacobi_iterations)
    assert stats.tokens_generated / stats.target_forwards >= 1.0


# ---------------------------------------------------------------- convergence
def test_perfect_guess_converges_in_one_sweep(model, tokenizer):
    """Seeded with the answer, the first sweep is already a fixed point.

    This is the best case and it validates the fixed-point detection: a converged
    block commits all B tokens plus the bonus prediction beyond it.
    """
    engine = JacobiEngine(model, tokenizer, block_size=6, max_iterations=6)
    context = tokenizer(PROMPT).input_ids.to(engine.device)
    truth = torch.tensor(
        reference_greedy(model, tokenizer(PROMPT).input_ids, 6), device=engine.device
    )

    new_block, bonus, _ = engine._relax(context, truth, None)
    assert torch.equal(new_block, truth), "the greedy block must be a fixed point"
    assert longest_common_prefix(new_block, truth) == 6
    # The bonus slot is the greedy token following a now-verified block.
    extended = reference_greedy(model, tokenizer(PROMPT).input_ids, 7)
    assert int(bonus) == extended[6]


def test_first_sweep_always_fixes_position_one(model, tokenizer):
    """Position 1 conditions only on the true context, so it is right immediately --
    the base case of the whole induction."""
    engine = JacobiEngine(model, tokenizer, block_size=8, max_iterations=8)
    context = tokenizer(PROMPT).input_ids.to(engine.device)
    truth = reference_greedy(model, tokenizer(PROMPT).input_ids, 8)

    garbage = torch.full((8,), 3, dtype=torch.long, device=engine.device)
    new_block, _, _ = engine._relax(context, garbage, None)
    assert int(new_block[0]) == truth[0]


def test_convergence_telemetry_is_recorded(model, tokenizer):
    engine = JacobiEngine(model, tokenizer, block_size=10, max_iterations=5)
    _, stats = _ids(engine, 40)
    assert len(stats.jacobi_iterations) == stats.iterations
    assert stats.target_forwards == sum(stats.jacobi_iterations)
    assert 1 <= stats.mean_jacobi_iterations <= 5
    assert 0.0 <= stats.jacobi_convergence_rate <= 1.0
    # Guess-source accounting covers every block.
    assert stats.ngram_drafts_used + stats.model_drafts_used == stats.iterations


def test_max_new_tokens_is_exact(model, tokenizer):
    engine = JacobiEngine(model, tokenizer, block_size=10, max_iterations=5)
    for n in (1, 2, 7, 11, 23):
        text, stats = engine.generate(PROMPT, max_new_tokens=n, stream=False)
        assert stats.tokens_generated == n and len(text) == n


def test_eos_stops_cleanly(model, tokenizer):
    engine = JacobiEngine(model, tokenizer, block_size=10, max_iterations=5)
    reference = reference_greedy(model, tokenizer(PROMPT).input_ids, 20)
    eos_id = reference[4]
    expected_length = reference.index(eos_id)

    class EosTokenizer(CharTokenizer):
        eos_token_id = eos_id

    engine.tokenizer = EosTokenizer()
    text, stats = engine.generate(PROMPT, max_new_tokens=20, stream=False)
    assert stats.tokens_generated == expected_length
    assert eos_id not in stats.token_ids


# ---------------------------------------------------------------- helper
@pytest.mark.parametrize("a, b, expected", [
    ([1, 2, 3], [1, 2, 3], 3),
    ([1, 2, 3], [1, 2, 9], 2),
    ([1, 2, 3], [9, 2, 3], 0),
    ([1, 2, 3], [1, 2], 2),
    ([], [1], 0),
    ([5], [5], 1),
])
def test_longest_common_prefix(a, b, expected):
    assert longest_common_prefix(
        torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)
    ) == expected
