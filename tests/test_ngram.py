"""Tests for the n-gram fast-path drafter.

Two things are load-bearing. The matcher must return exactly the tokens that
followed the most recent earlier occurrence of the pattern -- an off-by-one here
proposes plausible-looking garbage that verification silently absorbs, costing
throughput invisibly. And the one-hot `q` must make the verifier emit the target's
distribution exactly, which is checked statistically rather than argued.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ngram import find_ngram_draft, one_hot_q  # noqa: E402
from core.verifier import verify_tokens  # noqa: E402


def t(*ids):
    return torch.tensor(list(ids), dtype=torch.long)


# ---------------------------------------------------------------- matching
def test_finds_the_continuation_of_a_repeated_pattern():
    #                0  1  2  3  4  5  6  7   pattern = (1, 2) at the end
    context = t(1, 2, 7, 8, 9, 5, 1, 2)
    draft = find_ngram_draft(context, max_draft_len=3, min_match_len=2)
    assert draft is not None
    assert draft.tolist() == [7, 8, 9], "must return what followed the match"


def test_prefers_the_most_recent_match():
    """Recency is the better predictor, and the docstring promises it."""
    #            0  1  2  3  4  5  6  7  8  9      (1,2) at 0 -> 3; at 5 -> 99
    context = t(1, 2, 3, 3, 3, 1, 2, 99, 98, 1, 2)
    draft = find_ngram_draft(context, max_draft_len=2, min_match_len=2)
    assert draft.tolist() == [99, 98], "should use the later occurrence"


def test_truncates_to_available_tokens():
    """Fewer than max_draft_len tokens follow the match, so return what there is.

    Note the proposal runs past the match into the context's own tail: after the
    (4,5) at index 0 the context continues 6,4,5, and all three are legitimate
    predictions of what comes next. Overlapping the tail is not a bug -- it is
    exactly the repetition prompt-lookup exists to exploit.
    """
    context = t(4, 5, 6, 4, 5)
    draft = find_ngram_draft(context, max_draft_len=10, min_match_len=2)
    assert draft.tolist() == [6, 4, 5], "three tokens follow the match at index 0"
    assert draft.numel() < 10, "and that is fewer than the cap"


def test_respects_max_draft_len():
    context = t(1, 2, 3, 4, 5, 6, 7, 8, 1, 2)
    assert find_ngram_draft(context, max_draft_len=3, min_match_len=2).tolist() == [3, 4, 5]
    assert find_ngram_draft(context, max_draft_len=1, min_match_len=2).tolist() == [3]


@pytest.mark.parametrize("min_match_len", [1, 2, 3, 4])
def test_longer_patterns_match_more_specifically(min_match_len):
    context = t(9, 1, 2, 3, 4, 50, 60, 8, 1, 2, 3, 4)
    draft = find_ngram_draft(context, max_draft_len=2, min_match_len=min_match_len)
    assert draft is not None
    # Whatever the pattern length, the continuation of the matched occurrence is
    # 50, 60 -- so a correct implementation agrees across all of them here.
    assert draft.tolist() == [50, 60], f"min_match_len={min_match_len}"


def test_returns_none_when_the_pattern_never_recurs():
    assert find_ngram_draft(t(1, 2, 3, 4, 5, 6), max_draft_len=3, min_match_len=2) is None


def test_returns_none_for_short_contexts():
    for context in (t(), t(1), t(1, 2), t(1, 2, 3)):
        assert find_ngram_draft(context, max_draft_len=3, min_match_len=2) is None


def test_never_returns_an_empty_draft():
    """verify_tokens rejects an empty block, so 'nothing' must be None.

    The dangerous case is a pattern whose only earlier occurrence sits flush
    against the end, leaving no following token to propose.
    """
    for context in (t(1, 2, 1, 2), t(5, 5, 5), t(3, 3, 3, 3)):
        draft = find_ngram_draft(context, max_draft_len=4, min_match_len=2)
        assert draft is None or draft.numel() > 0


def test_accepts_batched_shape():
    context = torch.tensor([[1, 2, 7, 8, 1, 2]], dtype=torch.long)
    assert find_ngram_draft(context, max_draft_len=2, min_match_len=2).tolist() == [7, 8]


def test_result_does_not_alias_the_context():
    """A view would let a later context mutation rewrite an in-flight draft."""
    context = t(1, 2, 7, 8, 1, 2)
    draft = find_ngram_draft(context, max_draft_len=2, min_match_len=2)
    context[2] = 999
    assert draft.tolist() == [7, 8]


def test_single_token_pattern_still_works():
    context = t(4, 77, 88, 9, 4)
    assert find_ngram_draft(context, max_draft_len=2, min_match_len=1).tolist() == [77, 88]


def test_invalid_arguments():
    for kwargs in ({"min_match_len": 0}, {"max_draft_len": 0}):
        with pytest.raises(ValueError):
            find_ngram_draft(t(1, 2, 1, 2), **kwargs)


def test_realistic_repetition_is_matched():
    """The case the fast path exists for: code repeating its own identifiers."""
    # "... x for x in arr if x <" then later "... x for x in arr if x"
    context = t(10, 11, 12, 13, 14, 15, 20, 21, 10, 11, 12, 13, 14, 15)
    draft = find_ngram_draft(context, max_draft_len=2, min_match_len=6)
    assert draft.tolist() == [20, 21]


# ---------------------------------------------------------------- one-hot q
def test_one_hot_q_shape_and_mass():
    q = one_hot_q(t(3, 0, 7), vocab_size=8)
    assert q.shape == (3, 8)
    assert torch.equal(q.sum(-1), torch.ones(3))
    assert q[0, 3] == 1.0 and q[1, 0] == 1.0 and q[2, 7] == 1.0


def test_deterministic_draft_still_emits_the_target_distribution():
    """10,000 runs: a one-hot q must leave the output distributed exactly as p.

    This is the claim that licenses the whole fast path. If it failed, the n-gram
    router would be biasing generation rather than accelerating it -- and the bias
    would be invisible in any single sample.
    """
    torch.manual_seed(4242)
    target = torch.tensor(
        [[0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02]], dtype=torch.float64
    )
    vocab = target.shape[1]
    n_runs = 10_000

    # A deliberately mediocre deterministic guess: the target's 4th-likeliest.
    guess = t(3)
    q = one_hot_q(guess, vocab, dtype=torch.float64)

    counts = torch.zeros(vocab, dtype=torch.long)
    generator = torch.Generator().manual_seed(99)
    for _ in range(n_runs):
        emitted, _ = verify_tokens(q, target, guess, generator=generator)
        counts[int(emitted[0])] += 1

    empirical = counts.double() / n_runs
    expected = target[0]
    sd = torch.sqrt(expected * (1 - expected) / n_runs).clamp_min(1e-12)
    z = (empirical - expected) / sd
    worst = int(z.abs().argmax())
    assert z.abs().max() < 4.5, (
        "one-hot q biases the output: token {} z={:.2f} (empirical {:.4f} vs "
        "target {:.4f})\n  empirical: {}\n  expected : {}".format(
            worst, float(z[worst]), float(empirical[worst]), float(expected[worst]),
            [round(v, 4) for v in empirical.tolist()],
            [round(v, 4) for v in expected.tolist()],
        )
    )


def test_deterministic_draft_acceptance_equals_its_probability():
    """A one-hot q is accepted with probability exactly p(x).

    So the n-gram path's acceptance rate is a direct readout of how much
    probability the target assigned to the guess -- which is why a hit rate and an
    acceptance rate are different numbers and both need reporting.
    """
    torch.manual_seed(7)
    target = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
    guess = t(1)                       # target gives this 0.2
    q = one_hot_q(guess, 3, dtype=torch.float64)

    generator = torch.Generator().manual_seed(11)
    accepted = sum(
        verify_tokens(q, target, guess, generator=generator)[1] for _ in range(10_000)
    )
    rate = accepted / 10_000
    assert abs(rate - 0.2) < 0.02, f"acceptance {rate:.3f} should track p(x)=0.2"


def test_greedy_deterministic_draft_accepts_iff_it_matches():
    """At temperature 0 the target's p is one-hot, so the guess must match exactly."""
    for guess_id, expect_accept in ((2, True), (0, False)):
        target = torch.zeros(1, 4, dtype=torch.float64)
        target[0, 2] = 1.0
        guess = t(guess_id)
        q = one_hot_q(guess, 4, dtype=torch.float64)
        emitted, n_accepted = verify_tokens(q, target, guess)
        assert bool(n_accepted) is expect_accept
        assert int(emitted[-1]) == 2, "either way the target's own token is emitted"
