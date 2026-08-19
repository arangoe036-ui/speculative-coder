"""Statistical validation of the modified rejection sampling verifier.

The central claim of Leviathan et al. (2023) is a distributional identity, not
an approximation: tokens emitted by speculative decoding are distributed
*exactly* as if they had been sampled from the target model, whatever draft
model produced them.  These tests check that identity by Monte Carlo -- 10,000
independent runs per experiment -- rather than by inspecting the code path.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.verifier import verify_tokens  # noqa: E402

N_RUNS = 10_000
SEED = 20230101
DTYPE = torch.float64

# Upper-tail chi-square critical values at alpha = 0.001, indexed by degrees of
# freedom.  A tight alpha keeps the suite deterministically seeded *and* far
# from the flake boundary while still rejecting any real distributional bug.
CHI2_CRIT_001 = {1: 10.828, 2: 13.816, 3: 16.266, 4: 18.467,
                 5: 20.515, 6: 22.458, 7: 24.322, 8: 26.125, 9: 27.877}

# Per-category z-score bound.  |z| < 4.5 is a two-sided p of about 7e-6 per
# category, i.e. Bonferroni-safe across a vocabulary of this size.
Z_BOUND = 4.5


# --------------------------------------------------------------------------
# Dummy distributions.  The draft is deliberately a *bad* approximation of the
# target (total variation about 0.36) so that both the accept path and the
# residual resample path are exercised heavily.
# --------------------------------------------------------------------------
TARGET = torch.tensor(
    [
        [0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02],
        [0.05, 0.05, 0.40, 0.20, 0.10, 0.10, 0.05, 0.05],
        [0.12, 0.13, 0.12, 0.13, 0.12, 0.13, 0.12, 0.13],
        [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.25, 0.30],  # bonus-token row
    ],
    dtype=DTYPE,
)
DRAFT = torch.tensor(
    [
        [0.10, 0.30, 0.05, 0.25, 0.02, 0.20, 0.06, 0.02],
        [0.20, 0.20, 0.10, 0.10, 0.20, 0.05, 0.10, 0.05],
        [0.40, 0.30, 0.10, 0.05, 0.05, 0.05, 0.04, 0.01],
    ],
    dtype=DTYPE,
)
VOCAB = TARGET.shape[1]
GAMMA = DRAFT.shape[0]


def _check_rows_normalized() -> None:
    assert torch.allclose(TARGET.sum(-1), torch.ones(TARGET.shape[0], dtype=DTYPE))
    assert torch.allclose(DRAFT.sum(-1), torch.ones(DRAFT.shape[0], dtype=DTYPE))


def assert_matches_distribution(counts, expected, label) -> None:
    """Goodness of fit: per-category z-test, Pearson chi-square, TV distance."""
    n = int(counts.sum())
    assert n > 500, "{}: only {} samples, too few to test".format(label, n)

    expected = expected.to(DTYPE)
    observed = counts.to(DTYPE)
    expected_counts = expected * n
    empirical = observed / n

    # 1. Per-category normal-approximation z-scores.
    sd = torch.sqrt(expected * (1 - expected) * n)
    z = (observed - expected_counts) / sd.clamp_min(1e-12)
    worst = int(z.abs().argmax())
    assert z.abs().max() < Z_BOUND, (
        "{}: category {} off by z={:.2f} (observed {:.4f} vs expected {:.4f}, n={})\n"
        "  empirical: {}\n  expected : {}".format(
            label, worst, float(z[worst]), float(empirical[worst]),
            float(expected[worst]), n,
            [round(v, 4) for v in empirical.tolist()],
            [round(v, 4) for v in expected.tolist()],
        )
    )

    # 2. Pearson chi-square goodness of fit over the whole support.
    chi2 = float((((observed - expected_counts) ** 2) / expected_counts.clamp_min(1e-12)).sum())
    df = int((expected > 0).sum()) - 1
    crit = CHI2_CRIT_001[df]
    assert chi2 < crit, "{}: chi2={:.2f} exceeds critical {} at df={}, n={}".format(
        label, chi2, crit, df, n
    )

    # 3. Total variation distance, as a scale-free sanity bound.
    tv = 0.5 * float((empirical - expected).abs().sum())
    assert tv < 0.03, "{}: total variation {:.4f} too large (n={})".format(label, tv, n)


def _run_trials(draft_probs, target_probs, n_runs=N_RUNS, seed=SEED):
    """Run the verifier n_runs times, drafting tokens honestly from draft_probs."""
    gen = torch.Generator().manual_seed(seed)
    gamma = draft_probs.shape[0]
    outputs, accepts = [], []
    for _ in range(n_runs):
        # One token per row, sampled from that row -- exactly what a draft model
        # does when it autoregressively proposes a block.
        draft_tokens = torch.multinomial(draft_probs, 1, generator=gen).squeeze(1)
        assert draft_tokens.shape == (gamma,)
        tokens, n_accepted = verify_tokens(
            draft_probs, target_probs, draft_tokens, generator=gen
        )
        outputs.append(tokens)
        accepts.append(n_accepted)
    return outputs, accepts


def _counts_at(outputs, position):
    counts = torch.zeros(VOCAB, dtype=torch.long)
    for tokens in outputs:
        if tokens.shape[0] > position:
            counts[int(tokens[position])] += 1
    return counts


# ==========================================================================
# The core guarantee
# ==========================================================================
def test_emitted_distribution_matches_target():
    """10,000 runs: the emitted token distribution must equal the target's."""
    _check_rows_normalized()
    outputs, accepts = _run_trials(DRAFT, TARGET[:GAMMA])

    assert len(outputs) == N_RUNS
    # Every run emits n_accepted tokens plus one resample, unless the whole
    # block was accepted (no bonus row is supplied here).
    for tokens, n in zip(outputs, accepts):
        assert tokens.shape[0] == (n if n == GAMMA else n + 1)

    # Position 0 is reached by every run, so its marginal must be exactly p_0.
    assert int(_counts_at(outputs, 0).sum()) == N_RUNS
    assert_matches_distribution(_counts_at(outputs, 0), TARGET[0], "position 0")

    # Later positions are only reached when the earlier tokens were accepted.
    # Because each draft token is proposed independently, conditioning on
    # reaching position i leaves the emitted token at position i distributed as
    # p_i, so the same equality must hold on that sub-population.
    for position in range(1, GAMMA):
        assert_matches_distribution(
            _counts_at(outputs, position), TARGET[position],
            "position {}".format(position),
        )


def test_acceptance_rate_matches_theory():
    """Acceptance per position must equal 1 - TV(p, q) = sum_x min(p(x), q(x))."""
    _, accepts = _run_trials(DRAFT, TARGET[:GAMMA])

    per_position = torch.minimum(TARGET[:GAMMA], DRAFT).sum(-1)
    # E[n_accepted] = sum_i prod_{j <= i} alpha_j, since position i is only
    # reached when every earlier token was accepted.
    expected = float(torch.cumprod(per_position, dim=0).sum())

    observed = sum(accepts) / N_RUNS
    tol = 4.5 * GAMMA / math.sqrt(N_RUNS)  # generous bound on the SEM
    assert abs(observed - expected) < tol, (
        "mean accepted {:.4f} vs theoretical {:.4f} (per-position alphas {})".format(
            observed, expected, [round(v, 4) for v in per_position.tolist()]
        )
    )
    assert 0 < observed < GAMMA, "degenerate experiment: draft never or always accepted"


def test_rejection_resamples_from_normalized_residual():
    """Conditioned on rejection, the replacement follows norm(max(0, p - q))."""
    target, draft = TARGET[:1], DRAFT[:1]
    outputs, accepts = _run_trials(draft, target)

    residual = (target[0] - draft[0]).clamp_min(0.0)
    residual = residual / residual.sum()

    counts = torch.zeros(VOCAB, dtype=torch.long)
    for tokens, n in zip(outputs, accepts):
        if n == 0:
            assert tokens.shape[0] == 1
            counts[int(tokens[0])] += 1

    # The residual has zeros, so restrict the goodness of fit to its support and
    # separately assert the zero-probability categories are never emitted.
    support = residual > 0
    assert int(counts[~support].sum()) == 0, (
        "replacement drawn outside the residual support: {}".format(counts[~support].tolist())
    )
    assert_matches_distribution(counts[support], residual[support], "residual resample")


def test_accepted_tokens_follow_min_p_q():
    """Conditioned on acceptance, the token follows norm(min(p, q)).

    This is the other half of the proof: accept and reject branches mix back to
    p exactly because acceptance keeps min(p, q) and rejection supplies the rest.
    """
    target, draft = TARGET[:1], DRAFT[:1]
    outputs, accepts = _run_trials(draft, target)

    overlap = torch.minimum(target[0], draft[0])
    overlap = overlap / overlap.sum()

    counts = torch.zeros(VOCAB, dtype=torch.long)
    for tokens, n in zip(outputs, accepts):
        if n == 1:
            counts[int(tokens[0])] += 1

    assert_matches_distribution(counts, overlap, "accepted tokens")


# ==========================================================================
# Structural and edge-case behaviour
# ==========================================================================
def test_identical_distributions_accept_everything():
    """A perfect draft model must never be rejected."""
    probs = TARGET[:GAMMA]
    outputs, accepts = _run_trials(probs, probs, n_runs=500)
    assert all(n == GAMMA for n in accepts)
    assert all(tokens.shape[0] == GAMMA for tokens in outputs)


def test_disjoint_support_always_rejects_and_falls_back_to_target():
    """With disjoint supports the draft is worthless: everything is rejected and
    the emitted token comes from the target itself, since p - q = p there."""
    target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1]], dtype=DTYPE)
    draft = torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    outputs, accepts = _run_trials(draft, target)

    assert set(accepts) == {0}, "a draft with disjoint support must never be accepted"
    counts = torch.zeros(VOCAB, dtype=torch.long)
    for tokens in outputs:
        assert tokens.shape[0] == 1
        counts[int(tokens[0])] += 1
    support = target[0] > 0
    assert int(counts[~support].sum()) == 0
    assert_matches_distribution(counts[support], target[0][support], "disjoint fallback")


def test_bonus_token_emitted_on_full_acceptance():
    """A gamma+1-th target row grants a free token when nothing is rejected."""
    outputs, accepts = _run_trials(DRAFT, TARGET)  # 4 target rows for gamma=3

    bonus_counts = torch.zeros(VOCAB, dtype=torch.long)
    n_full = 0
    for tokens, n in zip(outputs, accepts):
        assert tokens.shape[0] == n + 1, "with a bonus row every run emits n_accepted + 1"
        if n == GAMMA:
            n_full += 1
            bonus_counts[int(tokens[-1])] += 1
    assert n_full > 500, "only {} full acceptances, too few to test the bonus token".format(n_full)
    assert_matches_distribution(bonus_counts, TARGET[GAMMA], "bonus token")


def test_deterministic_target_forces_its_own_token():
    """A one-hot target must emit that token whatever the draft proposed."""
    target = torch.zeros(1, VOCAB, dtype=DTYPE)
    target[0, 3] = 1.0
    draft = torch.full((1, VOCAB), 1.0 / VOCAB, dtype=DTYPE)
    outputs, accepts = _run_trials(draft, target, n_runs=1000)
    for tokens in outputs:
        assert tokens.shape[0] == 1
        assert int(tokens[0]) == 3
    # Accepted exactly when the draft happened to propose token 3, about 1/8.
    assert 0 < sum(accepts) < 1000


def test_returns_are_well_formed():
    gen = torch.Generator().manual_seed(7)
    for _ in range(200):
        draft_tokens = torch.multinomial(DRAFT, 1, generator=gen).squeeze(1)
        tokens, n_accepted = verify_tokens(DRAFT, TARGET[:GAMMA], draft_tokens, generator=gen)
        assert isinstance(tokens, torch.Tensor) and tokens.dtype == torch.long
        assert tokens.dim() == 1 and tokens.numel() >= 1
        assert isinstance(n_accepted, int) and 0 <= n_accepted <= GAMMA
        assert bool((tokens >= 0).all()) and bool((tokens < VOCAB).all())
        # The accepted prefix must be verbatim from the draft.
        assert torch.equal(tokens[:n_accepted], draft_tokens[:n_accepted])


def test_generator_makes_runs_reproducible():
    def once(seed):
        gen = torch.Generator().manual_seed(seed)
        tokens = torch.multinomial(DRAFT, 1, generator=gen).squeeze(1)
        return verify_tokens(DRAFT, TARGET, tokens, generator=gen)

    a_tokens, a_n = once(123)
    b_tokens, b_n = once(123)
    assert a_n == b_n and torch.equal(a_tokens, b_tokens)


@pytest.mark.parametrize(
    "draft_shape, target_shape, token_shape",
    [
        ((3, 8), (3, 8), (2,)),    # token count disagrees with draft rows
        ((3, 8), (5, 8), (3,)),    # too many target rows
        ((3, 8), (3, 7), (3,)),    # vocab mismatch
        ((3,), (3, 8), (3,)),      # draft not 2-D
        ((3, 8), (3, 8), (3, 1)),  # tokens not 1-D
    ],
)
def test_shape_validation(draft_shape, target_shape, token_shape):
    draft = torch.rand(draft_shape, dtype=DTYPE) + 0.1
    target = torch.rand(target_shape, dtype=DTYPE) + 0.1
    if draft.dim() == 2:
        draft = draft / draft.sum(-1, keepdim=True)
    target = target / target.sum(-1, keepdim=True)
    tokens = torch.zeros(token_shape, dtype=torch.long)
    with pytest.raises(ValueError):
        verify_tokens(draft, target, tokens)
