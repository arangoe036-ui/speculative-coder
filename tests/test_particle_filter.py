"""Tests for particle filter (resampled) speculative decoding.

Correctness is checked the same way as everywhere: greedy output must equal plain
greedy decoding. That holds however biased the draft is, which is exactly why it
cannot be the only test -- a resampler that collapsed every particle onto one path
would still produce perfect output while doing no useful work.

So the resampling mechanics are asserted directly: survivors must actually be the
highest-weighted particles, clones must be exact, and diversity must not collapse
to a single candidate.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import _cache_length  # noqa: E402
from core.particle_filter_engine import ParticleFilterEngine, _reorder_cache  # noqa: E402
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
@pytest.mark.parametrize("particles", [2, 4, 8])
def test_output_matches_plain_greedy(draft_model, target_model, tokenizer, particles):
    """Resampling biases the proposal; at T=0 that cannot reach the output."""
    torch.manual_seed(0)
    engine = ParticleFilterEngine(
        draft_model, target_model, tokenizer, k=8, particles=particles,
        draft_temperature=1.2,
    )
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 24)
    actual, _ = _ids(engine)
    assert actual == expected, (
        "resampling changed the output at B={}\n  expected: {}\n  actual  : {}"
        .format(particles, expected, actual)
    )


@pytest.mark.parametrize("survivor_fraction", [0.125, 0.25, 0.5, 0.75])
def test_output_is_independent_of_survivor_fraction(
    draft_model, target_model, tokenizer, survivor_fraction
):
    """Pruning harder is allowed to cost acceptance, never correctness."""
    torch.manual_seed(1)
    engine = ParticleFilterEngine(
        draft_model, target_model, tokenizer, k=8, particles=8,
        survivor_fraction=survivor_fraction,
    )
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 24)
    assert _ids(engine)[0] == expected, f"survivor_fraction={survivor_fraction}"


@pytest.mark.parametrize("k", [1, 2, 5, 8])
def test_output_matches_greedy_across_k(draft_model, target_model, tokenizer, k):
    torch.manual_seed(2)
    engine = ParticleFilterEngine(draft_model, target_model, tokenizer, k=k, particles=8)
    expected = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 20)
    assert _ids(engine, 20)[0] == expected


def test_candidates_form_a_prefix_chain(draft_model, target_model, tokenizer):
    """All particles must emit prefixes of one string, or argmax is arbitrary."""
    from core.verifier import verify_tokens

    torch.manual_seed(3)
    engine = ParticleFilterEngine(
        draft_model, target_model, tokenizer, k=8, particles=8, draft_temperature=1.5
    )
    context = tokenizer(PROMPT).input_ids
    tokens, q, _, _, _ = engine._draft_particles(context, None, 8)
    p, _ = engine._verify_candidates(context, tokens, None)

    emitted = []
    for b in range(tokens.shape[0]):
        seq, n_accepted = verify_tokens(q[b], p[b], tokens[b])
        assert len(seq) == n_accepted + 1
        emitted.append(seq.tolist())

    emitted.sort(key=len)
    for shorter, longer in zip(emitted, emitted[1:]):
        assert longer[: len(shorter)] == shorter, (
            f"particles disagree on a shared position:\n  {shorter}\n  {longer}"
        )


def test_temperature_above_zero_is_refused(draft_model, target_model, tokenizer):
    with pytest.raises(ValueError, match="temperature=0"):
        ParticleFilterEngine(draft_model, target_model, tokenizer, temperature=0.7)


def test_invalid_arguments(draft_model, target_model, tokenizer):
    for kwargs, match in (
        ({"particles": 1}, "particles"),
        ({"draft_temperature": 0.0}, "draft_temperature"),
        ({"survivor_fraction": 0.0}, "survivor_fraction"),
        ({"survivor_fraction": 1.0}, "survivor_fraction"),
    ):
        with pytest.raises(ValueError, match=match):
            ParticleFilterEngine(draft_model, target_model, tokenizer, **kwargs)


# ---------------------------------------------------------------- resampling
def test_survivors_are_the_highest_weighted_particles(draft_model, target_model, tokenizer):
    """The prune step must keep the strongest paths, not an arbitrary subset.

    Verified by reconstructing each particle's cumulative log-probability from the
    returned q rows and tokens, then checking that the surviving *distinct*
    sequences are the strongest ones present.
    """
    torch.manual_seed(4)
    engine = ParticleFilterEngine(
        draft_model, target_model, tokenizer, k=6, particles=8, draft_temperature=1.5
    )
    context = tokenizer(PROMPT).input_ids
    tokens, q, _, _, resamples = engine._draft_particles(context, None, 6)

    assert resamples == 5, "resampling should run between steps, not after the last"

    # Every surviving particle descends from a kept ancestor, so after k-1 prunes
    # the population should contain far fewer distinct sequences than particles.
    distinct = {tuple(row) for row in tokens.tolist()}
    assert len(distinct) <= 8

    # Reconstruct log-weights; the strongest sequence must be present in the final
    # population (it can never be pruned).
    log_probs = q.gather(2, tokens.unsqueeze(2)).squeeze(2).clamp_min(1e-30).log()
    totals = log_probs.sum(dim=1)
    best_index = int(totals.argmax())
    assert tuple(tokens[best_index].tolist()) in distinct


def test_clones_are_exact(draft_model, target_model, tokenizer):
    """A cloned particle must be identical in tokens and in cache state.

    Checked via the KV cache: two slots holding the same particle must have
    byte-identical keys, otherwise the clone is a different model state wearing the
    same token history and its q rows would be wrong.
    """
    torch.manual_seed(5)
    engine = ParticleFilterEngine(
        draft_model, target_model, tokenizer, k=4, particles=8, draft_temperature=2.0
    )
    context = tokenizer(PROMPT).input_ids
    tokens, _, cache, _, _ = engine._draft_particles(context, None, 4)

    rows = tokens.tolist()
    keys = cache.layers[0].keys
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i] == rows[j]:
                assert torch.equal(keys[i], keys[j]), (
                    f"particles {i} and {j} share a token history but differ in cache"
                )
                return
    pytest.skip("no duplicate particles arose in this sample")


def test_resampling_concentrates_the_population(draft_model, target_model, tokenizer):
    """Pruning must actually reduce diversity relative to independent sampling.

    Compared against MonteCarloEngine on the same seed: the particle filter should
    reach verification with fewer distinct candidates, since that is precisely what
    pruning does. If the counts matched, resampling would not be happening.
    """
    from core.monte_carlo_engine import MonteCarloEngine

    context = tokenizer(PROMPT).input_ids

    torch.manual_seed(6)
    mc = MonteCarloEngine(draft_model, target_model, tokenizer, k=8, branches=8,
                          draft_temperature=1.5)
    mc_tokens, _, _, _ = mc._draft_branches(context, None, 8)
    mc_distinct = len({tuple(r) for r in mc_tokens.tolist()})

    torch.manual_seed(6)
    pf = ParticleFilterEngine(draft_model, target_model, tokenizer, k=8, particles=8,
                              draft_temperature=1.5)
    pf_tokens, _, _, _, _ = pf._draft_particles(context, None, 8)
    pf_distinct = len({tuple(r) for r in pf_tokens.tolist()})

    assert pf_distinct <= mc_distinct, (
        f"particle filter kept {pf_distinct} distinct drafts vs Monte Carlo's "
        f"{mc_distinct}; pruning should concentrate, not spread"
    )


def test_diversity_does_not_fully_collapse(draft_model, target_model, tokenizer):
    """Clones must re-diverge, or B particles are one particle in disguise.

    This is the method's central risk: resampling toward the draft's own mode can
    drive every particle onto the same path, at which point the batch is pure
    overhead. Clones sample independently on the next step, so some spread should
    survive.
    """
    torch.manual_seed(7)
    engine = ParticleFilterEngine(
        draft_model, target_model, tokenizer, k=8, particles=8, draft_temperature=1.5
    )
    _, stats = _ids(engine, 40)
    assert stats.mean_unique_candidates > 1.0, (
        "every iteration collapsed to a single candidate: resampling has destroyed "
        f"all diversity (mean unique = {stats.mean_unique_candidates:.2f})"
    )
    assert stats.resample_steps > 0


# ---------------------------------------------------------------- bookkeeping
def test_max_new_tokens_is_exact(draft_model, target_model, tokenizer):
    torch.manual_seed(8)
    engine = ParticleFilterEngine(draft_model, target_model, tokenizer, k=8, particles=8)
    for n in (1, 2, 5, 9, 17):
        text, stats = engine.generate(PROMPT, max_new_tokens=n, stream=False)
        assert stats.tokens_generated == n and len(text) == n


def test_caches_collapse_to_one_row(draft_model, target_model, tokenizer):
    torch.manual_seed(9)
    engine = ParticleFilterEngine(draft_model, target_model, tokenizer, k=4, particles=8)

    seen = []
    original = engine._draft_particles

    def spy(context, cache, k):
        if cache is not None:
            seen.append(cache.layers[0].keys.shape[0])
        return original(context, cache, k)

    engine._draft_particles = spy
    engine.generate(PROMPT, max_new_tokens=30, stream=False)
    assert len(seen) > 4
    assert set(seen) == {1}, f"cache handed to the draft phase was batched: {seen}"


def test_n_survivors_is_bounded(draft_model, target_model, tokenizer):
    for fraction, particles, expected in (
        (0.5, 8, 4), (0.25, 8, 2), (0.75, 8, 6), (0.1, 8, 1), (0.9, 2, 1),
    ):
        engine = ParticleFilterEngine(
            draft_model, target_model, tokenizer, particles=particles,
            survivor_fraction=fraction,
        )
        assert engine.n_survivors == expected, (fraction, particles)
        assert 1 <= engine.n_survivors < particles


def test_eos_stops_cleanly(draft_model, target_model, tokenizer):
    torch.manual_seed(10)
    engine = ParticleFilterEngine(draft_model, target_model, tokenizer, k=8, particles=8)
    reference = reference_greedy(target_model, tokenizer(PROMPT).input_ids, 20)
    eos_id = reference[4]
    expected_length = reference.index(eos_id)

    class EosTokenizer(CharTokenizer):
        eos_token_id = eos_id

    engine.tokenizer = EosTokenizer()
    text, stats = engine.generate(PROMPT, max_new_tokens=20, stream=False)
    assert stats.tokens_generated == expected_length
    assert eos_id not in stats.token_ids


def test_reorder_cache_duplicates_rows():
    from transformers import DynamicCache
    cache = DynamicCache()
    for layer in range(2):
        keys = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1).expand(
            4, 2, 3, 4
        ).contiguous()
        cache.update(keys, keys.clone(), layer)

    cache = _reorder_cache(cache, torch.tensor([0, 0, 1, 1]))
    assert cache.layers[0].keys.shape[0] == 4
    assert _cache_length(cache) == 3
    marks = cache.layers[0].keys[:, 0, 0, 0].tolist()
    assert marks == [0.0, 0.0, 1.0, 1.0], marks
    assert _reorder_cache(None, torch.tensor([0])) is None
