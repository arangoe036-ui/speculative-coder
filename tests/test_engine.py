"""Correctness tests for the speculative decoding loop, on CPU, no downloads.

Tiny randomly-initialised Qwen2 models stand in for the real draft/target pair.
The draft is a *noise-perturbed copy* of the target rather than an independent
model: two independently initialised models agree on barely 1% of tokens, which
would leave the accept and rollback paths essentially untested.  Perturbation
scale tunes the acceptance rate, so one suite covers ~68%, ~26% and ~1% drafts.

The load-bearing test is `test_greedy_matches_plain_autoregressive`.  At
temperature 0 both p and q are one-hot, so the verifier accepts exactly when the
two models agree and otherwise emits the target's own argmax -- which means
speculative decoding must reproduce plain greedy decoding on the target *token
for token*.  Any off-by-one in the logits slicing, or any stale entry left in a
rolled-back KV cache, breaks that equality immediately.  It is a far sharper
instrument than eyeballing streamed text.
"""

from __future__ import annotations

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from core.engine import (  # noqa: E402
    SpeculativeEngine,
    _cache_length,
    _crop_cache,
    _fmean,
)
from core.gates import SyntaxState  # noqa: E402

VOCAB = 16
PROMPT = "cbadcbad"


class CharTokenizer:
    """Minimal tokenizer stub: one token per character, ids in [0, VOCAB)."""

    eos_token_id = None  # disabled so tests generate a fixed number of tokens

    def __call__(self, text, return_tensors=None):
        ids = [ord(c) % VOCAB for c in text]
        return type("Encoded", (), {"input_ids": torch.tensor([ids])})()

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(ord("a") + int(i) % 26) for i in ids)


def _tiny_model(seed: int) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # exercise GQA, like the real Qwen models
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def _perturbed_model(base, scale: float, seed: int):
    """A copy of ``base`` with Gaussian noise added to every weight.

    Two independently initialised models agree on ~1% of tokens, which means an
    equivalence test built on them only ever exercises the *rejection* path.  A
    perturbed copy of the target is a draft that agrees often but not always, so
    the accept path, the partial-acceptance rollback and the full-acceptance
    bonus token all get hit in a single run.  `scale` tunes the disagreement rate.
    """
    model = copy.deepcopy(base)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            spread = param.detach().std() if param.numel() > 1 else param.detach().abs()
            param.add_(torch.randn(param.shape, generator=generator) * scale * spread)
    model.eval()
    return model


@pytest.fixture(scope="module")
def target_model():
    return _tiny_model(1234)


@pytest.fixture(scope="module")
def models(target_model):
    """A *good* draft: ~68% acceptance, mixing full, partial and zero acceptance."""
    return _perturbed_model(target_model, scale=0.02, seed=7), target_model


@pytest.fixture(scope="module")
def poor_models(target_model):
    """A mediocre draft: ~26% acceptance, dominated by short accepted runs."""
    return _perturbed_model(target_model, scale=0.05, seed=7), target_model


@pytest.fixture(scope="module")
def unrelated_models(target_model):
    """The pathological case: an independent model that almost never agrees."""
    return _tiny_model(0), target_model


@pytest.fixture(scope="module")
def tokenizer():
    return CharTokenizer()


def reference_greedy(model, input_ids: torch.Tensor, n_tokens: int) -> list[int]:
    """Independent greedy decode: no KV cache, full recompute every step.

    Deliberately the slowest possible implementation.  Sharing no code with the
    engine is the point -- it is the oracle the engine is checked against.
    """
    ids = input_ids.clone()
    out: list[int] = []
    with torch.no_grad():
        for _ in range(n_tokens):
            logits = model(input_ids=ids).logits
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return out


# ==========================================================================
# The load-bearing equivalence test
# ==========================================================================
@pytest.mark.parametrize("quality", ["good", "poor", "unrelated"])
@pytest.mark.parametrize("k", [1, 2, 3, 5, 8])
def test_greedy_matches_plain_autoregressive(request, tokenizer, quality, k):
    """Greedy speculative decoding must equal greedy target decoding exactly.

    Parametrised over k *and* draft quality: a wrong logits offset or a botched
    cache crop typically survives one combination and fails another, and the
    three qualities put the acceptance rate at roughly 68% / 26% / 1% so all
    three branches (full, partial, zero acceptance) are covered.
    """
    fixture = {"good": "models", "poor": "poor_models", "unrelated": "unrelated_models"}
    draft, target = request.getfixturevalue(fixture[quality])
    engine = SpeculativeEngine(draft, target, tokenizer, k=k, temperature=0.0)

    expected = reference_greedy(target, tokenizer(PROMPT).input_ids, 24)
    _, stats = engine.generate(PROMPT, max_new_tokens=24, stream=False)
    actual = _generated_ids(engine, tokenizer, PROMPT, 24)

    assert actual == expected, (
        "speculative greedy diverged from plain greedy at k={}, draft={}\n"
        "  expected: {}\n  actual  : {}".format(k, quality, expected, actual)
    )
    assert stats.tokens_generated == 24
    assert 0 <= stats.draft_tokens_accepted <= stats.draft_tokens_proposed


def test_equivalence_test_actually_covers_partial_acceptance(models, tokenizer):
    """Guard the guard: the 'good' draft must really produce mixed acceptance.

    Without this, a future change that quietly drives acceptance to 0% or 100%
    would leave the equivalence test above green while it stopped testing the
    interesting rollback path at all.
    """
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=5, temperature=0.0)
    _, stats = engine.generate(PROMPT, max_new_tokens=24, stream=False)

    per_iteration = stats.accepted_per_iteration
    assert any(n == 5 for n in per_iteration), "no full acceptance: bonus path untested"
    assert any(0 < n < 5 for n in per_iteration), "no partial acceptance: rollback untested"
    assert 0.3 < stats.acceptance_rate < 0.95, stats.summary()


def _generated_ids(engine, tokenizer, prompt, n):
    """Re-run generate and recover ids by re-encoding the decoded text.

    The char tokenizer is a bijection on ids mod 26, so decode/encode round-trips
    exactly; this keeps `generate`'s public signature unchanged for the test.
    """
    text, _ = engine.generate(prompt, max_new_tokens=n, stream=False)
    return [ord(c) - ord("a") for c in text]


def test_perfect_draft_accepts_everything(models, tokenizer):
    """When the draft *is* the target, nothing is ever rejected."""
    _, target = models
    engine = SpeculativeEngine(target, target, tokenizer, k=5, temperature=0.0)
    _, stats = engine.generate(PROMPT, max_new_tokens=20, stream=False)

    assert stats.acceptance_rate == 1.0, stats.summary()
    # Every iteration emits k + 1 tokens, so 20 tokens needs ceil(20/6) = 4 passes.
    assert stats.target_forwards == 4, stats.summary()
    assert stats.speedup_vs_autoregressive == pytest.approx(5.0)


def test_perfect_draft_still_matches_greedy(models, tokenizer):
    """The bonus-token path must not corrupt the output either."""
    _, target = models
    engine = SpeculativeEngine(target, target, tokenizer, k=5, temperature=0.0)
    expected = reference_greedy(target, tokenizer(PROMPT).input_ids, 20)
    assert _generated_ids(engine, tokenizer, PROMPT, 20) == expected


# ==========================================================================
# Distributional check of the full loop (not just the verifier)
# ==========================================================================
def test_single_token_distribution_matches_target(models, tokenizer):
    """4,000 one-token generations must follow the target's true softmax.

    Phase 1 proved the verifier is unbiased given correct p and q.  This proves
    the *engine* hands it correct p and q: the prompt is encoded, the draft
    proposes, the target scores in one pass, and the emitted token still comes
    out distributed as the target's own next-token distribution.
    """
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=3, temperature=1.0)

    prompt_ids = tokenizer(PROMPT).input_ids
    with torch.no_grad():
        expected = torch.softmax(
            target(input_ids=prompt_ids).logits[0, -1].float(), dim=-1
        )

    n_runs = 4000
    torch.manual_seed(99)
    counts = torch.zeros(VOCAB, dtype=torch.long)
    for _ in range(n_runs):
        text, _ = engine.generate(PROMPT, max_new_tokens=1, stream=False)
        assert len(text) == 1
        counts[ord(text[0]) - ord("a")] += 1

    empirical = counts.double() / n_runs
    expected = expected.double()
    sd = torch.sqrt(expected * (1 - expected) / n_runs).clamp_min(1e-12)
    z = (empirical - expected) / sd
    worst = int(z.abs().argmax())
    assert z.abs().max() < 4.5, (
        "engine output is biased: token {} z={:.2f} (empirical {:.4f} vs target {:.4f})\n"
        "  empirical: {}\n  expected : {}".format(
            worst, float(z[worst]), float(empirical[worst]), float(expected[worst]),
            [round(v, 4) for v in empirical.tolist()],
            [round(v, 4) for v in expected.tolist()],
        )
    )
    tv = 0.5 * float((empirical - expected).abs().sum())
    assert tv < 0.03, "total variation {:.4f} too large".format(tv)


# ==========================================================================
# Budget, streaming and cache mechanics
# ==========================================================================
@pytest.mark.parametrize("max_new_tokens", [1, 2, 5, 6, 7, 13])
def test_max_new_tokens_is_exact(models, tokenizer, max_new_tokens):
    """Never overshoot the budget, even though full acceptance emits k + 1."""
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=5, temperature=0.0)
    text, stats = engine.generate(PROMPT, max_new_tokens=max_new_tokens, stream=False)
    assert stats.tokens_generated == max_new_tokens
    assert len(text) == max_new_tokens


def test_truncated_budget_does_not_corrupt_the_prefix(models, tokenizer):
    """A budget that cuts mid-block must still yield a prefix of the full run."""
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=5, temperature=0.0)
    full = _generated_ids(engine, tokenizer, PROMPT, 24)
    for n in (1, 3, 7, 11, 18):
        assert _generated_ids(engine, tokenizer, PROMPT, n) == full[:n], (
            "budget {} is not a prefix of the 24-token run".format(n)
        )


def test_eos_as_first_token_stops_immediately(models, tokenizer):
    """The degenerate case: EOS before any real token."""
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=5, temperature=0.0)
    first = reference_greedy(target, tokenizer(PROMPT).input_ids, 1)[0]

    class EosTokenizer(CharTokenizer):
        eos_token_id = first

    engine.tokenizer = EosTokenizer()
    text, stats = engine.generate(PROMPT, max_new_tokens=20, stream=False)
    assert stats.tokens_generated == 0 and text == ""


@pytest.mark.parametrize("eos_position", [1, 2, 3, 5, 6, 9])
def test_eos_mid_stream_stops_cleanly(models, tokenizer, eos_position):
    """EOS partway through a block must halt without desynchronising state.

    Parametrised across and past the K=5 block boundary. The earlier version of
    this test only placed EOS at position 0, where `generated` stays empty --
    which masked a real desync between `context` and `generated` on every run
    that ended naturally rather than by exhausting the token budget. That is the
    common case at a realistic budget, so it has to be covered explicitly.
    """
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=5, temperature=0.0)

    reference = reference_greedy(target, tokenizer(PROMPT).input_ids, 20)
    eos_id = reference[eos_position]
    # The chosen token may recur earlier; generation stops at its first occurrence.
    expected_length = reference.index(eos_id)

    class EosTokenizer(CharTokenizer):
        eos_token_id = eos_id

    engine.tokenizer = EosTokenizer()
    text, stats = engine.generate(PROMPT, max_new_tokens=20, stream=False)

    assert stats.tokens_generated == expected_length
    assert len(text) == expected_length
    assert eos_id not in stats.token_ids, "EOS must not be emitted"
    assert stats.token_ids == reference[:expected_length]


def test_stats_expose_generated_token_ids(models, tokenizer):
    """token_ids must match the decoded text, so callers can compare exactly."""
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=5, temperature=0.0)
    text, stats = engine.generate(PROMPT, max_new_tokens=16, stream=False)
    assert len(stats.token_ids) == stats.tokens_generated == 16
    assert stats.token_ids == [ord(c) - ord("a") for c in text]


def test_streaming_prints_the_generated_text(models, tokenizer, capsys):
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=5, temperature=0.0)
    text, _ = engine.generate(PROMPT, max_new_tokens=18, stream=True)
    printed = capsys.readouterr().out
    assert printed.strip() == text, "streamed output {!r} != returned {!r}".format(
        printed.strip(), text
    )


def test_cache_is_rolled_back_not_left_stale(models, tokenizer):
    """The KV cache must never claim more positions than the committed context.

    A cache longer than context - 1 means rejected draft tokens were left behind
    and will silently poison every later attention computation.
    """
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=5, temperature=0.0)

    seen: list[tuple[int, int, int]] = []
    original = engine._target_probs

    def spy(context, draft_tokens, cache):
        seen.append((context.shape[1], _cache_length(cache), _cache_length(cache)))
        return original(context, draft_tokens, cache)

    engine._target_probs = spy
    engine.generate(PROMPT, max_new_tokens=24, stream=False)

    assert len(seen) > 3, "not enough iterations to test rollback"
    for i, (context_len, cache_len, _) in enumerate(seen):
        # Equality, not "<=": a cache that is one position too long silently
        # corrupts attention, and one too short silently wastes a forward pass.
        # Only an exact assertion catches both, so pin the invariant precisely:
        # the cache holds every committed token except the last (which no
        # forward pass has seen yet). The first iteration starts cold at 0.
        expected = 0 if i == 0 else context_len - 1
        assert cache_len == expected, (
            "iteration {}: target cache holds {} positions, expected {} "
            "for a {}-token context".format(i, cache_len, expected, context_len)
        )


def test_rollback_reuses_the_cache_instead_of_recomputing(models, tokenizer):
    """Rollback must be a slice, not a silent full recompute.

    If the cache were being thrown away each iteration, every pass would re-feed
    the whole context and the loop would still be *correct* -- just quadratic.
    Counting fed tokens is the only way to tell the two apart.
    """
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=5, temperature=0.0)

    fed_widths: list[int] = []
    original = engine._forward

    def spy(model, input_ids, cache):
        if model is target:
            fed_widths.append(input_ids.shape[1])
        return original(model, input_ids, cache)

    engine._forward = spy
    engine.generate(PROMPT, max_new_tokens=30, stream=False)

    prompt_len = tokenizer(PROMPT).input_ids.shape[1]
    # First target pass prefills the prompt plus the drafted block.
    assert fed_widths[0] == prompt_len + 5
    # Every later pass feeds only the uncached tail: the emitted token, any
    # draft tokens the previous rollback discarded, and the new block.
    for i, width in enumerate(fed_widths[1:], start=1):
        assert width <= 7, (
            "target pass {} fed {} tokens; the cache is not being reused".format(i, width)
        )


def test_crop_cache_helper():
    """_crop_cache must handle None, real Cache objects and legacy tuples."""
    assert _crop_cache(None, 5) is None
    assert _cache_length(None) == 0

    legacy = tuple(
        (torch.zeros(1, 2, 10, 8), torch.zeros(1, 2, 10, 8)) for _ in range(3)
    )
    assert _cache_length(legacy) == 10
    cropped = _crop_cache(legacy, 4)
    assert _cache_length(cropped) == 4
    assert all(t.shape[-2] == 4 for layer in cropped for t in layer)
    # Cropping to a length it already satisfies is a no-op, not an error.
    assert _cache_length(_crop_cache(cropped, 99)) == 4


def test_invalid_construction_arguments(models, tokenizer):
    draft, target = models
    for kwargs in ({"k": 0}, {"temperature": -1.0}, {"top_p": 0.0}, {"top_p": 1.5}):
        with pytest.raises(ValueError):
            SpeculativeEngine(draft, target, tokenizer, **kwargs)
    engine = SpeculativeEngine(draft, target, tokenizer)
    with pytest.raises(ValueError):
        engine.generate(PROMPT, max_new_tokens=0)


def test_top_p_is_applied_to_both_models(models, tokenizer):
    """Nucleus filtering must renormalise and zero the tail on both p and q."""
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=3, temperature=1.0, top_p=0.5)
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.5] + [-20.0] * (VOCAB - 4)])
    probs = engine._logits_to_probs(logits)
    assert probs.sum(-1).item() == pytest.approx(1.0)
    assert int((probs > 0).sum()) < VOCAB, "top_p=0.5 should zero part of the tail"


# ==========================================================================
# Vocabulary reconciliation
# ==========================================================================
def _model_with_vocab(base_seed: int, vocab: int):
    torch.manual_seed(base_seed)
    config = Qwen2Config(
        vocab_size=vocab, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def test_mismatched_padded_vocabularies_are_reconciled(tokenizer):
    """Sibling checkpoints padded to different widths must still verify.

    This is not hypothetical: Qwen2.5-Coder-1.5B pads to 151936 and the 7B pads
    to 152064, so the pair this project targets hits it. Without reconciliation
    verify_tokens would compare two differently-shaped [K, V] tensors.
    """
    draft = _model_with_vocab(0, VOCAB)
    target = _model_with_vocab(1234, VOCAB + 8)  # target padded wider
    engine = SpeculativeEngine(draft, target, tokenizer, k=3, temperature=0.0)

    assert engine.vocab_size == VOCAB, "should verify over the narrower width"
    text, stats = engine.generate(PROMPT, max_new_tokens=12, stream=False)
    assert stats.tokens_generated == 12
    assert all(0 <= ord(c) - ord("a") < VOCAB for c in text)


def test_reconciled_greedy_matches_target_over_shared_prefix(tokenizer):
    """With a wider target, greedy output must match the target restricted to
    the shared prefix -- the truncated columns are padding and must not win."""
    draft = _model_with_vocab(0, VOCAB)
    target = _model_with_vocab(1234, VOCAB + 8)
    engine = SpeculativeEngine(draft, target, tokenizer, k=4, temperature=0.0)

    # Oracle: plain greedy on the target, argmax restricted to the shared prefix.
    ids = tokenizer(PROMPT).input_ids.clone()
    expected = []
    with torch.no_grad():
        for _ in range(12):
            nxt = int(target(input_ids=ids).logits[0, -1, :VOCAB].argmax())
            expected.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)

    text, _ = engine.generate(PROMPT, max_new_tokens=12, stream=False)
    assert [ord(c) - ord("a") for c in text] == expected


# ==========================================================================
# Dual-gated adaptive drafting
# ==========================================================================
class BracketTokenizer(CharTokenizer):
    """Decodes every token to a closing paren, so the syntactic gate can fire."""

    def decode(self, ids, skip_special_tokens=False):
        return ")" * len(list(ids))


def test_dual_gate_preserves_greedy_output(models, tokenizer):
    """The whole point: adaptive drafting must not change what is emitted.

    Gating chooses the block length from the *draft's* state alone, and
    verify_tokens is exact for any length, so greedy output must still match
    plain greedy decoding on the target. Swept from a threshold that never fires
    to one that fires on every token -- if any of those changed the output, the
    gate would be reading something it must not.
    """
    draft, target = models
    expected = reference_greedy(target, tokenizer(PROMPT).input_ids, 24)

    for threshold in (0.0, 0.05, 0.35, 1.0):
        engine = SpeculativeEngine(
            draft, target, tokenizer, temperature=0.0,
            use_dual_gate=True, max_draft_len=8, entropy_threshold=threshold,
        )
        actual = _generated_ids(engine, tokenizer, PROMPT, 24)
        assert actual == expected, (
            "dual gate changed the output at entropy_threshold={}\n"
            "  expected: {}\n  actual  : {}".format(threshold, expected, actual)
        )


def test_dual_gate_matches_static_k_output(models, tokenizer):
    """Adaptive and static drafting must agree token for token."""
    draft, target = models
    static = SpeculativeEngine(draft, target, tokenizer, k=5, temperature=0.0)
    adaptive = SpeculativeEngine(
        draft, target, tokenizer, temperature=0.0,
        use_dual_gate=True, max_draft_len=8, entropy_threshold=0.35,
    )
    assert (_generated_ids(adaptive, tokenizer, PROMPT, 24)
            == _generated_ids(static, tokenizer, PROMPT, 24))


def test_threshold_zero_never_gates(models, tokenizer):
    """A floor of 0 cannot be undercut, so blocks run to max_draft_len."""
    draft, target = models
    engine = SpeculativeEngine(
        draft, target, tokenizer, temperature=0.0,
        use_dual_gate=True, max_draft_len=6, entropy_threshold=0.0,
    )
    _, stats = engine.generate(PROMPT, max_new_tokens=24, stream=False)
    assert stats.statistical_gate_triggers == 0
    assert stats.ungated_iterations == stats.iterations
    # Every block hits the ceiling except possibly the last, clamped by budget.
    assert all(n == 6 for n in stats.draft_lengths[:-1]), stats.draft_lengths
    assert stats.mean_draft_length == pytest.approx(_fmean(stats.draft_lengths))


def test_threshold_one_gates_every_token(models, tokenizer):
    """A floor of 1.0 is unreachable, so every block stops after one token.

    The degenerate end of the adaptive range, which must still be correct: a
    one-token block is a legal block.
    """
    draft, target = models
    engine = SpeculativeEngine(
        draft, target, tokenizer, temperature=0.0,
        use_dual_gate=True, max_draft_len=8, entropy_threshold=1.0,
    )
    _, stats = engine.generate(PROMPT, max_new_tokens=20, stream=False)
    assert set(stats.draft_lengths) == {1}
    assert stats.statistical_gate_triggers == stats.iterations
    assert stats.ungated_iterations == 0
    assert stats.mean_draft_length == pytest.approx(1.0)


def test_statistical_gate_reads_the_unwarped_softmax(models, tokenizer):
    """At temperature 0 the warped q is one-hot, so a gate built on it is dead.

    The subtlest part of the design. `_logits_to_probs` returns a one-hot
    distribution at temperature 0, whose top-1 probability is always exactly 1.0.
    A confidence gate reading that would never fire in precisely the
    configuration the benchmark uses, and would look indistinguishable from a
    gate that simply never triggers.
    """
    draft, target = models
    engine = SpeculativeEngine(
        draft, target, tokenizer, temperature=0.0,
        use_dual_gate=True, max_draft_len=8, entropy_threshold=0.9,
    )
    _, stats = engine.generate(PROMPT, max_new_tokens=20, stream=False)
    assert stats.statistical_gate_triggers > 0, (
        "gate never fired at temperature 0 -- it is reading the warped one-hot q"
    )


def test_max_draft_len_is_the_ceiling(models, tokenizer):
    draft, target = models
    for ceiling in (1, 2, 4, 8):
        engine = SpeculativeEngine(
            draft, target, tokenizer, temperature=0.0,
            use_dual_gate=True, max_draft_len=ceiling, entropy_threshold=0.0,
        )
        _, stats = engine.generate(PROMPT, max_new_tokens=16, stream=False)
        assert max(stats.draft_lengths) <= ceiling
        assert all(n >= 1 for n in stats.draft_lengths), "a block is never empty"


def test_gates_ignore_k_when_enabled(models, tokenizer):
    """use_dual_gate replaces k with max_draft_len; a stale k must not leak in."""
    draft, target = models
    engine = SpeculativeEngine(
        draft, target, tokenizer, k=2, temperature=0.0,
        use_dual_gate=True, max_draft_len=7, entropy_threshold=0.0,
    )
    _, stats = engine.generate(PROMPT, max_new_tokens=21, stream=False)
    assert max(stats.draft_lengths) == 7, "k=2 should be ignored under the gate"


def test_syntactic_gate_fires_and_keeps_the_triggering_token():
    """A fatal bracket state stops the block, with the bad token retained.

    Retaining it is deliberate: the target scores the whole block in one forward
    either way, so a doubtful token is free to verify and may still be accepted,
    while dropping it can only lose a token. What the gate buys is skipping the
    *remaining* draft forwards.
    """
    target = _tiny_model(1234)
    draft = _perturbed_model(target, scale=0.02, seed=7)
    engine = SpeculativeEngine(
        draft, target, BracketTokenizer(), temperature=0.0,
        use_dual_gate=True, max_draft_len=8, entropy_threshold=0.0,
    )

    state = SyntaxState()
    state.feed("```python\n")           # inside code, empty bracket stack
    context = BracketTokenizer()(PROMPT).input_ids

    tokens, q, _, forwards, gates = engine._draft(context, None, 8, state)

    assert gates["syntactic"] == 1 and gates["statistical"] == 0
    assert tokens.shape[0] == 1, "stops immediately, keeping the bad token"
    assert q.shape[0] == 1
    assert forwards == 1, "the remaining 7 draft forwards were skipped"
    assert state.fatal


def test_syntactic_gate_silent_outside_code_fences():
    """The same token in prose is not an error and must not gate."""
    target = _tiny_model(1234)
    draft = _perturbed_model(target, scale=0.02, seed=7)
    engine = SpeculativeEngine(
        draft, target, BracketTokenizer(), temperature=0.0,
        use_dual_gate=True, max_draft_len=4, entropy_threshold=0.0,
    )
    state = SyntaxState()               # never entered a fence
    context = BracketTokenizer()(PROMPT).input_ids

    tokens, _, _, forwards, gates = engine._draft(context, None, 4, state)

    assert gates["syntactic"] == 0
    assert tokens.shape[0] == 4 and forwards == 4
    assert state.suppressed == 4, "counted rather than enforced"


def test_rejected_draft_does_not_pollute_committed_syntax_state(models, tokenizer):
    """The tracker must be cloned per block, not shared.

    A draft that hits a fatal bracket state is usually rejected. Had the
    committed tracker absorbed it, every later iteration would see a fatal state
    and the gate would jam permanently at draft length 1.
    """
    draft, target = models
    engine = SpeculativeEngine(
        draft, target, tokenizer, temperature=0.0,
        use_dual_gate=True, max_draft_len=6, entropy_threshold=0.0,
    )
    _, stats = engine.generate(PROMPT, max_new_tokens=30, stream=False)
    assert stats.draft_lengths[-1] >= 1
    assert max(stats.draft_lengths) == 6, "gate jammed: blocks collapsed"


def test_telemetry_is_recorded_per_iteration(models, tokenizer):
    draft, target = models
    engine = SpeculativeEngine(
        draft, target, tokenizer, temperature=0.0,
        use_dual_gate=True, max_draft_len=8, entropy_threshold=0.5,
    )
    _, stats = engine.generate(PROMPT, max_new_tokens=24, stream=False)

    assert len(stats.draft_lengths) == stats.iterations
    assert sum(stats.draft_lengths) == stats.draft_tokens_proposed
    triggered = stats.statistical_gate_triggers + stats.syntactic_gate_triggers
    assert triggered + stats.ungated_iterations == stats.iterations
    assert 0.0 <= stats.gate_trigger_rate <= 1.0
    assert "mean draft length" in stats.gate_summary()


def test_static_mode_leaves_gate_telemetry_empty(models, tokenizer):
    """Without the gate, counters stay zero but draft_lengths is still filled."""
    draft, target = models
    engine = SpeculativeEngine(draft, target, tokenizer, k=5, temperature=0.0)
    _, stats = engine.generate(PROMPT, max_new_tokens=24, stream=False)
    assert stats.statistical_gate_triggers == 0
    assert stats.syntactic_gate_triggers == 0
    assert stats.ungated_iterations == 0, "not applicable when the gate is off"
    assert set(stats.draft_lengths) <= {5, 4, 3, 2, 1}
    assert stats.gate_trigger_rate == 0.0


@pytest.mark.parametrize("kwargs", [
    {"max_draft_len": 0}, {"max_draft_len": -1},
    {"entropy_threshold": -0.1}, {"entropy_threshold": 1.5},
])
def test_invalid_gate_arguments(models, tokenizer, kwargs):
    draft, target = models
    with pytest.raises(ValueError):
        SpeculativeEngine(draft, target, tokenizer, use_dual_gate=True, **kwargs)
