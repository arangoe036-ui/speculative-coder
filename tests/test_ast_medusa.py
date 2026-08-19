"""Tests for the AST-guided Medusa architecture.

Three things carry this module. The base must actually be frozen, or the whole
premise (train only the heads) is violated. The AST mask must genuinely forbid
non-structural tokens, since that restriction is the architecture's defining claim.
And the factored projection must be mathematically identical to the literal
full-vocabulary masked version -- that equivalence is the only reason M=15 fits in
VRAM, so it is load-bearing rather than an optimisation.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ast_medusa import (  # noqa: E402
    MedusaAST,
    is_structural,
    structural_token_ids,
)
from tests.test_engine import VOCAB, CharTokenizer, _tiny_model  # noqa: E402


class VocabTokenizer(CharTokenizer):
    """Char tokenizer exposing a real vocab, with some tokens structural."""

    # ids 0..3 decode to structural text, the rest to identifier-ish letters.
    _DECODE = {0: "    ", 1: "def", 2: ":", 3: "\n"}

    def get_vocab(self):
        return {f"<{i}>": i for i in range(VOCAB)}

    def convert_tokens_to_string(self, tokens):
        return "".join(self._DECODE.get(int(t.strip("<>")), "x") for t in tokens)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._DECODE.get(int(i), "x") for i in ids)


@pytest.fixture(scope="module")
def base_model():
    return _tiny_model(1234)


@pytest.fixture(scope="module")
def tokenizer():
    return VocabTokenizer()


# ---------------------------------------------------------------- classification
@pytest.mark.parametrize("text", [
    "    ", "\n", "\t", "        ",            # indentation is Python's structure
    "def", "return", "class", "if", "else", "for", "while", "import", "lambda",
    "self", "None", "True", "False", "match", "case",
    ":", "(", ")", "[", "]", "{", "}", ",", ".", "=", "==", "->", "+=", "**",
    " def", "def ", " : ",                     # surrounding whitespace ignored
])
def test_structural_tokens_are_recognised(text):
    assert is_structural(text), f"{text!r} should be structural"


@pytest.mark.parametrize("text", [
    "quick_sort", "arr", "pivot", "x", "n", "result",   # identifiers
    "42", "3.14", "0x1f",                               # literals
    "hello", "Python", "sorted",                        # words and builtins
    "def_", "returns", "iff",                           # keyword-adjacent, not keywords
    "a=b",                                              # mixed content
])
def test_content_tokens_are_not_structural(text):
    assert not is_structural(text), f"{text!r} should NOT be structural"


def test_structural_ids_are_sorted_and_unique(tokenizer):
    ids = structural_token_ids(tokenizer, vocab_size=VOCAB)
    assert ids.dtype == torch.long
    assert ids.tolist() == sorted(set(ids.tolist()))
    assert ids.numel() == 4, "the stub marks exactly four tokens structural"


def test_structural_ids_respect_the_model_vocab_width(tokenizer):
    """Qwen's embedding matrix is wider than its tokenizer; ids must be clipped.

    An id beyond the model's output width would index out of bounds when scattered
    into the logits tensor.
    """
    ids = structural_token_ids(tokenizer, vocab_size=2)
    assert ids.max() < 2


# ---------------------------------------------------------------- freezing
def test_base_model_is_frozen(base_model, tokenizer):
    """Only the heads may train. An unfrozen base invalidates the premise."""
    for parameter in base_model.parameters():
        parameter.requires_grad_(True)          # start dirty on purpose

    medusa = MedusaAST(base_model, tokenizer, num_heads=3)

    assert all(not p.requires_grad for p in medusa.base_model.parameters())
    assert all(p.requires_grad for p in medusa.medusa_heads.parameters())
    trainable = sum(p.numel() for p in medusa.parameters() if p.requires_grad)
    assert trainable == medusa.head_parameters
    assert medusa.frozen_parameters > 0


def test_head_count_and_shape(base_model, tokenizer):
    for num_heads in (1, 3, 15):
        medusa = MedusaAST(base_model, tokenizer, num_heads=num_heads)
        assert len(medusa.medusa_heads) == num_heads
        for head in medusa.medusa_heads:
            assert head.in_features == medusa.hidden_size
            assert head.out_features == medusa.n_structural


def test_full_mode_projects_to_the_whole_vocabulary(base_model, tokenizer):
    medusa = MedusaAST(base_model, tokenizer, num_heads=2, head_mode="full")
    for head in medusa.medusa_heads:
        assert head.out_features == medusa.vocab_size


def test_factored_heads_are_far_smaller(base_model, tokenizer):
    """The reason M=15 is feasible at all."""
    factored = MedusaAST(base_model, tokenizer, num_heads=4, head_mode="factored")
    full = MedusaAST(base_model, tokenizer, num_heads=4, head_mode="full")
    assert factored.head_parameters < full.head_parameters
    ratio = full.head_parameters / factored.head_parameters
    assert ratio == pytest.approx(VOCAB / factored.n_structural, rel=1e-6)


# ---------------------------------------------------------------- the AST mask
@pytest.mark.parametrize("head_mode", ["factored", "full"])
def test_non_structural_tokens_are_impossible(base_model, tokenizer, head_mode):
    """The defining claim: heads cannot propose an identifier.

    Checked on probabilities rather than logits, since -inf logits must translate to
    exactly zero probability for the restriction to be real.
    """
    medusa = MedusaAST(base_model, tokenizer, num_heads=5, head_mode=head_mode)
    hidden = torch.randn(1, medusa.hidden_size, dtype=torch.float16)

    logits = medusa.head_logits(hidden)
    assert logits.shape == (5, 1, medusa.vocab_size)

    allowed = torch.zeros(medusa.vocab_size, dtype=torch.bool)
    allowed[medusa.structural_ids] = True

    assert torch.isinf(logits[..., ~allowed]).all(), "blocked columns must be -inf"
    assert not torch.isinf(logits[..., allowed]).any(), "allowed columns must be finite"

    probs = torch.softmax(logits.float(), dim=-1)
    assert (probs[..., ~allowed] == 0).all(), "blocked tokens must have zero mass"
    assert torch.allclose(probs.sum(-1), torch.ones(5, 1), atol=1e-5)


def test_argmax_always_lands_on_a_structural_token(base_model, tokenizer):
    medusa = MedusaAST(base_model, tokenizer, num_heads=8)
    structural = set(medusa.structural_ids.tolist())
    for seed in range(5):
        torch.manual_seed(seed)
        hidden = torch.randn(1, medusa.hidden_size, dtype=torch.float16)
        for token in medusa.head_logits(hidden).argmax(dim=-1).flatten().tolist():
            assert token in structural


def test_factored_and_full_modes_are_equivalent(base_model, tokenizer):
    """Load-bearing: the cheap projection must be the same function as the mask.

    Weights are copied into the corresponding columns of the full head, so the two
    parameterise the same map; the resulting distributions must match to numerical
    precision. If they did not, the 26x VRAM saving would be buying a different
    model rather than the same one.
    """
    torch.manual_seed(0)
    factored = MedusaAST(base_model, tokenizer, num_heads=3, head_mode="factored")
    full = MedusaAST(base_model, tokenizer, num_heads=3, head_mode="full")
    with torch.no_grad():
        for compact, wide in zip(factored.medusa_heads, full.medusa_heads):
            wide.weight.zero_()
            wide.weight[factored.structural_ids] = compact.weight

    hidden = torch.randn(2, factored.hidden_size, dtype=torch.float16)
    a = torch.softmax(factored.head_logits(hidden).float(), dim=-1)
    b = torch.softmax(full.head_logits(hidden).float(), dim=-1)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


# ---------------------------------------------------------------- forward
def test_forward_returns_base_and_head_logits(base_model, tokenizer):
    medusa = MedusaAST(base_model, tokenizer, num_heads=6)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    base_logits, heads, cache = medusa.forward(ids)

    assert base_logits.shape[:2] == (1, 5)
    assert heads.shape == (6, 1, medusa.vocab_size)
    assert cache is not None


def test_propose_yields_one_plus_m_tokens(base_model, tokenizer):
    """The base head supplies an exact token; the M heads supply proposals."""
    medusa = MedusaAST(base_model, tokenizer, num_heads=15)
    ids = torch.tensor([[1, 2, 3, 4]])
    tokens, _ = medusa.propose(ids)

    assert tokens.shape == (16,), "1 exact token + 15 proposals"
    structural = set(medusa.structural_ids.tolist())
    # The first token comes from the *base* head and is unrestricted; the rest are
    # AST-restricted proposals.
    assert all(int(t) in structural for t in tokens[1:])


def test_head_logits_accepts_batched_hidden_states(base_model, tokenizer):
    medusa = MedusaAST(base_model, tokenizer, num_heads=4)
    for shape in ((1,), (3,), (2, 5)):
        hidden = torch.randn(*shape, medusa.hidden_size, dtype=torch.float16)
        out = medusa.head_logits(hidden)
        assert out.shape == (4, *shape, medusa.vocab_size)


def test_buffers_share_the_heads_device(base_model, tokenizer):
    """Everything the forward path touches must agree on device.

    Regression test for a real bug: `structural_ids` was registered as a CPU buffer
    while the heads were moved to CUDA. On CPU the two devices coincide, so this
    assertion is trivially true here and could not have caught it -- which is why
    the constructor also raises on mismatch.
    """
    for head_mode in ("factored", "full"):
        medusa = MedusaAST(base_model, tokenizer, num_heads=2, head_mode=head_mode)
        device = medusa.medusa_heads[0].weight.device
        assert medusa.structural_ids.device == device
        if head_mode == "full":
            assert medusa.blocked_mask.device == device


def test_invalid_arguments(base_model, tokenizer):
    with pytest.raises(ValueError, match="num_heads"):
        MedusaAST(base_model, tokenizer, num_heads=0)
    with pytest.raises(ValueError, match="head_mode"):
        MedusaAST(base_model, tokenizer, head_mode="sparse")


# ---------------------------------------------------------------- run analysis
def test_structural_run_analysis(tokenizer):
    """The ceiling metric: how far consecutive structural tokens run.

    Verified against a hand-built sequence so the accounting is unambiguous, since
    this number is what bounds a trained system's speedup.
    """
    from benchmark_medusa_limit import structural_runs

    class FixedTokenizer(VocabTokenizer):
        def __init__(self, ids):
            self._ids = ids

        def __call__(self, text, return_tensors=None):
            return type("E", (), {"input_ids": self._ids})()

    # ids 0-3 are structural, 4+ are content: S S C S S S C
    stub = FixedTokenizer([0, 1, 4, 2, 3, 0, 5])
    stats = structural_runs(stub, ["ignored"], cap=15)

    assert stats["tokens"] == 7
    assert stats["structural_fraction"] == pytest.approx(5 / 7)
    # Runs looking forward from each position: 1,0,3,2,1,0 -> mean 7/6
    assert stats["mean_run"] == pytest.approx(7 / 6)
    assert stats["max_run"] == 3


def test_run_analysis_respects_the_cap(tokenizer):
    """A run longer than M cannot yield more than M accepted proposals."""
    from benchmark_medusa_limit import structural_runs

    class AllStructural(VocabTokenizer):
        def __call__(self, text, return_tensors=None):
            return type("E", (), {"input_ids": [0] * 30})()

    stats = structural_runs(AllStructural(), ["x"], cap=5)
    assert stats["max_run"] == 5, "must not report runs beyond the head count"
    assert stats["structural_fraction"] == 1.0


# ---------------------------------------------------------------- 6k mask sim
def test_run_lengths_counts_forward_runs():
    """The coverage metric, checked against a hand-built sequence."""
    from benchmark_6k_medusa_limit import run_lengths

    # allowed = {1, 2}; sequence: A A X A A A X  (A=allowed, X=not)
    stats = run_lengths([1, 2, 9, 1, 2, 1, 9], allowed={1, 2}, cap=10)
    assert stats["tokens"] == 7
    assert stats["coverage"] == pytest.approx(5 / 7)
    # forward runs from each of the first 6 positions: 1,0,3,2,1,0
    assert stats["mean_run"] == pytest.approx(7 / 6)
    assert stats["max_run"] == 3
    assert stats["zero"] == pytest.approx(2 / 6)


def test_run_lengths_respects_the_cap():
    from benchmark_6k_medusa_limit import run_lengths

    stats = run_lengths([1] * 40, allowed={1}, cap=10)
    assert stats["max_run"] == 10, "cannot accept more proposals than there are heads"
    assert stats["coverage"] == 1.0
    # Not 100% at cap: the final positions have fewer than `cap` tokens left after
    # them, so their runs truncate. A boundary effect worth under 1% on a corpus of
    # a thousand tokens, but it means at_cap is a slight under-count by construction.
    assert 0.7 < stats["at_cap"] < 1.0


def test_run_lengths_with_nothing_allowed():
    """A mask that permits nothing must yield a ceiling of exactly 1x."""
    from benchmark_6k_medusa_limit import run_lengths

    stats = run_lengths([1, 2, 3, 4], allowed=set(), cap=10)
    assert stats["coverage"] == 0.0
    assert stats["mean_run"] == 0.0
    assert stats["zero"] == pytest.approx(1.0)


def test_vram_scales_with_output_dimension():
    from benchmark_6k_medusa_limit import vram_for

    small_params, small_mib = vram_for(3584, 6000, 10)
    big_params, big_mib = vram_for(3584, 152064, 10)
    assert small_params == pytest.approx(3584 * 6000 * 10 / 1e6)
    assert big_params / small_params == pytest.approx(152064 / 6000)
    # The whole point: a 6k head set is a rounding error, a full-vocab one is not.
    assert small_mib / 1024 < 1.0
    assert big_mib / 1024 > 10.0


def test_circularity_trap_is_real():
    """Ranking on the evaluation corpus itself inflates the ceiling to the cap.

    Guards the methodology rather than the code: if a future edit ranked
    frequencies on the eval corpus, this is the failure it would produce, and the
    test documents why the corpora must stay disjoint.
    """
    from collections import Counter

    from benchmark_6k_medusa_limit import run_lengths

    # A small corpus with few distinct types, as generated code actually is.
    corpus = [i % 40 for i in range(600)]
    circular = {t for t, _ in Counter(corpus).most_common(6000)}
    honest = {0, 1, 2}                       # a genuinely restrictive mask

    circular_stats = run_lengths(corpus, circular, cap=10)
    honest_stats = run_lengths(corpus, honest, cap=10)

    assert circular_stats["coverage"] == 1.0, "top-6k of 40 types covers everything"
    assert circular_stats["mean_run"] > 9.9, "so nearly every run pins at the cap"
    assert honest_stats["mean_run"] < 1.0
    ratio = (1 + circular_stats["mean_run"]) / (1 + honest_stats["mean_run"])
    assert ratio > 6.0, "the circular measurement inflates the ceiling several-fold"


# ---------------------------------------------------------------- training
def test_compact_targets_maps_into_the_masked_space():
    """Vocabulary ids must become indices in the head's output space."""
    from core.medusa_train import compact_targets

    allowed = torch.tensor([3, 7, 11])
    targets = torch.tensor([[3, 7, 11, 5]])       # 5 is outside the mask
    mapped = compact_targets(targets, allowed)
    assert mapped.tolist() == [[0, 1, 2, -100]], mapped.tolist()


def test_compact_targets_ignores_out_of_mask_positions():
    """Out-of-mask targets become -100 so cross-entropy skips them.

    There is no gradient direction that makes an impossible token likelier, but they
    must still count as failures at evaluation time -- otherwise accuracy would
    silently exclude the mask's own cost.
    """
    from core.medusa_train import compact_targets

    allowed = torch.tensor([1, 2])
    mapped = compact_targets(torch.tensor([[9, 9, 1]]), allowed)
    assert (mapped[0, :2] == -100).all()
    assert mapped[0, 2] == 0


def test_expected_accepted_is_the_chain_product():
    """A step commits the longest all-correct prefix, so accuracies multiply."""
    from core.medusa_train import expected_accepted

    assert expected_accepted([1.0, 1.0, 1.0]) == pytest.approx(3.0)
    assert expected_accepted([0.0, 1.0, 1.0]) == pytest.approx(0.0)
    # 0.5 + 0.25 + 0.125
    assert expected_accepted([0.5, 0.5, 0.5]) == pytest.approx(0.875)
    # One weak early head caps everything behind it, however good the tail.
    assert expected_accepted([0.1, 1.0, 1.0, 1.0]) == pytest.approx(0.4)


def test_projected_speedup_adds_the_free_base_token():
    """The base forward emits one exact token regardless of head performance."""
    from core.medusa_train import projected_speedup

    dead = projected_speedup([0.0] * 10)
    assert dead["speedup"] == pytest.approx(1.0), "useless heads must give 1.0x, not 0"

    perfect = projected_speedup([1.0] * 10)
    assert perfect["speedup"] == pytest.approx(11.0), "1 + M with perfect heads"

    projection = projected_speedup([0.9, 0.8, 0.5])
    assert projection["chain"] == pytest.approx([0.9, 0.72, 0.36])
    assert projection["tokens_per_forward"] == pytest.approx(1 + 0.9 + 0.72 + 0.36)


def test_low_rank_heads_are_far_smaller(base_model, tokenizer):
    """The bottleneck exists for data efficiency, so the saving must be real."""
    full = MedusaAST(base_model, tokenizer, num_heads=4, head_rank=None)
    low = MedusaAST(base_model, tokenizer, num_heads=4, head_rank=2)
    assert low.head_parameters < full.head_parameters
    hidden = torch.randn(3, low.hidden_size, dtype=torch.float16)
    assert low.head_logits(hidden).shape == (4, 3, low.vocab_size)


def test_low_rank_heads_still_respect_the_mask(base_model, tokenizer):
    medusa = MedusaAST(base_model, tokenizer, num_heads=3, head_rank=2)
    hidden = torch.randn(1, medusa.hidden_size, dtype=torch.float16)
    logits = medusa.head_logits(hidden)
    allowed = torch.zeros(medusa.vocab_size, dtype=torch.bool)
    allowed[medusa.structural_ids] = True
    assert torch.isinf(logits[..., ~allowed]).all()


def test_invalid_head_rank(base_model, tokenizer):
    with pytest.raises(ValueError, match="head_rank"):
        MedusaAST(base_model, tokenizer, head_rank=0)
