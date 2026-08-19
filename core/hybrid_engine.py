"""Hybrid cascade: a free CPU n-gram drafter in front of the GPU draft model.

Two drafters of very different cost, tried in order of cheapness:

1. **Fast path** -- :func:`core.ngram.find_ngram_draft` looks up the last few
   tokens elsewhere in the context and proposes whatever followed them. A tensor
   comparison, microseconds, no weights. When it hits, the GPU draft model is
   skipped entirely and that iteration costs one target forward and nothing else.
2. **Fallback** -- the 0.5B draft model with the dual gate, exactly as before.

The cascade only ever changes *which* tokens get proposed, never how they are
verified, so :func:`core.verifier.verify_tokens` and its guarantee are untouched.
Both drafters are allowed to be wrong; being wrong costs acceptance.

The economics
-------------
A target forward is ~73 ms on this hardware and a 0.5B draft forward ~6 ms, so a
K=5 model-drafted iteration spends roughly 73 + 30 = 103 ms while an n-gram-drafted
one spends ~73 ms. A fast-path hit is therefore worth about 29% of an iteration --
*provided* its proposals are accepted at a comparable rate. That proviso is the
whole question, because a short pattern matches constantly and predicts poorly, so
hit rate and acceptance rate have to be reported separately. `min_match_len` is the
knob that trades one against the other.
"""

from __future__ import annotations

import torch

from core.engine import SpeculativeEngine
from core.ngram import find_ngram_draft, one_hot_q

_NO_GATES = {"statistical": 0, "syntactic": 0, "confidences": []}


class HybridCascadeEngine(SpeculativeEngine):
    """Speculative decoding that routes between an n-gram drafter and a model.

    Args:
        draft_model: the GPU fallback drafter.
        target_model: the model whose distribution is preserved.
        tokenizer: shared tokenizer.
        k: draft length for the fallback path (a ceiling under the dual gate).
        ngram_draft_len: how many tokens the fast path may propose.
        min_match_len: pattern length for the lookup. Larger means fewer but
            better-founded matches.
        use_dual_gate, max_draft_len, entropy_threshold: passed to the fallback,
            defaulting to the calibrated configuration that won the last sweep.

    Everything else is inherited. Only :meth:`_propose` differs, which is the
    point: routing is the only new idea, so it is the only new code.
    """

    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        k: int = 5,
        temperature: float = 1.0,
        top_p: float = 1.0,
        use_dual_gate: bool = True,
        max_draft_len: int = 8,
        entropy_threshold: float = 0.65,
        ngram_draft_len: int = 5,
        min_match_len: int = 2,
    ):
        if ngram_draft_len < 1:
            raise ValueError(f"ngram_draft_len must be >= 1, got {ngram_draft_len}")
        if min_match_len < 1:
            raise ValueError(f"min_match_len must be >= 1, got {min_match_len}")
        super().__init__(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            k=k,
            temperature=temperature,
            top_p=top_p,
            use_dual_gate=use_dual_gate,
            max_draft_len=max_draft_len,
            entropy_threshold=entropy_threshold,
        )
        self.ngram_draft_len = ngram_draft_len
        self.min_match_len = min_match_len

    @torch.no_grad()
    def _propose(self, context: torch.Tensor, cache, k: int, syntax=None):
        """Try the free drafter first; fall back to the model when it misses."""
        # The match runs on CPU as specified. The transfer is a few hundred
        # integers -- immaterial against a ~73 ms target forward -- and it keeps
        # the lookup off the GPU's critical path entirely.
        tokens = find_ngram_draft(
            context.reshape(-1).cpu(),
            max_draft_len=min(self.ngram_draft_len, k),
            min_match_len=self.min_match_len,
        )

        if tokens is not None:
            tokens = tokens.to(self.device)
            # A deterministic proposal is a one-hot q. This is exact, not an
            # approximation -- see core.ngram for the derivation. The draft cache
            # is returned untouched: the model never ran, so it has simply fallen
            # behind, and `_draft` re-feeds whatever the cache has not seen.
            q = one_hot_q(tokens, self.vocab_size, dtype=self._verify_dtype)
            return tokens, q, cache, 0, dict(_NO_GATES), "ngram"

        return super()._propose(context, cache, k, syntax)
