"""Self-speculative decoding: one model drafts and verifies itself.

Two-model speculative decoding needs a second checkpoint resident in VRAM. For a
7B target that means finding another ~3 GiB for a 1.5B draft, which on a 16 GB
card is the difference between fitting and not. Self-speculation removes the
second model entirely: the target drafts for itself, made cheaper by attending to
only a recent window of its own KV cache.

The Twin-Cache arrangement
--------------------------
The single model is driven through two separate KV caches:

* ``full_cache`` -- the authority. Holds the entire history and is used for the
  one verification forward per iteration. This is what defines the output
  distribution, so it must never be lossy.
* ``draft_cache`` -- the scratchpad. Holds only the most recent
  ``draft_window_size`` positions; older entries are evicted from the front. The
  draft is therefore a *different, weaker* model than the target -- same weights,
  truncated context -- which is exactly the asymmetry speculative decoding needs.

Because verification always runs on the full cache, the emitted distribution is
the target's own. `verify_tokens` is unchanged and the guarantee is unchanged: a
lossy draft costs acceptance, never correctness.

An honest note on where the win is
----------------------------------
The VRAM saving is real and large: the second model simply is not there.

The throughput case is weaker than it looks, and the reason is worth stating
rather than discovering later. Cropping the draft cache makes draft *attention*
cheap, but at batch size 1 attention is not what a decode forward spends its time
on. Reading the weights is -- roughly 8 GiB for this 8-bit 7B, against ~17 MiB of
KV cache for a 300-token context. Shrinking a 300-token cache to 64 removes about
0.2% of the memory traffic, so a windowed draft forward costs very nearly what a
full one costs.

That matters because the draft here *is* the target. Two-model speculation wins
by making the draft ~4.4x cheaper per forward; self-speculation via cache
truncation alone does not make it meaningfully cheaper at all, so K draft
forwards plus one verification forward buys at most K+1 tokens for K+1 forwards
-- which is what plain decoding already achieves. Methods that do win here make
the draft structurally cheaper (skipping layers, early exit, extra lightweight
heads) rather than merely shortening its context.

This module implements and measures the design as specified; see
results/benchmark_results_selfspec.md for what it actually does on real hardware.
"""

from __future__ import annotations

import time

import torch

from core.engine import (
    GenerationStats,
    SpeculativeEngine,
    _cache_length,
    _crop_cache,
)
from core.verifier import verify_tokens


def _evict_to_window(cache, window: int):
    """Drop the *oldest* entries so at most ``window`` positions remain.

    The counterpart of :func:`core.engine._crop_cache`, which drops the newest.
    Rollback needs the newest gone; windowing needs the oldest gone.

    Slices are made contiguous deliberately. A basic slice returns a view that
    keeps the whole original tensor alive, so the draft cache would never
    actually shrink and the bounded-memory property this function exists to
    provide would be quietly false.
    """
    if cache is None:
        return cache
    length = _cache_length(cache)
    if length <= window:
        return cache

    if hasattr(cache, "layers"):                      # transformers >= 5
        for layer in cache.layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys = layer.keys[:, :, -window:, :].contiguous()
                layer.values = layer.values[:, :, -window:, :].contiguous()
        return cache
    if hasattr(cache, "key_cache"):                   # transformers 4.x
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][:, :, -window:, :].contiguous()
            cache.value_cache[i] = cache.value_cache[i][:, :, -window:, :].contiguous()
        return cache
    return tuple(                                     # legacy tuple cache
        tuple(t[:, :, -window:, :].contiguous() for t in layer) for layer in cache
    )


class SelfSpeculativeEngine(SpeculativeEngine):
    """Speculative decoding with a single model drafting against itself.

    Args:
        target_model: the only model. It both drafts and verifies.
        tokenizer: its tokenizer.
        k: draft length.
        draft_window_size: how many recent positions the draft may attend to.
            Smaller is a weaker draft and a lower acceptance rate; large enough to
            exceed the context makes the draft identical to the target, so
            acceptance goes to 100% and no speculation happens at all.
        temperature, top_p: as in :class:`core.engine.SpeculativeEngine`, applied
            identically to draft and verify (here trivially so -- same model).

    The parent is initialised with the target as *both* models, which is not a
    trick: under self-speculation the draft model and the target model are the
    same weights. The difference lives entirely in the cache.
    """

    def __init__(
        self,
        target_model,
        tokenizer,
        k: int = 5,
        draft_window_size: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        if draft_window_size < 1:
            raise ValueError(f"draft_window_size must be >= 1, got {draft_window_size}")
        super().__init__(
            draft_model=target_model,
            target_model=target_model,
            tokenizer=tokenizer,
            k=k,
            temperature=temperature,
            top_p=top_p,
        )
        self.draft_window_size = draft_window_size

    # ---------------------------------------------------------------- phases
    @torch.no_grad()
    def _draft_windowed(self, context: torch.Tensor, cache, cache_start: int, k: int):
        """Propose ``k`` tokens using only the last ``draft_window_size`` positions.

        ``cache_start`` is the absolute position of the cache's first entry. The
        windowed cache holds a *suffix* of history, so cache index and true
        position diverge, and both have to be tracked.

        RoPE is the subtle part. Cached keys were rotated at their original
        absolute positions and eviction does not re-rotate them, so a new query
        must also carry its true absolute position or the relative offsets that
        RoPE actually encodes come out wrong. Getting this wrong does not crash --
        it silently destroys the draft's quality, visible only as an acceptance
        rate that collapses toward zero.

        Returns ``(tokens, q, cache, cache_start, n_forwards)``.
        """
        covered = cache_start + _cache_length(cache)
        pending = context[:, covered:]

        tokens: list[torch.Tensor] = []
        q_rows: list[torch.Tensor] = []
        forwards = 0

        for _ in range(k):
            n_new = pending.shape[1]
            position_ids = torch.arange(
                covered, covered + n_new, device=pending.device
            ).unsqueeze(0)

            logits, cache = self._forward(
                self.target_model, pending, cache, position_ids=position_ids
            )
            forwards += 1
            covered += n_new

            probs = self._logits_to_probs(logits[0, -1])
            token = self._sample(probs)
            tokens.append(token)
            q_rows.append(probs)
            pending = token.unsqueeze(0)

            # Bound the scratchpad. Done after each forward so the cache never
            # exceeds the window even transiently during a long prefill.
            cache = _evict_to_window(cache, self.draft_window_size)
            cache_start = covered - _cache_length(cache)

        return torch.cat(tokens), torch.stack(q_rows), cache, cache_start, forwards

    @torch.no_grad()
    def _verify_full(self, context: torch.Tensor, draft_tokens: torch.Tensor, cache):
        """Score the drafted block against the *complete* history, in one forward.

        The full cache is a contiguous prefix from position 0, so cache indices
        and true positions coincide and ``position_ids`` can be left implicit.
        """
        covered = _cache_length(cache)
        pending = context[:, covered:]
        fed = torch.cat([pending, draft_tokens.unsqueeze(0)], dim=1)

        logits, cache = self._forward(self.target_model, fed, cache)

        k_plus_1 = draft_tokens.shape[0] + 1
        return self._logits_to_probs(logits[0, -k_plus_1:]), cache

    # ---------------------------------------------------------------- public
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        stream: bool = True,
        stop_at_eos: bool = True,
    ) -> tuple[str, GenerationStats]:
        """Generate by self-speculation. Same contract as the two-model engine."""
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        if self.use_dual_gate:
            raise NotImplementedError(
                "the dual gate is not wired into the self-speculative loop; "
                "construct SelfSpeculativeEngine without it"
            )

        encoded = self.tokenizer(prompt, return_tensors="pt")
        context = encoded.input_ids.to(self.device)
        prompt_len = context.shape[1]

        eos_ids = set()
        if stop_at_eos and self.tokenizer.eos_token_id is not None:
            eos = self.tokenizer.eos_token_id
            eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}

        full_cache = None
        draft_cache = None
        draft_start = 0
        generated: list[int] = []
        printed_chars = 0
        stats = GenerationStats()
        start = time.perf_counter()

        while len(generated) < max_new_tokens:
            k = min(self.k, max_new_tokens - len(generated))

            # DRAFT on the windowed cache.
            draft_tokens, q, draft_cache, draft_start, n_draft_fwd = self._draft_windowed(
                context, draft_cache, draft_start, k
            )

            # VERIFY on the full cache, one forward.
            p, full_cache = self._verify_full(context, draft_tokens, full_cache)

            accepted_tokens, n_accepted = verify_tokens(q, p, draft_tokens)

            stats.iterations += 1
            stats.target_forwards += 1
            stats.draft_forwards += n_draft_fwd
            stats.draft_tokens_proposed += draft_tokens.shape[0]
            stats.draft_tokens_accepted += n_accepted
            stats.accepted_per_iteration.append(n_accepted)
            stats.draft_lengths.append(draft_tokens.shape[0])

            new_ids = accepted_tokens.tolist()[: max_new_tokens - len(generated)]

            hit_eos = False
            for i, token_id in enumerate(new_ids):
                if token_id in eos_ids:
                    new_ids = new_ids[:i]
                    hit_eos = True
                    break

            generated.extend(new_ids)
            stats.tokens_generated = len(generated)

            if stream:
                text_so_far = self.tokenizer.decode(generated, skip_special_tokens=True)
                if len(text_so_far) > printed_chars:
                    print(text_so_far[printed_chars:], end="", flush=True)
                    printed_chars = len(text_so_far)

            if new_ids:
                context = torch.cat(
                    [
                        context,
                        torch.tensor([new_ids], dtype=context.dtype, device=self.device),
                    ],
                    dim=1,
                )

            if hit_eos:
                break

            # ROLLBACK both caches to the committed prefix. Same bound as the
            # two-model engine: every committed token except the last, which no
            # forward has seen.
            valid_len = context.shape[1] - 1
            full_cache = _crop_cache(full_cache, valid_len)

            # The draft cache holds [draft_start, ...), so its own length limit is
            # measured from that offset rather than from zero.
            draft_keep = valid_len - draft_start
            if draft_keep <= 0:
                # The window slid past everything still valid. Cheaper to rebuild
                # than to reason about a cache with nothing usable left in it.
                draft_cache, draft_start = None, 0
            else:
                draft_cache = _crop_cache(draft_cache, draft_keep)

        stats.seconds = time.perf_counter() - start
        stats.token_ids = list(generated)
        if stream:
            print(flush=True)

        assert context.shape[1] == prompt_len + len(generated), (
            f"context ({context.shape[1]}) and generated ({len(generated)}) "
            f"disagree for a {prompt_len}-token prompt"
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True), stats
