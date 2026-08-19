"""Monte Carlo speculative decoding: draft B futures in parallel, keep the longest.

Batch-1 decoding wastes the GPU. A 7B int8 forward reads ~8 GiB of weights to
compute a handful of token positions, so arithmetic intensity is terrible and
widening the batch is nearly free. This engine spends that headroom on
*speculation breadth*: sample B divergent draft continuations, verify all of them
in one batched target forward, and commit whichever got furthest.

The draft is sampled at an elevated temperature on purpose. At draft temperature 0
all B branches would be identical and B would buy exactly nothing; the branches
have to actually explore different continuations for the max over them to beat a
single sample.

Why selecting the longest branch is lossless here -- and only here
----------------------------------------------------------------
Picking the branch with the most accepted tokens conditions on the acceptance
outcome, and in general that is *not* distribution-preserving. Acceptance
correlates with which token was proposed, so favouring accepted branches
over-samples ``norm(min(p, q))`` and throws away the residual mass that makes
speculative decoding exact. With a sampled target this engine would quietly bias
generation.

At target temperature 0 the bias vanishes, for a specific reason. ``p`` is one-hot
at the target's argmax ``a``, so a proposal is accepted only if it *equals* ``a``,
and when it is rejected the residual ``max(0, p - q)`` is also one-hot at ``a``.
Either way the branch emits ``a``. Inductively, a branch with ``n_accepted = j``
emits exactly the target's greedy continuation of length ``j + 1``. Every branch
therefore emits a *prefix of the same string*, and choosing the longest prefix of
one string cannot change its content -- only how much of it is committed per
iteration.

So B is a pure throughput knob at temperature 0, and a correctness hazard above it.
The constructor refuses temperature > 0 rather than silently biasing: unbiased
selection there would have to ignore the acceptance counts (say, always take branch
0), which would make B pointless anyway.

A note on the two temperatures
------------------------------
The draft samples at ``draft_temperature`` while the target is evaluated at 0. That
looks like it contradicts the rule that warping must be applied identically to both
models, but it does not. That rule is about *user-facing semantics* -- asking for
temperature 0.7 should give you samples from the 0.7-warped target. The
*correctness* requirement is narrower: ``q`` must be the true density the proposals
were drawn from, and ``p`` must be the distribution you want to emit. Here both hold,
so the output is the greedy target sequence regardless of how wild the draft gets.
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


def _expand_cache(cache, repeats: int):
    """Widen a batch-1 KV cache to ``repeats`` identical rows."""
    if cache is None or repeats == 1:
        return cache
    if hasattr(cache, "batch_repeat_interleave"):
        cache.batch_repeat_interleave(repeats)
        return cache
    return tuple(
        tuple(t.repeat_interleave(repeats, dim=0) for t in layer) for layer in cache
    )


def _select_cache_row(cache, index: int):
    """Collapse a batched KV cache down to a single row."""
    if cache is None:
        return cache
    if hasattr(cache, "batch_select_indices"):
        cache.batch_select_indices(
            torch.tensor([index], device=_cache_device(cache))
        )
        return cache
    return tuple(
        tuple(t[index: index + 1] for t in layer) for layer in cache
    )


def _cache_device(cache):
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if getattr(layer, "keys", None) is not None:
                return layer.keys.device
    return "cpu"


class MonteCarloEngine(SpeculativeEngine):
    """Speculative decoding that drafts ``branches`` futures and commits the best.

    Args:
        draft_model, target_model, tokenizer: as in the base engine.
        k: tokens drafted per branch.
        branches: how many parallel futures to explore (``B``). ``1`` reduces to
            ordinary speculative decoding with a hot draft.
        draft_temperature: sampling temperature for the *draft only*. Must be > 0,
            since identical branches make B meaningless.
        temperature: target sampling temperature. Must be 0 -- see the module
            docstring for why selection is only unbiased there.
    """

    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        k: int = 5,
        branches: int = 4,
        draft_temperature: float = 1.2,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        if temperature != 0.0:
            raise ValueError(
                "MonteCarloEngine requires temperature=0. Selecting the "
                "longest-accepted branch conditions on the acceptance outcome, "
                "which is only distribution-preserving when the target is greedy "
                f"(got temperature={temperature}). See the module docstring."
            )
        if branches < 1:
            raise ValueError(f"branches must be >= 1, got {branches}")
        if draft_temperature <= 0:
            raise ValueError(
                "draft_temperature must be > 0, otherwise every branch samples "
                f"the same tokens and B buys nothing (got {draft_temperature})"
            )
        super().__init__(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            k=k,
            temperature=0.0,
            top_p=top_p,
        )
        self.branches = branches
        self.draft_temperature = draft_temperature

    # ---------------------------------------------------------------- phases
    def _draft_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """The draft's true sampling density, at its own elevated temperature.

        Deliberately not ``_logits_to_probs``: that would apply the *target's*
        temperature of 0 and collapse every branch onto the same token.
        """
        sliced = logits[..., : self.vocab_size].to(self._verify_dtype)
        return torch.softmax(sliced / self.draft_temperature, dim=-1)

    @torch.no_grad()
    def _draft_branches(self, context: torch.Tensor, cache, k: int):
        """Sample ``branches`` divergent continuations of ``k`` tokens each.

        Costs k sequential forwards, same as single-path drafting -- but each is a
        batch-``B`` forward, which at these sizes costs barely more than batch 1.
        That is the entire bet.

        Returns ``(tokens [B, k], q [B, k, V], cache, n_forwards)``.
        """
        branches = self.branches
        cache = _expand_cache(cache, branches)
        pending = context[:, _cache_length(cache):].repeat(branches, 1)

        tokens: list[torch.Tensor] = []
        q_rows: list[torch.Tensor] = []
        forwards = 0

        for _ in range(k):
            logits, cache = self._forward(self.draft_model, pending, cache)
            forwards += 1
            probs = self._draft_probs(logits[:, -1])        # [B, V]
            token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            tokens.append(token)
            q_rows.append(probs)
            pending = token

        return (
            torch.cat(tokens, dim=1),           # [B, k]
            torch.stack(q_rows, dim=1),         # [B, k, V]
            cache,
            forwards,
        )

    @torch.no_grad()
    def _verify_branches(self, context: torch.Tensor, draft_tokens: torch.Tensor, cache):
        """Score all branches against the target in ONE batched forward.

        Returns ``(p [B, k + 1, V], cache)``.
        """
        branches, k = draft_tokens.shape
        cache = _expand_cache(cache, branches)
        pending = context[:, _cache_length(cache):].repeat(branches, 1)
        fed = torch.cat([pending, draft_tokens], dim=1)

        logits, cache = self._forward(self.target_model, fed, cache)
        return self._logits_to_probs(logits[:, -(k + 1):]), cache

    # ---------------------------------------------------------------- public
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        stream: bool = True,
        stop_at_eos: bool = True,
    ) -> tuple[str, GenerationStats]:
        """Generate by parallel-branch speculation. Same contract as the base engine."""
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        if self.use_dual_gate:
            raise NotImplementedError(
                "the dual gate is not wired into the Monte Carlo loop; its "
                "early-exit decision would have to be taken per branch"
            )

        encoded = self.tokenizer(prompt, return_tensors="pt")
        context = encoded.input_ids.to(self.device)
        prompt_len = context.shape[1]

        eos_ids = set()
        if stop_at_eos and self.tokenizer.eos_token_id is not None:
            eos = self.tokenizer.eos_token_id
            eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}

        draft_cache = None
        target_cache = None
        generated: list[int] = []
        printed_chars = 0
        stats = GenerationStats()
        start = time.perf_counter()

        while len(generated) < max_new_tokens:
            k = min(self.k, max_new_tokens - len(generated))

            draft_tokens, q, draft_cache, n_draft_fwd = self._draft_branches(
                context, draft_cache, k
            )
            p, target_cache = self._verify_branches(context, draft_tokens, target_cache)

            # Verify every branch independently with the Phase 1 math, then keep
            # the one that got furthest. At temperature 0 all branches emit
            # prefixes of the same greedy string, so this is a pure length choice.
            outcomes = [
                verify_tokens(q[b], p[b], draft_tokens[b])
                for b in range(draft_tokens.shape[0])
            ]
            accepted_counts = [n for _, n in outcomes]
            winner = max(range(len(accepted_counts)), key=accepted_counts.__getitem__)
            accepted_tokens, n_accepted = outcomes[winner]

            stats.iterations += 1
            stats.target_forwards += 1
            stats.draft_forwards += n_draft_fwd
            stats.draft_row_forwards += n_draft_fwd * self.branches
            stats.draft_tokens_proposed += k
            stats.draft_tokens_accepted += n_accepted
            stats.accepted_per_iteration.append(n_accepted)
            stats.draft_lengths.append(k)
            stats.branch_accepted.append(list(accepted_counts))
            stats.branch_wins.append(winner)
            # Recorded for comparability with the particle filter, which prunes.
            # Independent sampling should keep nearly all B branches distinct; a
            # low count here would mean the draft temperature is too cold to
            # diverge and B is being wasted.
            stats.unique_candidates.append(
                len({tuple(row) for row in draft_tokens.tolist()})
            )

            # Collapse the batch back to the winning row *before* the rollback, so
            # the caches are single-row again and the usual crop applies unchanged.
            draft_cache = _select_cache_row(draft_cache, winner)
            target_cache = _select_cache_row(target_cache, winner)

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

            valid_len = context.shape[1] - 1
            draft_cache = _crop_cache(draft_cache, valid_len)
            target_cache = _crop_cache(target_cache, valid_len)

        stats.seconds = time.perf_counter() - start
        stats.token_ids = list(generated)
        if stream:
            print(flush=True)

        assert context.shape[1] == prompt_len + len(generated), (
            f"context ({context.shape[1]}) and generated ({len(generated)}) "
            f"disagree for a {prompt_len}-token prompt"
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True), stats
