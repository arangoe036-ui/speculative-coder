"""Particle filter speculative decoding: prune weak drafts, clone strong ones.

Monte Carlo speculation draws B independent draft continuations and keeps whichever
got furthest. Its weakness is that several branches wander into low-probability
territory immediately and then spend the remaining K-1 steps being useless. This
engine resamples between steps: after every drafted token the particles are ranked
by cumulative draft log-probability, the weakest half are discarded, and the
strongest half are cloned into their slots. Compute concentrates on the paths the
draft model actually believes in.

Structurally this is stochastic beam search rather than a particle filter in the
inference sense. A real particle filter reweights on *observations*; here there is
no observation during drafting, because consulting the target is precisely the
expensive thing being deferred. The weights are the draft's own opinion of itself.

That distinction predicts the failure mode. Resampling toward high draft
probability pulls every particle toward the draft's own mode -- which is exactly
what a single greedy draft would produce -- so it can destroy the *diversity* that
gives Monte Carlo its edge. Clones do re-diverge on the next sampling step, since
each samples independently at T > 1, so the outcome is a genuine trade: higher
per-particle quality against lower spread between particles. Which wins is an
empirical question, and `branch_gain` in the telemetry is the number that answers
it.

Why the draft may be biased without breaking anything
-----------------------------------------------------
Resampling means a surviving particle's token sequence was selected *for* its
probability, so its marginal is no longer the product of the per-step ``q`` values.
Ordinarily that would invalidate the verifier, which assumes the proposal really was
drawn from ``q``.

At target temperature 0 it is irrelevant. ``p`` is one-hot at the target's argmax
``a``: a proposal is accepted only if it equals ``a``, and if it is rejected the
residual ``max(0, p - q)`` is also one-hot at ``a``. Either way the particle emits
``a``. So every particle emits a prefix of the target's greedy continuation
regardless of how its tokens were chosen, and both the resampling and the
argmax-over-particles selection can only change *how many* tokens are committed per
iteration -- never which ones.

Above temperature 0 both steps would bias generation, so the constructor refuses it.
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
from core.monte_carlo_engine import _expand_cache
from core.verifier import verify_tokens


def _reorder_cache(cache, indices: torch.Tensor):
    """Rearrange a batched KV cache by ``indices``, duplicating rows as needed.

    This is the clone step: an index appearing twice yields two identical particles.
    Not free -- it copies the whole draft cache once per resampling step, which is
    real memory traffic and is part of what the method has to earn back.
    """
    if cache is None:
        return cache
    if hasattr(cache, "batch_select_indices"):
        cache.batch_select_indices(indices)
        return cache
    return tuple(
        tuple(t.index_select(0, indices) for t in layer) for layer in cache
    )


class ParticleFilterEngine(SpeculativeEngine):
    """Speculative decoding with resampled (pruned and cloned) draft particles.

    Args:
        particles: number of parallel draft particles (``B``).
        k: draft depth.
        draft_temperature: must exceed 1.0 in spirit -- particles have to explore,
            and resampling already pulls them back toward the mode.
        survivor_fraction: share of particles kept at each resampling step. ``0.5``
            is the specified "keep the top half".
        temperature: target temperature. Must be 0; see the module docstring.
    """

    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        k: int = 8,
        particles: int = 8,
        draft_temperature: float = 1.2,
        survivor_fraction: float = 0.5,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        if temperature != 0.0:
            raise ValueError(
                "ParticleFilterEngine requires temperature=0. Resampling biases the "
                "proposal density and argmax selection conditions on the acceptance "
                "outcome; both are only harmless when the target is greedy (got "
                f"temperature={temperature}). See the module docstring."
            )
        if particles < 2:
            raise ValueError(
                f"particles must be >= 2 for resampling to mean anything, got {particles}"
            )
        if draft_temperature <= 0:
            raise ValueError(f"draft_temperature must be > 0, got {draft_temperature}")
        if not 0.0 < survivor_fraction < 1.0:
            raise ValueError(
                f"survivor_fraction must be in (0, 1), got {survivor_fraction}"
            )
        super().__init__(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            k=k,
            temperature=0.0,
            top_p=top_p,
        )
        self.particles = particles
        self.draft_temperature = draft_temperature
        self.survivor_fraction = survivor_fraction

    @property
    def n_survivors(self) -> int:
        """Particles kept at each resampling step; at least one, at most B - 1."""
        keep = int(round(self.particles * self.survivor_fraction))
        return max(1, min(self.particles - 1, keep))

    # ---------------------------------------------------------------- phases
    def _draft_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """The draft's per-step sampling density, at its own temperature."""
        sliced = logits[..., : self.vocab_size].to(self._verify_dtype)
        return torch.softmax(sliced / self.draft_temperature, dim=-1)

    @torch.no_grad()
    def _draft_particles(self, context: torch.Tensor, cache, k: int):
        """Draft ``k`` tokens with resampling between steps.

        Returns ``(tokens [B, k], q [B, k, V], cache, n_forwards, n_resamples)``.
        """
        branches = self.particles
        cache = _expand_cache(cache, branches)
        pending = context[:, _cache_length(cache):].repeat(branches, 1)

        tokens = torch.empty(
            (branches, 0), dtype=torch.long, device=context.device
        )
        q_rows: torch.Tensor | None = None
        log_weights = torch.zeros(
            branches, dtype=self._verify_dtype, device=context.device
        )
        forwards = 0
        resamples = 0

        for step in range(k):
            logits, cache = self._forward(self.draft_model, pending, cache)
            forwards += 1
            probs = self._draft_probs(logits[:, -1])              # [B, V]
            token = torch.multinomial(probs, num_samples=1)        # [B, 1]

            tokens = torch.cat([tokens, token], dim=1)
            row = probs.unsqueeze(1)                               # [B, 1, V]
            q_rows = row if q_rows is None else torch.cat([q_rows, row], dim=1)

            # Accumulate the log-probability of the path actually taken.
            chosen = probs.gather(1, token).squeeze(1).clamp_min(1e-30)
            log_weights = log_weights + chosen.log()

            pending = token

            # Resample between steps, but never after the last one. A final
            # resample would only duplicate particles just before verification,
            # shrinking the number of distinct candidates at the exact moment
            # breadth is worth the most.
            if step < k - 1:
                order = torch.argsort(log_weights, descending=True)
                survivors = order[: self.n_survivors]
                # Tile the survivors back up to B slots: the strongest particles
                # are kept and cloned over the weakest.
                repeats = (branches + survivors.numel() - 1) // survivors.numel()
                index = survivors.repeat(repeats)[:branches].contiguous()

                tokens = tokens[index]
                q_rows = q_rows[index]
                log_weights = log_weights[index]
                pending = pending[index]
                cache = _reorder_cache(cache, index)
                resamples += 1

        return tokens, q_rows, cache, forwards, resamples

    @torch.no_grad()
    def _verify_candidates(self, context: torch.Tensor, candidates: torch.Tensor, cache):
        """Score distinct candidate drafts against the target in one forward."""
        n_candidates, k = candidates.shape
        cache = _expand_cache(cache, n_candidates)
        pending = context[:, _cache_length(cache):].repeat(n_candidates, 1)
        fed = torch.cat([pending, candidates], dim=1)

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
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        if self.use_dual_gate:
            raise NotImplementedError(
                "the dual gate is not wired into the particle filter loop"
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

            tokens, q, draft_cache, n_draft_fwd, n_resamples = self._draft_particles(
                context, draft_cache, k
            )

            # Verify only distinct drafts. Resampling makes duplicates common, and
            # scoring the same sequence twice buys nothing.
            unique, inverse = torch.unique(tokens, dim=0, return_inverse=True)
            representative = torch.zeros(
                unique.shape[0], dtype=torch.long, device=tokens.device
            )
            representative.scatter_(
                0, inverse, torch.arange(tokens.shape[0], device=tokens.device)
            )
            unique_q = q[representative]

            p, target_cache = self._verify_candidates(context, unique, target_cache)

            outcomes = [
                verify_tokens(unique_q[i], p[i], unique[i])
                for i in range(unique.shape[0])
            ]
            accepted_counts = [n for _, n in outcomes]
            winner = max(range(len(accepted_counts)), key=accepted_counts.__getitem__)
            accepted_tokens, n_accepted = outcomes[winner]

            stats.iterations += 1
            stats.target_forwards += 1
            stats.draft_forwards += n_draft_fwd
            stats.draft_row_forwards += n_draft_fwd * self.particles
            stats.draft_tokens_proposed += k
            stats.draft_tokens_accepted += n_accepted
            stats.accepted_per_iteration.append(n_accepted)
            stats.draft_lengths.append(k)
            stats.branch_accepted.append(list(accepted_counts))
            stats.branch_wins.append(winner)
            stats.resample_steps += n_resamples
            stats.unique_candidates.append(int(unique.shape[0]))

            # Collapse both caches to the winning path. The draft cache is indexed
            # by *particle*, so the unique-row winner has to be mapped back.
            draft_cache = _reorder_cache(
                draft_cache, representative[winner: winner + 1]
            )
            target_cache = _reorder_cache(
                target_cache, torch.tensor([winner], device=self.device)
            )

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
