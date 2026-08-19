"""Speculative Jacobi decoding: greedy generation as fixed-point relaxation.

Greedy autoregressive decoding is the solution of a triangular non-linear system.
For a block of B tokens following context ``x``:

    y_i = argmax p( . | x, y_1 .. y_{i-1} )        for i = 1 .. B

Autoregressive decoding solves this by forward substitution -- B sequential forwards.
Jacobi iteration instead guesses the whole block, applies the map to every position
*at once* in a single forward, and repeats until the block stops changing. A block
that stops changing is a fixed point, and a fixed point of this map is exactly the
greedy continuation.

No rejection sampling
---------------------
Every other engine here wraps :func:`core.verifier.verify_tokens`. This one does not
need it, because the iteration is exact by construction rather than in distribution.

Position 1's logits condition only on ``x``, so ``y_1`` is the true greedy token from
the very first iteration. Position 2 then conditions on a correct token 1, so it is
right from the second iteration. Inductively, **after k iterations the first k tokens
are exactly what autoregressive greedy decoding would have produced.** And if the
block stops changing entirely, self-consistency from position 1 forces the whole
block to be the greedy continuation.

The same argument gives the commit rule. If iteration ``k`` and ``k-1`` agree on the
first ``j`` positions, then for every ``i <= j`` we have
``y_i = argmax p(. | x, y_1..y_{i-1})`` with the conditioning prefix drawn from the
*same* sequence -- the greedy fixed-point condition restricted to the first j
positions. So the longest common prefix is committable.

Why the commit rule needs the iteration count too
-------------------------------------------------
Committing only the longest common prefix can commit **zero** tokens: if the initial
guess is wrong at position 1, the new block differs there and the prefix is empty --
even though position 1 is *provably* correct. A loop that commits nothing makes no
progress and never terminates.

So this commits ``max(longest_common_prefix, iterations_performed)``. Both bounds are
sound, the maximum is therefore sound, and taking it guarantees at least one token per
iteration. That floor is also what bounds the method: B forwards for B tokens is
exactly autoregressive cost, so Jacobi can never be *slower* in forward count than the
decoder it replaces, and everything above 1 token per iteration is profit.

Temperature
-----------
The argument above is specific to argmax. Sampled decoding would need a probabilistic
acceptance test on top of the iteration -- which is what the literature means by
*speculative* Jacobi -- so the constructor refuses ``temperature > 0`` rather than
silently emitting from the wrong distribution.
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
from core.ngram import find_ngram_draft


def longest_common_prefix(a: torch.Tensor, b: torch.Tensor) -> int:
    """Length of the shared leading run of two 1-D token tensors."""
    limit = min(a.numel(), b.numel())
    if limit == 0:
        return 0
    same = (a[:limit] == b[:limit])
    if bool(same.all()):
        return limit
    return int(same.long().argmin())


class JacobiEngine(SpeculativeEngine):
    """Greedy decoding by Jacobi relaxation over a block of tokens.

    Args:
        target_model: the only model. There is no drafter, so this runs at the same
            VRAM as plain decoding -- no second checkpoint resident.
        tokenizer: its tokenizer.
        block_size: B, how many positions are relaxed at once.
        max_iterations: cap on relaxation sweeps per block. Since each sweep costs
            one forward and guarantees one token, a cap of B makes the worst case
            exactly autoregressive.
        min_match_len: pattern length for the n-gram initial guess.
        temperature: must be 0; see the module docstring.
    """

    def __init__(
        self,
        target_model,
        tokenizer,
        block_size: int = 10,
        max_iterations: int = 5,
        min_match_len: int = 2,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        if temperature != 0.0:
            raise ValueError(
                "JacobiEngine requires temperature=0. The fixed-point argument is "
                "specific to argmax; sampled decoding needs a probabilistic "
                f"acceptance test layered on top (got temperature={temperature})."
            )
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        super().__init__(
            draft_model=target_model,
            target_model=target_model,
            tokenizer=tokenizer,
            k=block_size,
            temperature=0.0,
            top_p=top_p,
        )
        self.block_size = block_size
        self.max_iterations = max_iterations
        self.min_match_len = min_match_len

    # ---------------------------------------------------------------- guess
    def initial_block(self, context: torch.Tensor, size: int) -> tuple[torch.Tensor, str]:
        """Seed the relaxation, preferring a prompt-lookup guess.

        A better starting point converges in fewer sweeps, and the n-gram lookup is
        free. When it misses, the last token is repeated -- a deliberately poor guess
        whose only virtue is being valid, since the iteration is exact regardless of
        where it starts. Only the *speed* depends on the seed.
        """
        guess = find_ngram_draft(
            context.reshape(-1).cpu(),
            max_draft_len=size,
            min_match_len=self.min_match_len,
        )
        if guess is not None and guess.numel() == size:
            return guess.to(self.device), "ngram"
        last = context[0, -1]
        return last.repeat(size), "repeat"

    # ---------------------------------------------------------------- sweep
    @torch.no_grad()
    def _relax(self, context: torch.Tensor, block: torch.Tensor, cache):
        """One Jacobi sweep: apply the greedy map to every block position at once.

        Feeds ``[last context token, block...]`` so the logits line up: position j's
        logits predict position j+1, giving a fresh value for every block slot plus a
        bonus prediction for the slot just past the block.

        Returns ``(new_block, bonus_token, cache)``.
        """
        pending = context[:, _cache_length(cache):]
        fed = torch.cat([pending, block.unsqueeze(0)], dim=1)
        logits, cache = self._forward(self.target_model, fed, cache)

        size = block.numel()
        predictions = logits[0, -(size + 1):].argmax(dim=-1)   # [size + 1]
        return predictions[:size], predictions[size], cache

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

        encoded = self.tokenizer(prompt, return_tensors="pt")
        context = encoded.input_ids.to(self.device)
        prompt_len = context.shape[1]

        eos_ids = set()
        if stop_at_eos and self.tokenizer.eos_token_id is not None:
            eos = self.tokenizer.eos_token_id
            eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}

        cache = None
        generated: list[int] = []
        printed_chars = 0
        stats = GenerationStats()
        start = time.perf_counter()

        while len(generated) < max_new_tokens:
            size = min(self.block_size, max_new_tokens - len(generated))
            block, source = self.initial_block(context, size)
            if source == "ngram":
                stats.ngram_drafts_used += 1
            else:
                stats.model_drafts_used += 1

            context_len = context.shape[1]
            iterations = 0
            committed = 0
            bonus: torch.Tensor | None = None
            converged = False

            for _ in range(self.max_iterations):
                # Each sweep re-scores the block from scratch, so the block's own KV
                # entries from the previous sweep are stale. Cropping back to the
                # committed context is the same rollback the speculative engines use.
                cache = _crop_cache(cache, context_len - 1)
                new_block, bonus, cache = self._relax(context, block, cache)
                iterations += 1

                overlap = longest_common_prefix(new_block, block)
                # Both bounds are sound; the iteration count is what guarantees
                # progress when the guess was wrong at position 1.
                committed = max(overlap, iterations)
                block = new_block
                if overlap == size:
                    converged = True
                    break
                if committed >= size:
                    break

            committed = min(committed, size)
            accepted = block[:committed]
            # A true fixed point also validates the bonus prediction: it is the greedy
            # token following a block now known to be the greedy continuation.
            if converged and bonus is not None and committed == size:
                accepted = torch.cat([accepted, bonus.view(1)])

            stats.iterations += 1
            stats.target_forwards += iterations
            stats.draft_tokens_proposed += size
            stats.draft_tokens_accepted += committed
            stats.accepted_per_iteration.append(committed)
            stats.draft_lengths.append(size)
            stats.jacobi_iterations.append(iterations)
            if converged:
                stats.jacobi_fixed_points += 1

            new_ids = accepted.tolist()[: max_new_tokens - len(generated)]

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
            elif not hit_eos:
                # Cannot happen: committed >= 1 whenever a sweep ran. Guarded because
                # a silent stall here would be an infinite loop rather than a wrong
                # answer, and that is the worse failure.
                raise RuntimeError(
                    "Jacobi block committed no tokens; the iteration-count floor "
                    "should make this unreachable"
                )

            if hit_eos:
                break

            cache = _crop_cache(cache, context.shape[1] - 1)

        stats.seconds = time.perf_counter() - start
        stats.token_ids = list(generated)
        if stream:
            print(flush=True)

        assert context.shape[1] == prompt_len + len(generated), (
            f"context ({context.shape[1]}) and generated ({len(generated)}) "
            f"disagree for a {prompt_len}-token prompt"
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True), stats
