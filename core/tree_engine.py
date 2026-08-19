"""Evolutionary tree speculation: branch the draft only where it is uncertain.

Monte Carlo speculation pays for B full-width draft branches from the first token,
including at the many positions where the draft model is completely sure what comes
next. Widening every step to explore a decision that only matters at step 5 is
wasted breadth.

This engine grows the tree instead. Drafting starts with a single branch taking
argmax tokens. At each step it inspects each branch's top-1 probability, and where
that falls below ``split_threshold`` -- i.e. where the draft is genuinely unsure --
it clones the branch: one child takes the top-1 token, the other takes the top-2.
Certainty costs one row; uncertainty costs two. The trunk is drafted once and
diversity is spent exactly where the draft might be wrong.

What this actually saves, and what it does not
----------------------------------------------
It saves draft *batch rows*. Monte Carlo B=8, K=8 pushes 64 rows through the draft
model per iteration regardless of context; a tree that splits twice pushes closer to
20. That is a genuine reduction in draft FLOPs.

Whether it saves *time* is a separate question, and the honest answer is: much less
than the FLOP reduction suggests. `batch_scaling.py` measured a batch-8 forward at
1.06x the cost of batch-1 on this hardware, because decoding is bound by reading
weights rather than by arithmetic. Narrowing a batch reclaims compute that was
nearly free. The tree's real hope is therefore *acceptance*, not compute: branching
at the exact points of uncertainty should be a better use of breadth than eight
random samples, because those are the positions where the draft actually errs.

All leaves are the same length
------------------------------
Every branch appends exactly one token per step, and a split duplicates its parent's
history, so after k steps every leaf holds exactly k tokens. No padding and no tree
attention mask are required -- each leaf is materialised as its own batch row with
its own cloned KV cache. That trades memory (B copies of the trunk) for
implementation simplicity; a shared-trunk tree mask would save that memory but not
the time, for the reason above.

Losslessness
------------
Same argument as the other breadth engines, and it is what makes this aggressive
branching safe. The draft here is *deterministic* (top-1 or top-2, no sampling), so
``q`` is not the density anything was sampled from -- but at target temperature 0
that cannot matter. ``p`` is one-hot at the target's argmax ``a``: a proposal is
accepted only if it equals ``a``, and a rejected proposal is replaced by the residual
``max(0, p - q)``, which is also one-hot at ``a``. Every leaf therefore emits a
prefix of the target's greedy continuation, and taking the argmax over leaves picks
the longest prefix of one string. Above temperature 0 the selection would bias
generation, so the constructor refuses it.
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
from core.particle_filter_engine import _reorder_cache
from core.verifier import verify_tokens


class TreeSpeculativeEngine(SpeculativeEngine):
    """Speculative decoding with a dynamically branched draft tree.

    Args:
        k: draft depth (tree height).
        max_leaves: ceiling on concurrent branches.
        split_threshold: split a branch when its top-1 probability falls below
            this. Higher splits more often; ``1.0`` would split at every step and
            reduce to a breadth-first beam, ``0.0`` never splits and reduces to a
            single greedy draft.
        temperature: target temperature. Must be 0; see the module docstring.
    """

    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        k: int = 8,
        max_leaves: int = 8,
        split_threshold: float = 0.80,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        if temperature != 0.0:
            raise ValueError(
                "TreeSpeculativeEngine requires temperature=0. Selecting the "
                "longest-accepted leaf conditions on the acceptance outcome, which "
                "is only distribution-preserving when the target is greedy (got "
                f"temperature={temperature}). See the module docstring."
            )
        if max_leaves < 1:
            raise ValueError(f"max_leaves must be >= 1, got {max_leaves}")
        if not 0.0 <= split_threshold <= 1.0:
            raise ValueError(
                f"split_threshold is a probability in [0, 1], got {split_threshold}"
            )
        super().__init__(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            k=k,
            temperature=0.0,
            top_p=top_p,
        )
        self.max_leaves = max_leaves
        self.split_threshold = split_threshold

    # ---------------------------------------------------------------- phases
    def _draft_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """The draft model's beliefs, unwarped.

        Read at temperature 1 rather than through ``_logits_to_probs``: the target's
        temperature of 0 would make every top-1 probability exactly 1.0 and no
        branch would ever be judged uncertain, silently disabling all splitting.
        """
        sliced = logits[..., : self.vocab_size].to(self._verify_dtype)
        return torch.softmax(sliced, dim=-1)

    @torch.no_grad()
    def _draft_tree(self, context: torch.Tensor, cache, k: int):
        """Grow a draft tree of up to ``max_leaves`` leaves, each ``k`` tokens deep.

        Returns ``(tokens [L, k], q [L, k, V], cache, calls, rows, splits)`` where
        ``L`` is the final leaf count, ``calls`` counts forward launches and ``rows``
        counts batch rows pushed through the draft model.
        """
        pending = context[:, _cache_length(cache):]
        tokens = torch.empty((1, 0), dtype=torch.long, device=context.device)
        q_rows: torch.Tensor | None = None
        calls = 0
        rows = 0
        splits = 0

        for _ in range(k):
            active = pending.shape[0]
            logits, cache = self._forward(self.draft_model, pending, cache)
            calls += 1
            rows += active

            probs = self._draft_probs(logits[:, -1])            # [active, V]
            top = torch.topk(probs, 2, dim=-1)
            top1_prob = top.values[:, 0].tolist()
            top1_id = top.indices[:, 0]
            top2_id = top.indices[:, 1]

            # Decide which branches to split. Taken in order, stopping at the leaf
            # ceiling, so the earliest uncertain branch gets first claim on budget.
            budget = self.max_leaves - active
            parents: list[int] = []
            chosen: list[torch.Tensor] = []
            for b in range(active):
                parents.append(b)
                chosen.append(top1_id[b])
                if budget > 0 and top1_prob[b] < self.split_threshold:
                    parents.append(b)          # clone the same parent...
                    chosen.append(top2_id[b])  # ...but take its runner-up
                    budget -= 1
                    splits += 1

            index = torch.tensor(parents, device=context.device, dtype=torch.long)
            next_tokens = torch.stack(chosen).view(-1, 1)        # [new_active, 1]

            # Duplicate each split parent's history, cache and per-step
            # distribution. Both children were proposed from the same parent state,
            # so they legitimately share that step's q row.
            tokens = torch.cat([tokens[index], next_tokens], dim=1)
            step_q = probs[index].unsqueeze(1)                   # [new_active, 1, V]
            q_rows = step_q if q_rows is None else torch.cat(
                [q_rows[index], step_q], dim=1
            )
            if index.numel() != active or splits:
                cache = _reorder_cache(cache, index)
            pending = next_tokens

        return tokens, q_rows, cache, calls, rows, splits

    @torch.no_grad()
    def _verify_leaves(self, context: torch.Tensor, leaves: torch.Tensor, cache):
        """Score every leaf against the target in one batched forward."""
        n_leaves, k = leaves.shape
        cache = _expand_cache(cache, n_leaves)
        pending = context[:, _cache_length(cache):].repeat(n_leaves, 1)
        fed = torch.cat([pending, leaves], dim=1)

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
                "the dual gate is not wired into the tree loop; its early exit and "
                "the split decision would be competing for the same signal"
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

            leaves, q, draft_cache, calls, rows, splits = self._draft_tree(
                context, draft_cache, k
            )

            # Distinct leaves only. Two branches can reconverge on the same tokens
            # after a split, and scoring an identical sequence twice buys nothing.
            unique, inverse = torch.unique(leaves, dim=0, return_inverse=True)
            representative = torch.zeros(
                unique.shape[0], dtype=torch.long, device=leaves.device
            )
            representative.scatter_(
                0, inverse, torch.arange(leaves.shape[0], device=leaves.device)
            )
            unique_q = q[representative]

            p, target_cache = self._verify_leaves(context, unique, target_cache)

            outcomes = [
                verify_tokens(unique_q[i], p[i], unique[i])
                for i in range(unique.shape[0])
            ]
            accepted_counts = [n for _, n in outcomes]
            winner = max(range(len(accepted_counts)), key=accepted_counts.__getitem__)
            accepted_tokens, n_accepted = outcomes[winner]

            stats.iterations += 1
            stats.target_forwards += 1
            stats.draft_forwards += calls
            stats.draft_row_forwards += rows
            stats.draft_tokens_proposed += k
            stats.draft_tokens_accepted += n_accepted
            stats.accepted_per_iteration.append(n_accepted)
            stats.draft_lengths.append(k)
            stats.branch_accepted.append(list(accepted_counts))
            stats.branch_wins.append(winner)
            stats.tree_splits += splits
            stats.leaf_counts.append(int(leaves.shape[0]))
            stats.unique_candidates.append(int(unique.shape[0]))

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
