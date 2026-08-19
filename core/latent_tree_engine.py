"""Latent Mitosis: a dynamically branched draft tree grown in feature space.

Combines the two things that worked best independently. The Evolutionary Tree earned
2.39x by spending breadth only where the drafter was uncertain; the EAGLE extrapolator
recovered autoregressive feedback without running a second model. This grows the tree
using the extrapolator instead of a draft model.

Each active branch holds a predicted hidden state and its own attention cache over the
latents it has produced. At every depth step the head advances all branches at once,
`lm_head` turns the new states into token logits, and any branch whose top-1
probability falls below ``split_threshold`` undergoes mitosis: one child takes the
top-1 token's embedding, its sibling takes the top-2, and both step forward from the
same latent with a cloned cache.

Two things the base model still does
------------------------------------
The first token of every block is ``argmax lm_head(h_t)`` -- the base model's own
output, exact and free, not a head prediction. And verification is a real forward of
the frozen 7B over the flattened leaves, so acceptance is decided by the target as
always.

That verification forward does double duty, which is what makes the loop cheap: it
returns both the acceptance probabilities *and* the true hidden states along the
winning leaf. The state at the last committed position seeds the next block's draft,
so no extra forward is needed to re-anchor the latent chain in reality.

Why this is expected to trail the 0.5B tree
-------------------------------------------
Worth stating up front, because the arithmetic is unfavourable before any code runs.
The 0.5B tree reaches 2.39x on a drafter whose per-step accuracy is high. This head's
free-running accuracy was measured at 32.1% for t+2 and 7.9% for t+3, so its chain is
already thin at depth 3. Branching lifts effective accuracy from top-1 toward top-2 at
split points, but 16 leaves across 8 steps affords roughly four splits against 2^8
possible top-2 paths.

Nor is the head as cheap as its parameter count suggests. Getting top-2 tokens
requires the full 152k-wide projection, so every step reads the frozen ``lm_head``
(545M parameters) on top of the head's own 115M -- around 660 MB against roughly 1 GB
for a 0.5B draft forward. Call it 1.5x cheaper per step, against a draft phase that
was only ~26% of the tree's runtime to begin with.
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
from core.latent_extrapolator import ExtrapolatorHead
from core.monte_carlo_engine import _expand_cache
from core.ngram import one_hot_q
from core.particle_filter_engine import _reorder_cache
from core.verifier import verify_tokens


def load_extrapolator_head(path: str, device: str = "cuda:0") -> ExtrapolatorHead:
    """Rebuild a trained :class:`ExtrapolatorHead` from a checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    head = ExtrapolatorHead(
        hidden_size=payload["hidden_size"],
        num_attention_heads=payload["num_attention_heads"],
        intermediate_size=payload["intermediate_size"],
    )
    head.load_state_dict(payload["state_dict"])
    return head.to(device=device, dtype=torch.float32).eval()


class LatentTreeEngine(SpeculativeEngine):
    """Speculative decoding with a latent draft tree grown by an extrapolator head.

    Args:
        target_model: the frozen base. Supplies the seed hidden state, the embedding,
            ``lm_head``, and verification. No second model is loaded.
        tokenizer: its tokenizer.
        head: a trained :class:`ExtrapolatorHead`.
        k: tree depth in extrapolation steps. A leaf carries ``1 + k`` tokens, the
            first being the base model's own exact prediction.
        max_leaves: ceiling on concurrent branches.
        split_threshold: split a branch when its top-1 probability falls below this.
        temperature: must be 0 -- selecting the longest-accepted leaf conditions on
            the acceptance outcome, which is only distribution-preserving for a greedy
            target. See :mod:`core.monte_carlo_engine` for the argument.
    """

    def __init__(
        self,
        target_model,
        tokenizer,
        head: ExtrapolatorHead,
        k: int = 8,
        max_leaves: int = 16,
        split_threshold: float = 0.85,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        if temperature != 0.0:
            raise ValueError(
                "LatentTreeEngine requires temperature=0. Committing the "
                "longest-accepted leaf conditions on the acceptance outcome, which "
                "is only distribution-preserving when the target is greedy (got "
                f"temperature={temperature})."
            )
        if max_leaves < 1:
            raise ValueError(f"max_leaves must be >= 1, got {max_leaves}")
        if not 0.0 <= split_threshold <= 1.0:
            raise ValueError(
                f"split_threshold is a probability in [0, 1], got {split_threshold}"
            )
        super().__init__(
            draft_model=target_model,
            target_model=target_model,
            tokenizer=tokenizer,
            k=k,
            temperature=0.0,
            top_p=top_p,
        )
        self.head = head
        self.max_leaves = max_leaves
        self.split_threshold = split_threshold

    # ---------------------------------------------------------------- pieces
    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.target_model.get_input_embeddings()(token_ids).to(torch.float32)

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self.target_model.get_output_embeddings()
        return head(hidden.to(head.weight.dtype))

    # ---------------------------------------------------------------- draft
    @torch.no_grad()
    def _grow_tree(self, anchor: torch.Tensor, consume: list[int], k: int):
        """Grow a latent tree, first advancing past any already-committed tokens.

        Args:
            anchor: ``[H]`` a *true* hidden state, from a position the target actually
                processed.
            consume: committed tokens sitting after the anchor that the target never
                saw -- the residual or bonus token from the previous block. They are
                fed through the head to advance the latent chain without proposing
                anything, since they are already decided.
            k: how many tokens to propose.

        Returns ``(tokens [L, k], q [L, k, V], splits, head_steps)``.

        Only a block whose ``consume`` is empty gets an exact first token from
        ``lm_head``; after a rejection the chain necessarily starts from a predicted
        state, because the corrected token was never run through the target.
        """
        state = anchor.view(1, 1, -1).to(torch.float32)
        cache = None
        head_steps = 0

        for token_id in consume:
            token = torch.tensor([[token_id]], device=state.device, dtype=torch.long)
            state, cache = self.head(self._embed(token), state, cache)
            head_steps += 1

        tokens: torch.Tensor | None = None
        q_rows: torch.Tensor | None = None
        splits = 0

        for step in range(k):
            active = state.shape[0]
            logits = self._project(state[:, 0]).float()            # [active, V]
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, 2, dim=-1)
            confidence = top.values[:, 0].tolist()

            budget = self.max_leaves - active
            parents: list[int] = []
            chosen: list[torch.Tensor] = []
            for b in range(active):
                parents.append(b)
                chosen.append(top.indices[b, 0])
                if budget > 0 and confidence[b] < self.split_threshold:
                    parents.append(b)                 # same latent...
                    chosen.append(top.indices[b, 1])  # ...different token
                    budget -= 1
                    splits += 1

            index = torch.tensor(parents, device=state.device, dtype=torch.long)
            picked = torch.stack(chosen).view(-1, 1)                # [new_active, 1]

            # Mitosis: clone each split parent's history, latent, cache and this
            # step's distribution. Siblings legitimately share the q they were
            # proposed from.
            tokens = picked if tokens is None else torch.cat(
                [tokens[index], picked], dim=1
            )
            step_q = probs[index].unsqueeze(1)
            q_rows = step_q if q_rows is None else torch.cat(
                [q_rows[index], step_q], dim=1
            )
            state = state[index]
            if cache is not None:
                cache = (cache[0][index], cache[1][index])

            if step < k - 1:
                state, cache = self.head(self._embed(picked), state, cache)
                head_steps += 1

        return tokens, q_rows, splits, head_steps

    # ---------------------------------------------------------------- verify
    @torch.no_grad()
    def _verify_leaves(self, context: torch.Tensor, leaves: torch.Tensor, cache):
        """Score every leaf with one batched target forward.

        Also returns the hidden states, because the state along the winning leaf is
        what seeds the next block -- extracting it here is what keeps the loop to a
        single base forward per iteration.
        """
        n_leaves, length = leaves.shape
        cache = _expand_cache(cache, n_leaves)
        pending = context[:, _cache_length(cache):].repeat(n_leaves, 1)
        fed = torch.cat([pending, leaves], dim=1)

        cache_len = _cache_length(cache)
        total = cache_len + fed.shape[1]
        outputs = self.target_model(
            input_ids=fed,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=torch.ones(
                (n_leaves, total), dtype=torch.long, device=fed.device
            ),
            cache_position=torch.arange(cache_len, total, device=fed.device),
        )
        probabilities = self._logits_to_probs(outputs.logits[:, -(length + 1):])
        return probabilities, outputs.hidden_states[-1], outputs.past_key_values, pending.shape[1]

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
        anchor_hidden: torch.Tensor | None = None
        pending_consume: list[int] = []
        generated: list[int] = []
        printed_chars = 0
        stats = GenerationStats()
        start = time.perf_counter()

        while len(generated) < max_new_tokens:
            budget = max_new_tokens - len(generated)
            depth = max(0, min(self.k, budget - 1))

            if anchor_hidden is None:
                # First block only: one forward to obtain a true hidden state. Every
                # later block takes its anchor from its own verification pass.
                pending = context[:, _cache_length(cache):]
                primer = self.target_model(
                    input_ids=pending, past_key_values=cache, use_cache=True,
                    output_hidden_states=True,
                )
                anchor_hidden = primer.hidden_states[-1][0, -1]
                pending_consume = []
                cache = primer.past_key_values
                stats.target_forwards += 1
                cache = _crop_cache(cache, context.shape[1] - 1)

            propose = max(1, min(self.k, budget))
            leaves, q, splits, head_steps = self._grow_tree(
                anchor_hidden, pending_consume, propose
            )
            length = leaves.shape[1]

            unique, inverse = torch.unique(leaves, dim=0, return_inverse=True)
            representative = torch.zeros(
                unique.shape[0], dtype=torch.long, device=leaves.device
            )
            representative.scatter_(
                0, inverse, torch.arange(leaves.shape[0], device=leaves.device)
            )

            probabilities, hidden_states, cache, fed_offset = self._verify_leaves(
                context, unique, cache
            )
            stats.target_forwards += 1

            outcomes = []
            for i in range(unique.shape[0]):
                # Deterministic proposals, so q is one-hot; exact at temperature 0.
                proposal = unique[i]
                outcomes.append(
                    verify_tokens(
                        one_hot_q(proposal, self.vocab_size, dtype=self._verify_dtype),
                        probabilities[i],
                        proposal,
                    )
                )
            accepted_counts = [n for _, n in outcomes]
            winner = max(range(len(accepted_counts)), key=accepted_counts.__getitem__)
            accepted_tokens, n_accepted = outcomes[winner]

            stats.iterations += 1
            stats.draft_tokens_proposed += length
            stats.draft_tokens_accepted += n_accepted
            stats.accepted_per_iteration.append(n_accepted)
            stats.draft_lengths.append(length)
            stats.branch_accepted.append(list(accepted_counts))
            stats.branch_wins.append(winner)
            stats.tree_splits += splits
            stats.leaf_counts.append(int(leaves.shape[0]))
            stats.unique_candidates.append(int(unique.shape[0]))
            stats.draft_forwards += head_steps
            stats.draft_row_forwards += int(leaves.shape[0]) * max(1, head_steps)

            new_ids = accepted_tokens.tolist()[:budget]

            hit_eos = False
            for i, token_id in enumerate(new_ids):
                if token_id in eos_ids:
                    new_ids = new_ids[:i]
                    hit_eos = True
                    break

            # Re-anchor for the next block. Only the *accepted leaf* tokens were
            # actually fed to the target, so only their hidden states exist. The
            # committed residual or bonus token was never seen by the target, so it
            # becomes something for the head to consume rather than an anchor.
            committed = len(new_ids)
            fed_committed = min(committed, n_accepted)
            if committed == 0:
                anchor_hidden, pending_consume = None, []
            else:
                anchor_index = fed_offset + fed_committed - 1
                anchor_hidden = hidden_states[winner, anchor_index]
                pending_consume = new_ids[fed_committed:]

            generated.extend(new_ids)
            stats.tokens_generated = len(generated)

            if stream:
                text_so_far = self.tokenizer.decode(generated, skip_special_tokens=True)
                if len(text_so_far) > printed_chars:
                    print(text_so_far[printed_chars:], end="", flush=True)
                    printed_chars = len(text_so_far)

            cache = _reorder_cache(cache, torch.tensor([winner], device=self.device))

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
