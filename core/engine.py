"""Two-model speculative decoding generation loop.

Wraps the Phase 1 verifier (:func:`core.verifier.verify_tokens`) in the actual
draft/verify loop of Leviathan et al. (2023):

    1. the small draft model autoregressively proposes K tokens (K forwards),
    2. the large target model scores all K in ONE forward pass,
    3. modified rejection sampling decides how many survive,
    4. the KV caches of both models are rolled back to the accepted prefix.

The win is step 2: one expensive forward yields up to K + 1 tokens instead of 1.
Correctness does not depend on the draft model being good -- only throughput does.

This class does no model loading and no device placement: pass in models that are
already loaded, in eval mode, on the device you want.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from core.verifier import verify_tokens


@dataclass
class GenerationStats:
    """Throughput bookkeeping for one :meth:`SpeculativeEngine.generate` call."""

    tokens_generated: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    iterations: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    seconds: float = 0.0
    accepted_per_iteration: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list, repr=False)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed draft tokens that survived verification (alpha)."""
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.draft_tokens_accepted / self.draft_tokens_proposed

    @property
    def tokens_per_second(self) -> float:
        return self.tokens_generated / self.seconds if self.seconds > 0 else 0.0

    @property
    def speedup_vs_autoregressive(self) -> float:
        """Tokens emitted per target forward pass.

        Plain autoregressive decoding emits exactly 1.0 tokens per target
        forward, so this ratio is the theoretical wall-clock speedup in the limit
        where the target model dominates cost and the draft model is free.
        """
        if self.target_forwards == 0:
            return 0.0
        return self.tokens_generated / self.target_forwards

    def summary(self) -> str:
        return (
            f"{self.tokens_generated} tokens in {self.seconds:.2f}s "
            f"({self.tokens_per_second:.1f} tok/s) | "
            f"acceptance {self.acceptance_rate:.1%} "
            f"({self.draft_tokens_accepted}/{self.draft_tokens_proposed}) | "
            f"{self.iterations} iterations, {self.target_forwards} target forwards | "
            f"{self.speedup_vs_autoregressive:.2f} tokens per target forward"
        )


def _output_vocab_size(model) -> int:
    """Width of a model's next-token distribution (its lm_head output dim)."""
    head = model.get_output_embeddings()
    if head is not None:
        return int(head.weight.shape[0])
    return int(model.config.vocab_size)


# --------------------------------------------------------------------------
# KV-cache helpers
#
# transformers has used three cache representations over time.  These two
# helpers isolate that churn so the loop below stays readable.
# --------------------------------------------------------------------------
def _cache_length(cache) -> int:
    """Number of positions currently stored in a KV cache."""
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    # Legacy tuple-of-tuples: (layer, [key, value]) with key [B, H, S, D].
    return int(cache[0][0].shape[-2])


def _crop_cache(cache, max_length: int):
    """Truncate a KV cache to its first ``max_length`` positions.

    This is the rollback that makes speculative decoding cheap: rejected draft
    tokens are already in the cache, and dropping them is a slice rather than a
    recomputation of the prefix.
    """
    current = _cache_length(cache)
    if cache is None or max_length >= current:
        return cache
    if hasattr(cache, "crop"):
        # Pass the *negative delta* rather than the absolute length. Both are
        # accepted, but transformers deprecated the absolute form in 5.18, and
        # the relative form has meant "drop this many tokens" for far longer --
        # so this is the spelling that works across the widest version range.
        cache.crop(max_length - current)
        return cache
    # Legacy tuple cache: rebuild it with every layer sliced on the sequence axis.
    return tuple(
        tuple(tensor[:, :, :max_length, :] for tensor in layer) for layer in cache
    )


class SpeculativeEngine:
    """Speculative decoding over a (draft, target) model pair sharing a tokenizer.

    Args:
        draft_model: the small, fast model that proposes tokens.
        target_model: the large model that defines the output distribution.
        tokenizer: tokenizer shared by both models.  The two models must agree
            on token *ids*; differing embedding-matrix widths are reconciled
            automatically (see ``vocab_size`` below), but genuinely different
            tokenizers would produce silent garbage rather than an error.
        k: draft length (gamma in the paper).  Larger k means more tokens per
            target forward when the draft is good, and more wasted draft compute
            when it is not.
        temperature: sampling temperature, applied identically to both models.
            ``0.0`` means greedy.
        top_p: nucleus threshold, applied identically to both models.

    Attributes:
        vocab_size: the shared vocabulary width actually used for verification,
            ``min`` of the two models' output widths.
    """

    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        k: int = 5,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.k = k
        self.temperature = temperature
        self.top_p = top_p

        # Reconcile the two vocabularies.
        #
        # Model families routinely pad the embedding matrix to a hardware-friendly
        # width, and pad *different sibling checkpoints differently*: Qwen2.5-Coder
        # 1.5B has vocab_size 151936 while 7B has 152064, despite sharing a
        # tokenizer with ~151.6k real tokens. verify_tokens compares p and q
        # index by index, so mismatched widths would either crash on the tensor
        # shapes or, worse, silently compare different token spaces.
        #
        # Truncating both to the narrower width is safe because the surplus rows
        # are padding: no real token id reaches them, and they are untrained.
        # Slicing before the softmax means both p and q are proper, identically
        # supported distributions rather than two renormalisations of different sets.
        self.vocab_size = min(
            _output_vocab_size(draft_model), _output_vocab_size(target_model)
        )

        # Verification happens in fp32 regardless of model dtype.  The p/q ratio
        # is a division of two small numbers; in bf16 (3 decimal digits of
        # mantissa) that ratio carries visible error, which would bias the
        # acceptance test.  fp32 on a [K, V] tensor costs microseconds.
        self._verify_dtype = torch.float32

        self.device = next(target_model.parameters()).device

    # ---------------------------------------------------------------- utils
    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Map logits to the sampling distribution, in fp32.

        Applied identically to the draft and the target.  This matters: the
        paper's guarantee is that output ~ p, where p is *whatever distribution
        the target is sampled from*.  Warping q with temperature/top-p but not p
        (or vice versa) silently breaks the identity, so both go through here.
        """
        # Slice to the shared vocabulary first, so p and q live on the same
        # support before any normalisation happens.
        logits = logits[..., : self.vocab_size].to(self._verify_dtype)

        if self.temperature == 0:
            # Greedy: a one-hot p and q. verify_tokens still behaves correctly --
            # it accepts iff both models pick the same argmax, and the residual
            # max(0, p - q) collapses to the target's own choice on rejection.
            probs = torch.zeros_like(logits)
            probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
            return probs

        probs = torch.softmax(logits / self.temperature, dim=-1)

        if self.top_p < 1.0:
            ordered, indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = ordered.cumsum(dim=-1)
            # Keep the smallest prefix whose mass reaches top_p: shifting the
            # comparison keeps the token that crosses the threshold.
            drop = cumulative - ordered > self.top_p
            ordered = ordered.masked_fill(drop, 0.0)
            probs = torch.zeros_like(probs).scatter_(-1, indices, ordered)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs

    @staticmethod
    def _sample(probs: torch.Tensor) -> torch.Tensor:
        """Sample one token id from a 1-D distribution."""
        return torch.multinomial(probs, num_samples=1)

    def _forward(self, model, input_ids: torch.Tensor, cache):
        """One forward pass that appends ``input_ids`` to an existing KV cache.

        ``attention_mask`` and ``cache_position`` are passed explicitly rather
        than inferred.  After a rollback the cache is shorter than the tokens
        already emitted, and letting transformers guess the offset is the classic
        source of silent off-by-one corruption in speculative decoding.
        """
        cache_len = _cache_length(cache)
        total_len = cache_len + input_ids.shape[1]

        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            attention_mask=torch.ones(
                (input_ids.shape[0], total_len), dtype=torch.long, device=input_ids.device
            ),
            cache_position=torch.arange(cache_len, total_len, device=input_ids.device),
        )
        return outputs.logits, outputs.past_key_values

    # ---------------------------------------------------------------- phases
    @torch.no_grad()
    def _draft(self, context: torch.Tensor, cache, k: int):
        """Autoregressively propose ``k`` tokens with the draft model.

        Costs k sequential forwards -- this is the price speculative decoding
        pays, and why the draft model has to be small.

        Returns ``(draft_tokens [k], q [k, V], cache, n_forwards)``.
        """
        # Feed only what the cache has not seen yet.  On the first iteration that
        # is the whole prompt (prefill); afterwards it is the short tail left
        # over from the previous rollback.
        pending = context[:, _cache_length(cache):]

        tokens: list[torch.Tensor] = []
        q_rows: list[torch.Tensor] = []
        forwards = 0

        for _ in range(k):
            logits, cache = self._forward(self.draft_model, pending, cache)
            forwards += 1
            probs = self._logits_to_probs(logits[0, -1])   # [V]
            token = self._sample(probs)                    # [1]
            tokens.append(token)
            q_rows.append(probs)
            # The freshly sampled token is the only pending input next time.
            pending = token.unsqueeze(0)                   # [1, 1]

        # Note the final sampled token was never fed to the draft model, so the
        # draft cache ends one position short of the drafted block. The
        # "feed whatever is pending" rule above absorbs that automatically.
        return torch.cat(tokens), torch.stack(q_rows), cache, forwards

    @torch.no_grad()
    def _target_probs(self, context: torch.Tensor, draft_tokens: torch.Tensor, cache):
        """Score the whole drafted block with ONE target forward pass.

        Returns ``(p [k + 1, V], cache)``.  Row ``i < k`` is the target
        distribution for drafted position ``i``; row ``k`` is the distribution
        that follows the full block, i.e. the bonus token's distribution.
        """
        pending = context[:, _cache_length(cache):]
        fed = torch.cat([pending, draft_tokens.unsqueeze(0)], dim=1)

        logits, cache = self._forward(self.target_model, fed, cache)

        # The last k + 1 fed positions are (last context token, d_0 ... d_{k-1}).
        # A causal LM's logits at position j predict position j + 1, so those
        # rows are exactly the distributions for d_0 ... d_{k-1} plus the bonus.
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
        """Generate a continuation of ``prompt`` by speculative decoding.

        Returns ``(generated_text, stats)``.  ``generated_text`` excludes the
        prompt.  With ``stream=True`` the text is also printed incrementally as
        each block is verified.
        """
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

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
            context_len = context.shape[1]
            # No point drafting further than the caller asked for.
            k = min(self.k, max_new_tokens - len(generated))

            # (a) draft k tokens with the small model
            draft_tokens, q, draft_cache, n_draft_fwd = self._draft(
                context, draft_cache, k
            )

            # (b) one target forward over context + drafts -> p, including the
            #     k + 1-th row that grants the bonus token on full acceptance
            p, target_cache = self._target_probs(context, draft_tokens, target_cache)

            # (c) Phase 1 modified rejection sampling
            accepted_tokens, n_accepted = verify_tokens(q, p, draft_tokens)

            stats.iterations += 1
            stats.target_forwards += 1
            stats.draft_forwards += n_draft_fwd
            stats.draft_tokens_proposed += k
            stats.draft_tokens_accepted += n_accepted
            stats.accepted_per_iteration.append(n_accepted)

            # Trim to the caller's budget: full acceptance emits k + 1 tokens,
            # one more than we drafted for.
            new_ids = accepted_tokens.tolist()[: max_new_tokens - len(generated)]

            # Stop at EOS, keeping the EOS token itself out of the output.
            hit_eos = False
            for i, token_id in enumerate(new_ids):
                if token_id in eos_ids:
                    new_ids = new_ids[:i]
                    hit_eos = True
                    break

            generated.extend(new_ids)
            stats.tokens_generated = len(generated)

            # (e) stream. Decode the whole suffix and print only the delta: a
            # single BPE token can be half a UTF-8 character or half a word, so
            # decoding tokens individually mangles the output.
            if stream:
                text_so_far = self.tokenizer.decode(generated, skip_special_tokens=True)
                if len(text_so_far) > printed_chars:
                    print(text_so_far[printed_chars:], end="", flush=True)
                    printed_chars = len(text_so_far)

            # (d) commit the accepted tokens to the context.
            #
            # This happens before the EOS break, not after. Committing afterwards
            # leaves context one block behind `generated` on any run that ends
            # naturally, which is the common case at a realistic token budget.
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

            # KV rollback: drop every cached position past the committed context.
            #
            # One bound does all the work. The positions actually verified are
            # the old context plus the accepted drafts (context_len +
            # n_accepted), but the emitted block is at most n_accepted + 1 tokens
            # long, so the new context can never reach further than one past that
            # -- and its last token has been through no forward pass yet, since
            # it is the residual resample or the bonus token. So cropping to
            # "everything except the last committed token" is both necessary and
            # sufficient, and it stays correct if the block was cut short by the
            # token budget or by EOS.
            #
            # This is the rollback: rejected drafts sitting in the cache past
            # this point are dropped, and the next iteration re-feeds the tail.
            valid_len = context.shape[1] - 1
            draft_cache = _crop_cache(draft_cache, valid_len)
            target_cache = _crop_cache(target_cache, valid_len)

        stats.seconds = time.perf_counter() - start
        stats.token_ids = list(generated)
        if stream:
            print(flush=True)

        # Holds unconditionally now, including on EOS and budget termination.
        assert context.shape[1] == prompt_len + len(generated), (
            f"context ({context.shape[1]}) and generated ({len(generated)}) "
            f"disagree for a {prompt_len}-token prompt"
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True), stats
