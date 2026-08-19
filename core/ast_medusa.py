"""AST-guided Medusa: M lightweight heads predicting only structural tokens.

Medusa bolts extra prediction heads onto a frozen base model. Head ``i`` predicts
the token ``i`` positions ahead directly from the *same* final hidden state, so one
base forward yields M proposals at once instead of one token. Those proposals are
then verified by the next base forward, which is what keeps the output distribution
honest.

The AST restriction here narrows every head to *structural* tokens -- keywords,
punctuation, indentation -- on the theory that code boilerplate is the predictable
part. That restriction turns out to matter for a reason beyond accuracy, which is
the main architectural finding in this module.

Why the head projection must be factored
----------------------------------------
A full-vocabulary head for this model is ``hidden_size x vocab_size`` =
3584 x 152064 = **545M parameters**. Fifteen of them is 8.17B parameters, 15.2 GiB in
fp16, on top of an 8.1 GiB 8-bit base: **23.3 GiB, which does not fit on a 16 GB
card**. The architecture as usually written cannot be instantiated here at M=15.

But only 5,954 of 151,665 tokens are structural (3.9%). A head restricted to those
needs to project to 5,954 outputs, not 152,064 -- 21.3M parameters, **26x smaller**,
610 MiB for all fifteen. Masking 96% of an output to negative infinity and then
computing it anyway is pure waste; projecting only to the retained columns and
scattering them into a ``-inf`` background is *mathematically identical* and is what
makes M=15 feasible. ``head_mode="full"`` implements the literal masked version for
comparison, and :func:`assert_modes_equivalent` checks the two agree.

What the theoretical ceiling actually is
---------------------------------------
One base forward emitting 1 + M tokens looks like an ``M + 1``x speedup, and the raw
latency ratio will report roughly that. It is a real measurement of the hardware, and
it is also an upper bound that assumes something the AST restriction forbids: that
every one of the next M tokens is structural. Heads that can only emit ``def``, ``:``
and indentation cannot propose an identifier, so a Medusa step accepts only as far as
the *consecutive run* of structural tokens extends. ``benchmark_medusa_limit.py``
measures that run-length distribution on real generated code, which is the ceiling
that governs a trained system.
"""

from __future__ import annotations

import keyword
import string

import torch
from torch import nn

# Tokens treated as structural. Keywords and soft keywords carry program shape;
# `self`, `None`, `True`, `False` are included because they are fixed identifiers
# rather than user-chosen names. Everything whitespace-only counts, since
# indentation *is* Python's block structure.
_KEYWORDS = frozenset(
    set(keyword.kwlist)
    | set(getattr(keyword, "softkwlist", []))
    | {"self", "cls", "None", "True", "False"}
)
_PUNCTUATION = frozenset(string.punctuation)


def is_structural(text: str) -> bool:
    """Whether a decoded token counts as program structure rather than content.

    Three families: whitespace/indentation, keywords, and pure punctuation or
    operator runs. Deliberately excludes identifiers, numbers and string bodies --
    those are the content a head cannot be expected to guess.
    """
    stripped = text.strip()
    if stripped == "":
        return True                                   # indentation and newlines
    if stripped in _KEYWORDS:
        return True
    return all(char in _PUNCTUATION for char in stripped)


def structural_token_ids(tokenizer, vocab_size: int | None = None) -> torch.Tensor:
    """Sorted ids of every structural token in the tokenizer's vocabulary.

    ``vocab_size`` clips to the model's output width, which for Qwen is wider than
    the tokenizer (152064 vs 151665) because the embedding matrix is padded.
    """
    ids = []
    for token, index in tokenizer.get_vocab().items():
        if vocab_size is not None and index >= vocab_size:
            continue
        if is_structural(tokenizer.convert_tokens_to_string([token])):
            ids.append(index)
    return torch.tensor(sorted(ids), dtype=torch.long)


class MedusaAST(nn.Module):
    """A frozen base model with M AST-restricted lookahead heads.

    Args:
        base_model: the target model. Every parameter is frozen; only the heads
            would ever train.
        tokenizer: used to determine which token ids are structural.
        num_heads: M. Head ``i`` predicts the token ``i + 1`` positions ahead.
        head_mode: ``"factored"`` projects to the structural subset only (default,
            and the only mode that fits at M=15); ``"full"`` projects to the whole
            vocabulary and masks, matching the usual formulation.
        dtype: head parameter dtype. Defaults to the base model's hidden dtype.
    """

    def __init__(
        self,
        base_model,
        tokenizer,
        num_heads: int = 15,
        head_mode: str = "factored",
        head_rank: int | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if head_mode not in ("factored", "full"):
            raise ValueError(f"head_mode must be 'factored' or 'full', got {head_mode!r}")
        if head_rank is not None and head_rank < 1:
            raise ValueError(f"head_rank must be >= 1 or None, got {head_rank}")

        self.base_model = base_model
        self.tokenizer = tokenizer
        self.num_heads = num_heads
        self.head_mode = head_mode
        self.head_rank = head_rank

        config = base_model.config
        self.hidden_size = int(config.hidden_size)
        self.vocab_size = int(config.vocab_size)

        # Freeze the base. Medusa's premise is that the expensive model is a fixed
        # asset and only the heads are trained, so this is load-bearing rather than
        # hygiene: an unfrozen base would make the heads' job trivial and the
        # measurement meaningless.
        frozen = 0
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
            frozen += parameter.numel()
        self.frozen_parameters = frozen

        device = next(base_model.parameters()).device
        head_dtype = dtype or torch.float16

        # Register the id buffer already on the base model's device. Leaving it on
        # CPU while the heads live on the GPU is silently fine until the scatter in
        # head_logits, and CPU-only tests cannot catch it because there the two
        # devices coincide.
        structural = structural_token_ids(tokenizer, self.vocab_size).to(device)
        self.register_buffer("structural_ids", structural, persistent=False)

        out_features = (
            structural.numel() if head_mode == "factored" else self.vocab_size
        )
        # A full linear head is hidden x out_features parameters, which at
        # out_features ~ 11k is 40M per head -- far more than a few hundred thousand
        # training examples can determine. `head_rank` inserts a bottleneck,
        # trading capacity for data efficiency; None keeps the plain linear map.
        def build_head() -> nn.Module:
            if head_rank is None:
                return nn.Linear(self.hidden_size, out_features, bias=False)
            return nn.Sequential(
                nn.Linear(self.hidden_size, head_rank, bias=False),
                nn.Linear(head_rank, out_features, bias=False),
            )

        self.medusa_heads = nn.ModuleList(
            build_head() for _ in range(num_heads)
        ).to(device=device, dtype=head_dtype)

        # Everything the forward path touches must agree on device, checked at
        # construction so a mismatch fails loudly here rather than deep in a matmul.
        head_device = self.head_weight(0).device
        if head_device != self.structural_ids.device:
            raise RuntimeError(
                f"device mismatch: heads on {head_device}, structural ids on "
                f"{self.structural_ids.device}"
            )

        if head_mode == "full":
            # Precompute the -inf mask once. Building it per forward would dominate
            # the very latency this module exists to measure.
            allowed = torch.zeros(self.vocab_size, dtype=torch.bool)
            allowed[structural] = True
            self.register_buffer(
                "blocked_mask", (~allowed).to(device), persistent=False
            )

    # ------------------------------------------------------------------ info
    def head_weight(self, index: int) -> torch.Tensor:
        """A representative weight tensor for head ``index``, whatever its shape."""
        head = self.medusa_heads[index]
        return head.weight if isinstance(head, nn.Linear) else head[0].weight

    @property
    def n_structural(self) -> int:
        return int(self.structural_ids.numel())

    @property
    def head_parameters(self) -> int:
        return sum(p.numel() for p in self.medusa_heads.parameters())

    def parameter_report(self) -> str:
        full_equivalent = self.hidden_size * self.vocab_size * self.num_heads
        return (
            f"base frozen: {self.frozen_parameters / 1e9:.2f}B | "
            f"heads ({self.head_mode}"
            + (f", rank {self.head_rank}" if self.head_rank else ", full")
            + f", M={self.num_heads}): "
            f"{self.head_parameters / 1e6:.1f}M "
            f"({self.head_parameters * 2 / 2**20:.0f} MiB fp16) | "
            f"full-vocab equivalent would be {full_equivalent / 1e9:.2f}B "
            f"({full_equivalent * 2 / 2**30:.1f} GiB fp16)"
        )

    # --------------------------------------------------------------- forward
    def head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run all M heads on a hidden state, returning full-vocab logits.

        Args:
            hidden: ``[..., hidden_size]`` final hidden state.

        Returns:
            ``[M, ..., vocab_size]`` logits with every non-structural column at
            ``-inf``, so the AST restriction holds identically in both modes.
        """
        hidden = hidden.to(self.head_weight(0).dtype)

        if self.head_mode == "full":
            stacked = torch.stack([head(hidden) for head in self.medusa_heads])
            return stacked.masked_fill(self.blocked_mask, float("-inf"))

        # Factored: project to the structural subset, then scatter into a -inf
        # background. Same distribution as the masked full projection, 26x cheaper.
        compact = torch.stack([head(hidden) for head in self.medusa_heads])
        shape = compact.shape[:-1] + (self.vocab_size,)
        full = torch.full(
            shape, float("-inf"), dtype=compact.dtype, device=compact.device
        )
        index = self.structural_ids.expand(shape[:-1] + (self.n_structural,))
        return full.scatter(-1, index, compact)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values=None, **kwargs):
        """One base forward plus all M heads.

        Returns ``(base_logits, head_logits, past_key_values)`` where
        ``base_logits`` is ``[B, T, V]`` from the base model's own head -- the token
        at ``t + 1`` -- and ``head_logits`` is ``[M, B, V]`` for ``t + 2 ... t + M + 1``
        read from the final position's hidden state.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            **kwargs,
        )
        last_hidden = outputs.hidden_states[-1][:, -1]        # [B, hidden]
        return outputs.logits, self.head_logits(last_hidden), outputs.past_key_values

    @torch.no_grad()
    def propose(self, input_ids: torch.Tensor, past_key_values=None):
        """Greedy proposal of 1 + M tokens from a single base forward.

        The first token comes from the base model's own head and is exact; the
        remaining M are the heads' guesses and would need verification before being
        committed. Returns ``(tokens [1 + M], past_key_values)``.
        """
        base_logits, heads, cache = self.forward(input_ids, past_key_values)
        first = base_logits[:, -1].argmax(dim=-1)             # [B]
        rest = heads.argmax(dim=-1)                           # [M, B]
        return torch.cat([first.unsqueeze(0), rest]).squeeze(-1), cache


def assert_modes_equivalent(
    base_model, tokenizer, num_heads: int = 2, seed: int = 0
) -> float:
    """Check the factored and full-vocab masked heads give identical distributions.

    The factored projection is the only thing that makes M=15 fit in VRAM, so its
    equivalence to the literal masked formulation is load-bearing rather than an
    optimisation detail. Returns the max absolute softmax difference.
    """
    torch.manual_seed(seed)
    factored = MedusaAST(base_model, tokenizer, num_heads, head_mode="factored")
    full = MedusaAST(base_model, tokenizer, num_heads, head_mode="full")

    # Copy the factored weights into the corresponding columns of the full heads so
    # the two parameterise the same function.
    with torch.no_grad():
        for compact_head, full_head in zip(factored.medusa_heads, full.medusa_heads):
            full_head.weight.zero_()
            full_head.weight[factored.structural_ids] = compact_head.weight

    weight = factored.head_weight(0)
    hidden = torch.randn(
        1, factored.hidden_size, device=weight.device, dtype=weight.dtype
    )
    a = torch.softmax(factored.head_logits(hidden).float(), dim=-1)
    b = torch.softmax(full.head_logits(hidden).float(), dim=-1)
    return float((a - b).abs().max())
