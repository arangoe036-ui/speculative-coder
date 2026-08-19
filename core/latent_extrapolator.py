"""EAGLE-style latent extrapolation: draft in feature space, not token space.

Medusa failed here for a specific reason. Every head projected from the *same* hidden
state `h_t`, so head 2 had to predict token t+2 without ever learning what t+1 turned
out to be. Measured per-head accuracy collapsed 65.5% -> 27.7% -> 10.9%, and since a
step commits the longest all-correct prefix, the chain product killed it by head 3.

This attacks exactly that deficit. Instead of M independent projections, a single
lightweight transformer layer *extrapolates the hidden state itself*:

    h_{t+1} = Extrapolator( concat( emb(y_{t+1}), h_t ) )

and the base model's own frozen ``lm_head`` turns that predicted state into logits for
token t+2. Run it autoregressively and each step conditions on the previous step's
prediction -- in latent space rather than by re-running the base model. That is the
whole idea: recover autoregressive feedback at a fraction of the cost.

The accounting, stated carefully
--------------------------------
One base forward yields ``h_t``, and ``lm_head(h_t)`` gives token t+1 **exactly** --
it is the base model's own prediction, not a guess. So the free first token is not
attributable to the head at all.

This matters because it is easy to double-count. A Medusa head trained to predict
``g_{t+i-1}`` produces token t+i, so its *first* head duplicates what ``lm_head``
already computes from the same vector, at strictly worse accuracy. Counting both the
base token and head 1's token credits position t+1 twice. Expected accepted length is

    1.0  +  a_2  +  a_2 a_3  +  ...

where ``a_i`` is the accuracy at position t+i and the leading 1.0 is the base's exact
token. :func:`chain_speedup` implements exactly that, and the same formula applies to
both architectures, which is what makes the comparison fair.

What the head can and cannot inherit
------------------------------------
``lm_head`` and the input embedding are the base model's, frozen and reused -- so the
head never learns a vocabulary projection, only the dynamics of the feature space.
That is why it can be lightweight: at hidden 3584 with a single layer it is ~115M
parameters against the 545M a single full-vocabulary Medusa head needs.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, matching the base model's normalisation style."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class ExtrapolatorHead(nn.Module):
    """One lightweight transformer decoder layer that advances a hidden state.

    Input is the concatenation of the next token's embedding and the current hidden
    state; output is the predicted next hidden state. Attention runs causally over the
    chain of latent states, so a K-step draft is a K-token sequence to this layer --
    the head sees its own history rather than a single vector, which is the structural
    difference from a Medusa projection.

    Args:
        hidden_size: the base model's hidden width.
        num_attention_heads: attention heads inside the layer.
        intermediate_size: MLP width. Defaults to ``hidden_size``, keeping the head
            light; the base model's own MLP is 5x wider.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 28,
        intermediate_size: int | None = None,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must divide by "
                f"num_attention_heads {num_attention_heads}"
            )
        intermediate_size = intermediate_size or hidden_size

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        # Fuse (embedding, hidden) down to one hidden-width vector. This is the only
        # place the doubled input width appears.
        self.fuse = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.input_norm = RMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.post_norm = RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        hidden: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """Advance ``hidden`` by one position per timestep.

        Args:
            token_embeddings: ``[B, T, H]`` embeddings of the tokens being consumed.
            hidden: ``[B, T, H]`` current hidden states.
            cache: optional ``(keys, values)`` from previous steps, for autoregressive
                drafting. Training passes ``None`` and relies on causal masking.

        Returns:
            ``(predicted_hidden [B, T, H], cache)``.
        """
        fused = self.fuse(torch.cat([token_embeddings, hidden], dim=-1))

        residual = fused
        x = self.input_norm(fused)
        query = self._split(self.q_proj(x))
        key = self._split(self.k_proj(x))
        value = self._split(self.v_proj(x))

        if cache is not None:
            key = torch.cat([cache[0], key], dim=2)
            value = torch.cat([cache[1], value], dim=2)
        new_cache = (key, value)

        # Causal only when scoring a whole sequence at once. With a cache the query is
        # a single new step that legitimately attends to everything stored.
        attended = torch.nn.functional.scaled_dot_product_attention(
            query, value=value, key=key, is_causal=cache is None and query.shape[2] > 1
        )
        attended = attended.transpose(1, 2).reshape_as(fused)
        x = residual + self.o_proj(attended)

        residual = x
        normed = self.post_norm(x)
        gated = torch.nn.functional.silu(self.gate_proj(normed)) * self.up_proj(normed)
        return residual + self.down_proj(gated), new_cache


class LatentExtrapolator(nn.Module):
    """A frozen base model plus one latent extrapolation head.

    The base contributes three frozen pieces the head reuses rather than relearns:
    the input embedding, the final hidden state, and ``lm_head``. Only the
    extrapolator trains.
    """

    def __init__(
        self,
        base_model,
        num_attention_heads: int = 28,
        intermediate_size: int | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.base_model = base_model
        config = base_model.config
        self.hidden_size = int(config.hidden_size)
        self.vocab_size = int(config.vocab_size)

        frozen = 0
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
            frozen += parameter.numel()
        self.frozen_parameters = frozen

        device = next(base_model.parameters()).device
        self.head = ExtrapolatorHead(
            self.hidden_size, num_attention_heads, intermediate_size
        ).to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ info
    @property
    def head_parameters(self) -> int:
        return sum(p.numel() for p in self.head.parameters())

    def parameter_report(self) -> str:
        medusa_equivalent = self.hidden_size * self.vocab_size
        return (
            f"base frozen: {self.frozen_parameters / 1e9:.2f}B | "
            f"extrapolator: {self.head_parameters / 1e6:.1f}M "
            f"({self.head_parameters * 4 / 2**20:.0f} MiB fp32) | "
            f"one full-vocab Medusa head would be "
            f"{medusa_equivalent / 1e6:.0f}M"
        )

    # --------------------------------------------------------------- pieces
    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.base_model.get_input_embeddings()(token_ids)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Frozen ``lm_head``: hidden state to vocabulary logits."""
        head = self.base_model.get_output_embeddings()
        return head(hidden.to(head.weight.dtype))

    # --------------------------------------------------------------- drafting
    @torch.no_grad()
    def draft(self, hidden: torch.Tensor, steps: int) -> torch.Tensor:
        """Propose ``1 + steps`` tokens from a single base hidden state.

        The first token is ``argmax lm_head(h_t)`` -- the base model's own exact
        prediction, free. Each subsequent token comes from an extrapolated state, fed
        the token the previous step predicted. Errors therefore compound, which is the
        honest drafting condition and how it must be evaluated.

        Args:
            hidden: ``[B, H]`` final hidden state from a base forward.
            steps: how many extrapolation steps to run.

        Returns:
            ``[B, 1 + steps]`` proposed token ids.
        """
        state = hidden.unsqueeze(1)                                # [B, 1, H]
        token = self.project(state).argmax(dim=-1)                 # [B, 1] exact
        proposals = [token]
        cache = None

        for _ in range(steps):
            embedding = self.embed(token).to(state.dtype)
            state, cache = self.head(embedding, state, cache)
            token = self.project(state).argmax(dim=-1)
            proposals.append(token)

        return torch.cat(proposals, dim=1)

    def forward(self, token_embeddings: torch.Tensor, hidden: torch.Tensor):
        """Teacher-forced training pass over a whole sequence."""
        return self.head(token_embeddings, hidden)[0]


def chain_speedup(step_accuracies: list[float]) -> dict:
    """Tokens per base forward, given accuracies for positions t+2 onward.

    Position t+1 is exact -- it is ``argmax lm_head(h_t)``, the base model's own
    output -- so it contributes a fixed 1.0 and is deliberately *not* taken from
    ``step_accuracies``. Counting a head's guess at t+1 as well would credit that
    position twice.

    A step commits the longest all-correct prefix, so the remaining positions enter as
    a running product.
    """
    accepted = 1.0
    running = 1.0
    chain = []
    for accuracy in step_accuracies:
        running *= accuracy
        accepted += running
        chain.append(running)
    return {
        "tokens_per_forward": accepted,
        "speedup": accepted,
        "chain": chain,
    }
