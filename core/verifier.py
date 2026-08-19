"""Modified rejection sampling for speculative decoding.

Implements Algorithm 1 ("SpeculativeDecodingStep") from

    Leviathan, Kalman & Matias, *Fast Inference from Transformers via
    Speculative Decoding*, ICML 2023.  https://arxiv.org/abs/2211.17192

This module is deliberately model-free: it knows nothing about transformers,
tokenizers or caches.  It operates purely on probability distributions and
token ids, which makes the correctness guarantee of the paper -- that the
emitted tokens are distributed *exactly* according to the target model --
directly testable.

Notation follows the paper: ``q`` is the draft (small) model, ``p`` is the
target (large) model.
"""

from __future__ import annotations

import torch

# Guards a division by a zero draft probability.  A token can only be proposed
# if q(x) > 0, so this branch is unreachable for well-formed input; it exists so
# that adversarial input yields a finite acceptance probability instead of NaN.
_EPS = 1e-12


def verify_tokens(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, int]:
    """Verify a block of drafted tokens against the target distribution.

    Args:
        draft_probs: ``[gamma, V]`` draft distributions ``q_i``, where row ``i``
            is the distribution the draft model used to propose
            ``draft_tokens[i]``.
        target_probs: ``[gamma, V]`` or ``[gamma + 1, V]`` target distributions
            ``p_i`` evaluated on the same prefixes.  Supplying the extra
            ``gamma + 1``-th row enables the paper's "bonus token": when every
            drafted token is accepted, one free token is sampled from the
            target distribution that follows the whole block.
        draft_tokens: ``[gamma]`` integer token ids proposed by the draft model.
        generator: optional :class:`torch.Generator` for reproducible sampling.

    Returns:
        ``(tokens, n_accepted)`` where

        * ``tokens`` is the 1-D tensor of tokens to actually emit -- the
          accepted prefix of ``draft_tokens``, followed by either the residual
          resample (on rejection) or the bonus token (if all were accepted and
          a ``gamma + 1``-th target row was supplied).
        * ``n_accepted`` is the number of drafted tokens that survived
          verification, in ``[0, gamma]``.

    The emitted token at every position is distributed exactly as the target
    model would have sampled it; ``n_accepted`` only affects throughput, never
    the output distribution.
    """
    if draft_probs.dim() != 2 or target_probs.dim() != 2:
        raise ValueError(
            "draft_probs and target_probs must be 2-D [sequence, vocab]; got "
            f"{tuple(draft_probs.shape)} and {tuple(target_probs.shape)}"
        )
    if draft_tokens.dim() != 1:
        raise ValueError(
            f"draft_tokens must be 1-D [sequence]; got {tuple(draft_tokens.shape)}"
        )

    gamma = draft_tokens.shape[0]
    if gamma == 0:
        raise ValueError("draft_tokens must contain at least one token")
    if draft_probs.shape[0] != gamma:
        raise ValueError(
            f"draft_probs has {draft_probs.shape[0]} rows but there are {gamma} draft tokens"
        )
    if target_probs.shape[0] not in (gamma, gamma + 1):
        raise ValueError(
            f"target_probs must have {gamma} or {gamma + 1} rows; got {target_probs.shape[0]}"
        )
    if draft_probs.shape[1] != target_probs.shape[1]:
        raise ValueError(
            f"vocab mismatch: draft has {draft_probs.shape[1]}, target has {target_probs.shape[1]}"
        )

    tokens = draft_tokens.reshape(-1).long()
    has_bonus_row = target_probs.shape[0] == gamma + 1

    # --- 1. Gather q_i(x_i) and p_i(x_i) for the proposed tokens ------------
    index = tokens.unsqueeze(1)                              # [gamma, 1]
    q = draft_probs.gather(1, index).squeeze(1)               # [gamma]
    p = target_probs[:gamma].gather(1, index).squeeze(1)      # [gamma]

    # --- 2. Accept token i with probability min(1, p_i(x_i) / q_i(x_i)) -----
    # All gamma uniforms are drawn at once.  Only the draws up to the first
    # rejection are consumed, and each draw is independent of the others, so
    # this is distributionally identical to drawing them one at a time.
    accept_prob = (p / q.clamp_min(_EPS)).clamp(max=1.0)
    u = torch.rand(gamma, generator=generator, dtype=accept_prob.dtype, device=accept_prob.device)
    accepted = u < accept_prob

    # --- 3. Truncate at the first rejection --------------------------------
    rejected = ~accepted
    if bool(rejected.any()):
        n_accepted = int(rejected.long().argmax())
    else:
        n_accepted = gamma

    emitted = [tokens[:n_accepted]]

    if n_accepted < gamma:
        # Rejection: resample from the residual  norm(max(0, p - q)).
        residual = (target_probs[n_accepted] - draft_probs[n_accepted]).clamp_min(0.0)
        total = residual.sum()
        if not bool(total > _EPS):
            # p == q on this row, so rejection has probability zero and this is
            # pure floating-point noise; falling back to p keeps the guarantee.
            residual = target_probs[n_accepted]
        emitted.append(torch.multinomial(residual, num_samples=1, generator=generator))
    elif has_bonus_row:
        # Every draft token accepted: take the free token from p_{gamma+1}.
        emitted.append(
            torch.multinomial(target_probs[gamma], num_samples=1, generator=generator)
        )

    return torch.cat(emitted), n_accepted
