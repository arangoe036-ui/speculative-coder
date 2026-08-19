"""Prompt-lookup drafting: an n-gram matcher standing in for a draft model.

The cheapest possible draft model is no model at all. Generated text repeats
itself constantly -- identifiers, `return` patterns, an explanation restating the
code it just wrote -- so the recent past is often a usable prediction of the near
future. This looks up the last `min_match_len` tokens elsewhere in the context and
proposes whatever followed them last time.

Cost is a tensor comparison over the context: microseconds, no weights, no
device transfer, no gradient. Compared with a 0.5B draft forward at several
milliseconds, a hit is effectively free.

Why a deterministic draft is still exact
----------------------------------------
This proposes tokens with no probability model behind them, which looks like it
should break the verifier. It does not, and the reason is worth spelling out
because it is what makes the fast path legitimate rather than a fudge.

Treat the proposal as a one-hot ``q`` at the proposed token ``x``. Modified
rejection sampling then accepts with probability ``min(1, p(x)/1) = p(x)``, and on
rejection draws from ``norm(max(0, p - q))``, which puts zero mass on ``x`` and
``p(y)/(1 - p(x))`` on every other token. So

    P(emit x) = p(x)
    P(emit y) = (1 - p(x)) * p(y)/(1 - p(x)) = p(y)      for y != x

which is exactly ``p``. A deterministic draft is the degenerate-but-valid corner
of the same algorithm: confidence 1 in a wrong guess costs acceptance, never
correctness.
"""

from __future__ import annotations

import torch


def find_ngram_draft(
    context_tokens: torch.Tensor,
    max_draft_len: int = 5,
    min_match_len: int = 2,
) -> torch.Tensor | None:
    """Propose continuation tokens by matching the context's own recent history.

    Takes the final ``min_match_len`` tokens as a pattern, finds its most recent
    earlier occurrence, and returns up to ``max_draft_len`` tokens that followed
    it there.

    Args:
        context_tokens: 1-D or ``[1, L]`` tensor of token ids.
        max_draft_len: cap on how many tokens to propose.
        min_match_len: pattern length. Longer means rarer but better-founded
            matches; 2 fires often and is frequently wrong.

    Returns:
        A 1-D tensor of 1..``max_draft_len`` proposed token ids, or ``None`` when
        no usable match exists. Never returns an empty tensor -- ``verify_tokens``
        requires a non-empty block, so "nothing to propose" has to be ``None`` and
        the caller falls back to the model.

    The *most recent* match is used rather than the first. Recency is the better
    predictor, and it is also what makes the scan cheap to reason about: one pass,
    take the last hit.
    """
    if min_match_len < 1:
        raise ValueError(f"min_match_len must be >= 1, got {min_match_len}")
    if max_draft_len < 1:
        raise ValueError(f"max_draft_len must be >= 1, got {max_draft_len}")

    tokens = context_tokens.reshape(-1)
    length = tokens.shape[0]

    # Need the pattern itself plus at least one earlier position to match against
    # and one token following that match.
    if length < min_match_len + 2:
        return None

    pattern = tokens[-min_match_len:]

    # All sliding windows of pattern length. The final window is the pattern
    # itself, and the second-to-last would leave no following token, so both are
    # excluded from the search.
    windows = tokens.unfold(0, min_match_len, 1)          # [L - m + 1, m]
    searchable = windows[:-1]                             # drop the pattern's own window
    if searchable.shape[0] == 0:
        return None

    hits = (searchable == pattern).all(dim=1)
    if not bool(hits.any()):
        return None

    # Most recent match: the highest index that matched.
    index = int(hits.shape[0] - 1 - int(hits.flip(0).float().argmax()))

    start = index + min_match_len
    draft = tokens[start: start + max_draft_len]
    if draft.numel() == 0:
        return None
    return draft.clone()


def one_hot_q(
    draft_tokens: torch.Tensor,
    vocab_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the ``[k, V]`` distribution a deterministic proposal implies.

    A one-hot row per proposed token. See the module docstring for why this is the
    mathematically correct ``q`` rather than a convenient stand-in.
    """
    tokens = draft_tokens.reshape(-1)
    q = torch.zeros(
        (tokens.shape[0], vocab_size), dtype=dtype, device=tokens.device
    )
    q.scatter_(1, tokens.unsqueeze(1), 1.0)
    return q
