"""Training and evaluation for AST/top-K Medusa heads.

Phase 1 established a *coverage* ceiling: with a top-6k mask, consecutive tokens
fall inside the allowed set deeply enough to permit 5.69x. Coverage only says a
proposal is permitted. This module measures the number that actually matters --
whether a trained head names the right token -- and turns the resulting per-head
accuracy curve into a projected speedup.

The training target
-------------------
A head must predict what the *base model* will emit, not what a corpus happens to
say next. Both are available from a single teacher-forced forward pass: the base
logits at position ``j`` give the model's greedy next token ``g_j``, so head ``i``
is trained to predict ``g_{t+i-1}`` from the hidden state ``h_t``. No generation
required, and the target is exactly the thing acceptance is decided against.

That framing also supplies a free sanity check. Head 1 predicts ``g_t`` from
``h_t``, which is precisely what the base model's own ``lm_head`` computes -- a
linear map from the same vector. So head 1 is distilling a linear function of its
own input and must reach near-ceiling accuracy. If it does not, the training loop
is broken rather than the architecture.

Honest accounting for the mask
------------------------------
When the true target lies outside the allowed set the head *cannot* emit it. Those
positions are excluded from the loss (no gradient direction helps) but counted as
failures in the accuracy metric. So the reported accuracy folds in both failure
modes -- outside-the-mask and inside-but-wrong -- and needs no separate coverage
correction.
"""

from __future__ import annotations

import math
import time

import torch
from torch import nn


@torch.no_grad()
def extract_features(
    base_model,
    tokenizer,
    texts: list[str],
    num_heads: int,
    chunk_len: int = 512,
    max_tokens: int | None = None,
    device: str = "cuda:0",
    progress_every: int = 50,
):
    """Collect ``(hidden state, base greedy token)`` pairs from a corpus.

    One teacher-forced forward per chunk yields both: the final hidden state at each
    position, and the model's own greedy prediction for the following position.

    Chunks are processed independently and the last ``num_heads`` positions of each
    are dropped, because their targets would fall outside the chunk -- crossing that
    boundary would train heads against tokens the hidden state never conditioned on.

    Returns ``(hidden [N, H] fp16 on CPU, greedy [N + num_heads] long on CPU)``
    per-chunk-concatenated, plus the token count consumed.
    """
    hidden_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    consumed = 0
    chunks = 0

    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        for start in range(0, ids.numel() - num_heads - 1, chunk_len):
            chunk = ids[start: start + chunk_len]
            if chunk.numel() < num_heads + 2:
                continue
            batch = chunk.unsqueeze(0).to(device)
            out = base_model(input_ids=batch, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1][0]              # [T, H]
            greedy = out.logits[0].argmax(dim=-1)          # [T]  g_j for j = 0..T-1

            # Head i (1-indexed) predicts g_{t + i - 1}. Keep only positions whose
            # deepest target is still inside this chunk.
            usable = chunk.numel() - num_heads
            hidden_parts.append(hidden[:usable].to(torch.float16).cpu())
            target_parts.append(
                torch.stack(
                    [greedy[i: i + usable] for i in range(num_heads)]
                ).cpu()                                     # [M, usable]
            )
            consumed += chunk.numel()
            chunks += 1
            if progress_every and chunks % progress_every == 0:
                print(f"    {chunks} chunks, {consumed:,} tokens", flush=True)
            if max_tokens and consumed >= max_tokens:
                return (
                    torch.cat(hidden_parts),
                    torch.cat(target_parts, dim=1),
                    consumed,
                )

    return torch.cat(hidden_parts), torch.cat(target_parts, dim=1), consumed


def compact_targets(targets: torch.Tensor, structural_ids: torch.Tensor) -> torch.Tensor:
    """Map vocabulary ids to indices in the masked output space.

    Targets outside the mask become ``-100`` so cross-entropy ignores them: there is
    no gradient direction that makes an impossible token more likely. They are still
    counted as failures at evaluation time.
    """
    lookup = torch.full(
        (int(structural_ids.max()) + 2,), -100, dtype=torch.long
    )
    lookup[structural_ids] = torch.arange(structural_ids.numel())
    clipped = targets.clamp(max=lookup.numel() - 1)
    mapped = lookup[clipped]
    mapped[targets >= lookup.numel()] = -100
    return mapped


def train_heads(
    medusa,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 3,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = "cuda:0",
    log_every: int = 200,
):
    """Fit all heads by cross-entropy on cached features.

    Each head is a separate multinomial logistic regression on a frozen feature, so
    the problem is convex per head and converges quickly; the interesting question
    is not optimisation but how much signal ``h_t`` carries about token ``t + i``.
    """
    heads = medusa.medusa_heads.to(device=device, dtype=torch.float32)
    compact = compact_targets(targets, medusa.structural_ids.cpu())

    optimiser = torch.optim.Adam(heads.parameters(), lr=lr)
    n = hidden.shape[0]
    history: list[dict] = []
    step = 0
    start = time.perf_counter()

    for epoch in range(epochs):
        order = torch.randperm(n)
        running = 0.0
        seen = 0
        for begin in range(0, n - batch_size + 1, batch_size):
            index = order[begin: begin + batch_size]
            features = hidden[index].to(device=device, dtype=torch.float32)
            batch_targets = compact[:, index].to(device)

            loss = features.new_zeros(())
            for i, head in enumerate(heads):
                logits = head(features)
                loss = loss + nn.functional.cross_entropy(
                    logits, batch_targets[i], ignore_index=-100
                )
            loss = loss / len(heads)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            running += float(loss)
            seen += 1
            step += 1
            if log_every and step % log_every == 0:
                print(f"    epoch {epoch + 1} step {step:>5} "
                      f"loss {running / seen:.4f}", flush=True)
        history.append({"epoch": epoch + 1, "loss": running / max(1, seen)})
        print(f"  epoch {epoch + 1}/{epochs}  mean loss {running / max(1, seen):.4f}  "
              f"({time.perf_counter() - start:.0f}s)", flush=True)
    return history


@torch.no_grad()
def evaluate_heads(
    medusa,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 2048,
    device: str = "cuda:0",
) -> list[float]:
    """Per-head top-1 accuracy against the base model's own greedy tokens.

    Targets outside the mask count as failures, since the head physically cannot
    emit them. The returned accuracies therefore already include the coverage loss
    that Phase 1 measured separately.
    """
    heads = medusa.medusa_heads
    structural = medusa.structural_ids.to(device)
    correct = [0] * len(heads)
    total = 0

    for begin in range(0, hidden.shape[0], batch_size):
        features = hidden[begin: begin + batch_size].to(device=device, dtype=torch.float32)
        batch_targets = targets[:, begin: begin + batch_size].to(device)
        total += features.shape[0]
        for i, head in enumerate(heads):
            predicted = structural[head(features).argmax(dim=-1)]
            correct[i] += int((predicted == batch_targets[i]).sum())

    return [c / max(1, total) for c in correct]


def expected_accepted(accuracies: list[float]) -> float:
    """Expected accepted tokens per iteration from a per-head accuracy curve.

    A Medusa step commits the longest prefix of heads that are *all* correct, so the
    expectation is ``sum_j prod_{i<=j} a_i``. The product is why the curve's tail
    matters so little: one weak early head caps everything behind it.
    """
    total = 0.0
    running = 1.0
    for accuracy in accuracies:
        running *= accuracy
        total += running
    return total


def projected_speedup(accuracies: list[float]) -> dict:
    """Turn measured accuracies into the throughput projection."""
    accepted = expected_accepted(accuracies)
    return {
        "expected_accepted": accepted,
        "tokens_per_forward": 1.0 + accepted,
        "speedup": 1.0 + accepted,
        "chain": [math.prod(accuracies[: j + 1]) for j in range(len(accuracies))],
    }
