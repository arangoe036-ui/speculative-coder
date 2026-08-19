"""Calibrate the statistical gate's threshold against measured acceptance.

    python gate_calibration.py

The gate's premise is that a low-confidence draft token is likely to be rejected,
so drafting past it wastes forwards. That premise is testable, and the threshold
should follow from the measurement rather than from a round number.

For every drafted token this records the draft's top-1 probability and whether
verification accepted it, then reports:

* the distribution of draft confidence, which says whether a given threshold can
  fire at all on this model pair;
* acceptance rate bucketed by confidence, which says whether confidence predicts
  rejection -- if acceptance were flat across buckets, the gate would be noise;
* for each candidate threshold, the fraction of blocks it would cut short and the
  acceptance rate of the tokens it would have skipped.

The last table is the decision: a good threshold cuts blocks whose remaining
tokens were going to be rejected anyway, and leaves the rest alone.
"""

from __future__ import annotations

import argparse
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmark import DRAFT_ID, PROMPTS, TARGET_ID
from core.engine import SpeculativeEngine

CANDIDATES = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95)


def collect(engine, tokenizer, prompts, max_new_tokens):
    """Run generation with gating disabled, recording confidence vs acceptance.

    ``entropy_threshold=0`` means the gate never fires, so the full block is
    always drafted and every token gets a verification verdict. Calibrating on
    gated runs would be circular: the gate would have removed exactly the tokens
    whose fate is the thing being measured.
    """
    blocks = []          # (confidences, n_accepted) per iteration
    for prompt_id, text in prompts:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        _, stats = engine.generate(templated, max_new_tokens=max_new_tokens, stream=False)
        for confidences, accepted in zip(stats.draft_confidences,
                                         stats.accepted_per_iteration):
            blocks.append((confidences, accepted))
        print(f"  {prompt_id:<11} {stats.iterations:>3} blocks  "
              f"alpha={stats.acceptance_rate:.1%}  "
              f"mean draft len {stats.mean_draft_length:.2f}")
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-draft-len", type=int, default=8)
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this calibration needs a GPU.")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    print(f"Loading target (8-bit): {TARGET_ID}")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda:0", dtype=torch.float16,
    ).eval()
    print(f"Loading draft (bfloat16): {DRAFT_ID}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_ID, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    engine = SpeculativeEngine(
        draft_model, target_model, tokenizer, temperature=0.0,
        use_dual_gate=True, max_draft_len=args.max_draft_len,
        entropy_threshold=0.0,          # record, never gate
    )

    print(f"\nCollecting over {args.limit} prompts, ceiling {args.max_draft_len}:")
    blocks = collect(engine, tokenizer, PROMPTS[: args.limit], args.max_new_tokens)

    # ---- flatten to per-token (confidence, accepted, index within block) ----
    rows = []
    for confidences, accepted in blocks:
        for i, conf in enumerate(confidences):
            rows.append((conf, i < accepted, i))
    confs = [c for c, _, _ in rows]
    print(f"\n{len(rows)} drafted tokens across {len(blocks)} blocks")

    # ---- 1. is the threshold even reachable? ----
    print(f"\n{'=' * 72}\n1. Draft top-1 confidence distribution\n{'=' * 72}")
    quantiles = [1, 5, 10, 25, 50, 75, 90]
    ordered = sorted(confs)
    for q in quantiles:
        idx = min(len(ordered) - 1, int(len(ordered) * q / 100))
        print(f"  p{q:<3} {ordered[idx]:.4f}")
    print(f"  mean {statistics.fmean(confs):.4f}   min {min(confs):.4f}")

    # ---- 2. does confidence predict acceptance? ----
    print(f"\n{'=' * 72}\n2. Acceptance rate by confidence bucket\n{'=' * 72}")
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.01]
    print(f"  {'bucket':<16}{'tokens':>8}{'share':>8}{'accepted':>10}")
    for low, high in zip(edges, edges[1:]):
        bucket = [ok for c, ok, _ in rows if low <= c < high]
        if not bucket:
            continue
        print(f"  [{low:.2f}, {high:.2f}){len(bucket):>8}"
              f"{len(bucket) / len(rows):>8.1%}"
              f"{sum(bucket) / len(bucket):>10.1%}")

    # ---- 3. what would each candidate threshold actually do? ----
    print(f"\n{'=' * 72}\n3. Effect of each candidate threshold\n{'=' * 72}")
    print(f"  {'thresh':>7}{'blocks cut':>12}{'draft fwds':>12}{'saved':>8}"
          f"{'skipped tok':>13}{'their alpha':>13}")
    baseline_forwards = sum(len(c) for c, _ in blocks)
    for threshold in CANDIDATES:
        cut_blocks = 0
        forwards = 0
        skipped_accepted = skipped_total = 0
        for confidences, accepted in blocks:
            stop = len(confidences)
            for i, conf in enumerate(confidences):
                if conf < threshold:
                    stop = i + 1        # the triggering token is kept
                    cut_blocks += 1
                    break
            forwards += stop
            for i in range(stop, len(confidences)):
                skipped_total += 1
                skipped_accepted += int(i < accepted)
        saved = 1 - forwards / baseline_forwards
        alpha_skipped = (skipped_accepted / skipped_total) if skipped_total else float("nan")
        print(f"  {threshold:>7.2f}{cut_blocks:>12}{forwards:>12}{saved:>8.1%}"
              f"{skipped_total:>13}{alpha_skipped:>13.1%}")

    print(f"\n{'=' * 72}\nReading this\n{'=' * 72}")
    print("  A threshold is worth using when it saves a real share of draft")
    print("  forwards AND the tokens it skips have low acceptance -- those were")
    print("  going to be thrown away. If the skipped tokens' acceptance is close")
    print("  to the overall rate, the gate is discarding good tokens and will")
    print("  cost throughput rather than save it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
