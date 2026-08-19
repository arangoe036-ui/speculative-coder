"""Is early-exit verification safe, and does it fire? Measured before it is built.

The cascade idea rests on two empirical claims:

1. an intermediate layer's projection is often confident enough to accept a token; and
2. when it is confident, it agrees with the full model.

Claim 2 is the important one, because acting on claim 1 without it makes the engine
lossy -- accepting a token the target would not have produced. Both are measurable
from ordinary teacher-forced forwards, without building the early-exit machinery, so
they are measured first.

    python benchmark_cascade_limit.py
    python benchmark_cascade_limit.py --layers 10 14 18 22 --thresholds 0.9 0.95 0.99

Reported per (layer, threshold): how often the gate fires, how often a fired gate
agrees with the full model, and the resulting weight-traffic saving -- counting the
KV work that an early exit defers rather than deletes.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmark import PROMPTS, TARGET_ID
from core.cascade_verifier import decoder_layers, measure_cascade


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=[7, 14, 18, 22, 25])
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.90, 0.95, 0.99])
    parser.add_argument("--prompts", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this measurement needs a GPU.")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    print(f"Loading target (8-bit): {TARGET_ID}")
    base = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda:0", dtype=torch.float16,
    ).eval()
    total_layers = len(decoder_layers(base))
    print(f"  {total_layers} decoder layers")

    # Evaluate on the model's own greedy output: that is the distribution a verifier
    # actually sees, and it is what acceptance is decided against.
    print(f"\nGenerating {args.prompts} responses to measure on...")
    sequences = []
    for prompt_id, prompt in PROMPTS[: args.prompts]:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(templated, return_tensors="pt").input_ids.to(base.device)
        with torch.no_grad():
            out = base.generate(ids, max_new_tokens=args.max_new_tokens,
                                do_sample=False, repetition_penalty=1.0, top_k=0,
                                pad_token_id=tokenizer.eos_token_id)
        sequences.append(out)
        print(f"  {prompt_id:<11} {out.shape[1]} tokens")

    print(f"\n{'=' * 78}\nEARLY-EXIT GATE: does it fire, and is it right?\n{'=' * 78}")
    print(f"  {'layer':>6}{'thresh':>8}{'fires':>9}{'agree|fired':>13}"
          f"{'agree(all)':>12}{'wrong':>8}{'layers/tok':>12}{'vs full':>9}")

    rows = []
    for layer in args.layers:
        for threshold in args.thresholds:
            totals = {"positions": 0, "fired": 0, "agree_fired": 0,
                      "agree_all": 0, "wrong": 0}
            for ids in sequences:
                stats = measure_cascade(base, ids, layer, threshold)
                totals["positions"] += stats["positions"]
                totals["fired"] += stats["fired"]
                totals["agree_fired"] += round(stats["agreement"] * stats["fired"])
                totals["agree_all"] += round(stats["agreement_all"] * stats["positions"])
                totals["wrong"] += stats["wrong_accepts"]

            n = totals["positions"]
            fired = totals["fired"]
            fire_rate = fired / max(1, n)
            agreement = totals["agree_fired"] / max(1, fired)

            # Weight traffic per token. A fired gate reads `layer` layers now but
            # still owes the remaining ones for its KV entries; charging only the
            # first would count a deferral as a saving.
            run_only = fire_rate * layer + (1 - fire_rate) * total_layers
            with_debt = total_layers                     # the debt is always repaid
            rows.append((layer, threshold, fire_rate, agreement,
                         totals["agree_all"] / max(1, n), totals["wrong"],
                         run_only, with_debt))
            print(f"  {layer:>6}{threshold:>8.2f}{fire_rate:>8.1%}{agreement:>12.1%}"
                  f"{totals['agree_all'] / max(1, n):>11.1%}{totals['wrong']:>8}"
                  f"{run_only:>12.1f}{run_only / total_layers:>8.2f}x")

    print(f"\n{'=' * 78}\nREADING THIS\n{'=' * 78}")
    best = max(rows, key=lambda r: r[2] * r[3])          # fires often AND is right
    layer, threshold, fire_rate, agreement, agree_all, wrong, run_only, _ = best
    print(f"  Best fire x agreement: layer {layer}, threshold {threshold:.2f}")
    print(f"    fires on {fire_rate:.1%} of positions, agreeing {agreement:.1%} of the time")
    print(f"    that is {wrong} tokens the full model would have rejected")
    print()
    print("  `agree|fired` is the losslessness cost. Anything below 100% means the")
    print("  cascade emits tokens the target would not have produced, so the engine")
    print("  is an approximation of the model rather than an acceleration of it.")
    print()
    print("  `layers/tok` counts layers actually executed. It ignores that an early")
    print("  exit leaves layers", f"{layer}-{total_layers - 1}", "without KV entries for")
    print("  that position, which a later full pass must supply. Deferred work is")
    print("  only a saving if it can be repaid in a batch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
