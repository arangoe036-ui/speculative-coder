"""Measure how a target forward scales with batch width.

Monte Carlo speculation bets that batch-1 decoding wastes the GPU: if a batch-4
forward costs about what a batch-1 forward costs, speculation breadth is nearly
free. That is an empirical claim about this model on this hardware, and it is the
premise the whole design rests on, so it is worth measuring before drawing
conclusions from a benchmark that assumes it.

Reports wall time for a (batch, positions) grid, plus the marginal cost of width.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmark import TARGET_ID


def time_forward(model, batch: int, positions: int, context_len: int, repeats: int) -> float:
    """Median seconds for one forward of `batch` rows x `positions` new tokens."""
    device = model.device
    prefix = torch.randint(1000, 5000, (batch, context_len), device=device)
    with torch.no_grad():
        # Build a cache for the prefix, so the timed call is a decode-shaped step.
        out = model(input_ids=prefix, use_cache=True)
        cache = out.past_key_values
        new_ids = torch.randint(1000, 5000, (batch, positions), device=device)

        samples = []
        for _ in range(repeats + 1):
            length = cache.get_seq_length()
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(
                input_ids=new_ids,
                past_key_values=cache,
                use_cache=True,
                attention_mask=torch.ones(
                    (batch, length + positions), dtype=torch.long, device=device
                ),
                cache_position=torch.arange(length, length + positions, device=device),
            )
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
            cache.crop(-positions)          # undo, so every sample is comparable
        samples.sort()
        return samples[len(samples) // 2]   # median, discarding the warm first call


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-len", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--positions", type=int, nargs="+", default=[1, 6])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this measurement needs a GPU.")
        return 1

    AutoTokenizer.from_pretrained(TARGET_ID)
    print(f"Loading target (8-bit): {TARGET_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda:0", dtype=torch.float16,
    ).eval()

    print(f"\nContext {args.context_len} tokens, median of {args.repeats} timed calls\n")
    for positions in args.positions:
        print(f"=== {positions} new position(s) per row ===")
        print(f"  {'batch':>6}{'ms':>9}{'vs batch 1':>12}{'ms/row':>9}")
        base = None
        for batch in args.batches:
            seconds = time_forward(model, batch, positions, args.context_len, args.repeats)
            base = base or seconds
            print(f"  {batch:>6}{seconds * 1000:>9.1f}{seconds / base:>11.2f}x"
                  f"{seconds * 1000 / batch:>9.1f}")
        print()

    print("Reading this: if widening the batch is nearly free, the 'vs batch 1'")
    print("column stays near 1.00x and speculation breadth costs almost nothing.")
    print("If it climbs linearly, the forward is already compute-bound and every")
    print("extra branch is paid for in full.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
