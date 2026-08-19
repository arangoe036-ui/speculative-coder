"""Why greedy speculative decoding does not reproduce the baseline bitwise.

    python exactness_check.py

Speculative decoding is mathematically lossless: at temperature 0 it should
reproduce plain greedy decoding on the target token for token. On real GPU models
it does not, and this script isolates the reason by sweeping the *target's
precision* while holding everything else fixed -- same draft, same target
weights, same engine, same prompts, greedy throughout:

    fp32      float32     ~7 decimal digits of mantissa
    bf16      bfloat16    ~3 digits
    int8      LLM.int8    quantized

This sweep was written to test two hypotheses that both turned out to be wrong,
and it is kept because it is what falsified them. First guess: int8 kernels. But
bf16 diverged too. Second guess: reduced precision generally. But fp32 diverged
as well, at the *same token indices*, with a regime gap of 3e-05 against a
tightest top-2 margin of 9e-02 -- a thousand times too small to flip an argmax.

Identical divergence points across three precisions is not what numerical noise
looks like; it is what a systematic difference looks like. The actual cause was
in the baseline: Qwen ships `repetition_penalty: 1.1` in generation_config.json,
and transformers applies it even when do_sample is False, so the "greedy
baseline" was greedy-plus-penalty while the engine was pure greedy. Both
baselines here now override it. Precision is still worth sweeping, because it
governs how exact the match can be once the comparison is fair.

What is being tested is not the algorithm -- tests/test_engine.py already proves
token-for-token equivalence on fp32 CPU models across K and draft quality. It is
the *premise* the proof needs: that the target assigns a position the same
probabilities however it is evaluated. A baseline scores one token on top of a KV
cache; the engine scores a K+1-wide block. Those are mathematically identical and
numerically different, and the gap scales with precision. Where the target has a
near-tie, that gap decides it, and one flipped token changes everything after it.

A 1.5B target is used rather than the 7B so the fp32 arm fits in VRAM. The
numerical question does not depend on model size.
"""

from __future__ import annotations

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core.engine import SpeculativeEngine

DRAFT_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
TARGET_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

PROMPTS = [
    "Write a quick sort algorithm in Python.",
    "Write a Python function that reverses a linked list.",
    "Write a Python decorator that caches function results.",
    "Write a Python function that flattens a nested list.",
    "Write a Python class for a simple stack.",
]


def baseline_ids(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            # Qwen's generation_config.json sets repetition_penalty=1.1, and
            # transformers applies it even when do_sample is False. Without this
            # override the "greedy baseline" is not greedy at all.
            repetition_penalty=1.0, top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out[0, ids.shape[1]:].tolist()


def first_difference(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def load_target(precision: str):
    kwargs = {"device_map": "cuda:0"}
    if precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]
    model = AutoModelForCausalLM.from_pretrained(TARGET_ID, **kwargs)
    model.eval()
    return model


def measure_logit_gap(model, tokenizer, prompt: str, block: int = 6) -> dict:
    """How far apart are the two evaluation regimes, in this precision?

    Regime 1 scores the next token from the full prefix. Regime 2 scores the same
    position inside a `block`-wide forward, as the engine does. Reporting the gap
    alongside the divergence result is what turns "it diverged" into "it diverged
    because the premise fails by this much at this precision".
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        seq = model.generate(ids, max_new_tokens=64, do_sample=False,
                             repetition_penalty=1.0, top_k=0,
                             pad_token_id=tokenizer.eos_token_id)
        wide = model(input_ids=seq).logits[0].float()
        deltas, margins = [], []
        for pos in range(ids.shape[1] - 1, seq.shape[1] - 1):
            narrow = model(input_ids=seq[:, : pos + 1]).logits[0, -1].float()
            deltas.append(float((wide[pos] - narrow).abs().max()))
            top2 = torch.topk(narrow, 2).values
            margins.append(float(top2[0] - top2[1]))
    return {
        "max_delta": max(deltas), "mean_delta": sum(deltas) / len(deltas),
        "min_margin": min(margins),
    }


def run_arm(precision: str, draft_model, tokenizer, args) -> dict:
    names = {"fp32": "float32", "bf16": "bfloat16", "int8": "8-bit (LLM.int8)"}
    print(f"\n{'=' * 74}")
    print(f"ARM {precision}: target = {TARGET_ID} in {names[precision]}")
    print("=" * 74)

    target_model = load_target(precision)

    engine = SpeculativeEngine(
        draft_model, target_model, tokenizer, k=args.k, temperature=0.0
    )

    matches, results = 0, []
    for prompt in PROMPTS:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        base = baseline_ids(target_model, tokenizer, templated, args.max_new_tokens)
        _, stats = engine.generate(
            templated, max_new_tokens=args.max_new_tokens, stream=False
        )
        spec = list(stats.token_ids)

        same = spec == base
        matches += int(same)
        where = first_difference(base, spec)
        results.append({"prompt": prompt, "identical": same, "diverges_at": where})
        verdict = "identical" if same else f"diverges at token {where}"
        print(f"  {prompt[:48]:<50} {len(base):>3} tok  a={stats.acceptance_rate:5.1%}  {verdict}")

    gap = measure_logit_gap(target_model, tokenizer, tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPTS[0]}], tokenize=False, add_generation_prompt=True
    ))
    print(f"  regime logit gap: max {gap['max_delta']:.2e}  mean {gap['mean_delta']:.2e}"
          f"  | tightest top-2 margin {gap['min_margin']:.2e}")

    del target_model, engine
    gc.collect()
    torch.cuda.empty_cache()
    return {"label": precision, "matches": matches, "total": len(PROMPTS),
            "results": results, "gap": gap}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--precisions", nargs="+", default=["fp32", "bf16", "int8"],
                        choices=["fp32", "bf16", "int8"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this check needs a GPU.")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    print(f"Loading draft (bfloat16): {DRAFT_ID}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )
    draft_model.eval()

    arms = [run_arm(p, draft_model, tokenizer, args) for p in args.precisions]

    print(f"\n{'=' * 74}\nRESULT\n{'=' * 74}")
    header = f"  {'precision':<10} {'identical':>10} {'mean gap':>12} {'max gap':>12} {'min margin':>12}"
    print(header)
    for arm in arms:
        print(f"  {arm['label']:<10} {arm['matches']:>4}/{arm['total']:<5} "
              f"{arm['gap']['mean_delta']:>12.2e} {arm['gap']['max_delta']:>12.2e} "
              f"{arm['gap']['min_margin']:>12.2e}")

    exact = [a["label"] for a in arms if a["matches"] == a["total"]]
    inexact = [a["label"] for a in arms if a["matches"] < a["total"]]
    print()
    if exact and inexact:
        print(f"  CONFIRMED, and the earlier hypothesis was wrong. Exact at: "
              f"{', '.join(exact)}. Diverges at: {', '.join(inexact)}.")
        print("  Exactness is a question of numerical precision, not of quantization")
        print("  specifically: the two evaluation regimes agree to within the precision's")
        print("  resolution, and wherever the target has a near-tie tighter than that gap,")
        print("  the tie is decided by rounding. The speculative decoding loop is correct")
        print("  -- the premise it relies on (a shape-invariant target) is what fails.")
    elif not inexact:
        print("  Exact at every precision tested. The main benchmark's divergence at 8-bit")
        print("  is not reproduced here; the 7B target or a longer budget is implicated.")
    else:
        print("  Diverges at every precision tested, including fp32. Precision alone does")
        print("  not explain it, which points at a real bug in the loop that the small")
        print("  fp32 CPU tests do not reach. Investigate before trusting the benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
