"""Physical latency limit of AST-guided Medusa: 15 tokens sequentially vs at once.

Measures, on the real 7B target:

    Time A   15 autoregressive base forwards, one token each (the status quo)
    Time B   1 base forward + all 15 heads evaluated in parallel

and reports the ratio. No training: this is the hardware ceiling, established before
committing to a training run.

Reading the ratio honestly
-------------------------
Time A / Time B will come out near M + 1, because it is close to a restatement of
"one forward costs a fifteenth of fifteen forwards". It is a real and useful upper
bound on the physics, but two things stand between it and a working system, and the
second is specific to the AST idea.

*Verification.* Medusa heads propose; they do not decide. Committing their output
unverified would emit the heads' distribution rather than the target's. In a correct
implementation the verification forward *is* the next iteration's base forward, so
the accounting still works out at ``1 + accepted`` tokens per base forward -- Time B
is the right per-iteration cost, but only if acceptance is counted honestly.

*The AST restriction caps acceptance structurally.* Heads that can only emit ``def``,
``:`` and indentation cannot propose an identifier. A Medusa step therefore accepts
only as far as the consecutive run of *structural* tokens extends. That is measurable
without any training, by classifying real generated code, and it is the ceiling that
governs a trained system. This script reports both numbers and keeps them apart.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmark import PROMPTS, TARGET_ID
from core.ast_medusa import MedusaAST, assert_modes_equivalent, is_structural


def _sync():
    torch.cuda.synchronize()


def time_sequential(model, ids, n_tokens: int, repeats: int) -> float:
    """Time A: n_tokens autoregressive base forwards, one token at a time."""
    samples = []
    for _ in range(repeats + 1):
        with torch.no_grad():
            primed = model(input_ids=ids, use_cache=True)
            cache = primed.past_key_values
            token = primed.logits[:, -1].argmax(dim=-1, keepdim=True)
            _sync()
            start = time.perf_counter()
            for _ in range(n_tokens):
                out = model(
                    input_ids=token, past_key_values=cache, use_cache=True
                )
                cache = out.past_key_values
                token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            _sync()
            samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def time_medusa(medusa, ids, repeats: int) -> tuple[float, float]:
    """Time B: one base forward plus all M heads. Returns (total, heads_only)."""
    totals, heads_only = [], []
    for _ in range(repeats + 1):
        with torch.no_grad():
            _sync()
            start = time.perf_counter()
            outputs = medusa.base_model(
                input_ids=ids, use_cache=True, output_hidden_states=True
            )
            hidden = outputs.hidden_states[-1][:, -1]
            _sync()
            mid = time.perf_counter()
            medusa.head_logits(hidden)
            _sync()
            end = time.perf_counter()
        totals.append(end - start)
        heads_only.append(end - mid)
    totals.sort()
    heads_only.sort()
    return totals[len(totals) // 2], heads_only[len(heads_only) // 2]


def structural_runs(tokenizer, texts: list[str], cap: int) -> dict:
    """How far consecutive structural tokens actually run in real generated code.

    This is the AST restriction's ceiling. For every position, the number of
    immediately following tokens that are all structural, capped at ``cap`` -- i.e.
    exactly how many of the M head proposals could possibly be accepted.
    """
    flags: list[bool] = []
    for text in texts:
        for token_id in tokenizer(text).input_ids:
            flags.append(is_structural(tokenizer.decode([token_id])))

    runs = []
    for i in range(len(flags) - 1):
        length = 0
        while length < cap and i + 1 + length < len(flags) and flags[i + 1 + length]:
            length += 1
        runs.append(length)

    histogram: dict[int, int] = {}
    for r in runs:
        histogram[r] = histogram.get(r, 0) + 1
    return {
        "tokens": len(flags),
        "structural_fraction": sum(flags) / max(1, len(flags)),
        "mean_run": statistics.fmean(runs) if runs else 0.0,
        "median_run": statistics.median(runs) if runs else 0.0,
        "max_run": max(runs) if runs else 0,
        "histogram": dict(sorted(histogram.items())),
        "reached_cap": sum(1 for r in runs if r == cap) / max(1, len(runs)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-heads", type=int, default=15, help="M")
    parser.add_argument("--head-mode", default="factored", choices=["factored", "full"])
    parser.add_argument("--context-len", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--sample-prompts", type=int, default=6,
                        help="prompts to generate for the structural-run analysis")
    parser.add_argument("--sample-tokens", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this measurement needs a GPU.")
        return 1

    M = args.num_heads
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    print(f"Loading base (8-bit, frozen): {TARGET_ID}")
    base = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda:0", dtype=torch.float16,
    ).eval()
    _sync()
    vram_base = torch.cuda.memory_allocated() / 2**30

    print(f"\n{'=' * 78}\nARCHITECTURE\n{'=' * 78}")
    medusa = MedusaAST(base, tokenizer, num_heads=M, head_mode=args.head_mode)
    _sync()
    vram_total = torch.cuda.memory_allocated() / 2**30
    print(f"  {medusa.parameter_report()}")
    print(f"  structural tokens: {medusa.n_structural} of {medusa.vocab_size} "
          f"({medusa.n_structural / medusa.vocab_size:.2%})")
    print(f"  VRAM: base {vram_base:.2f} GiB + heads "
          f"{vram_total - vram_base:.2f} GiB = {vram_total:.2f} GiB")
    trainable = sum(p.numel() for p in medusa.parameters() if p.requires_grad)
    print(f"  trainable parameters: {trainable / 1e6:.1f}M "
          f"(base fully frozen: {medusa.frozen_parameters / 1e9:.2f}B)")

    delta = assert_modes_equivalent(base, tokenizer, num_heads=2)
    print(f"  factored vs full-vocab masked heads: max softmax delta {delta:.2e} "
          f"({'identical' if delta < 1e-6 else 'DIVERGENT'})")

    # ---------------------------------------------------------------- latency
    print(f"\n{'=' * 78}\nRAW LATENCY  (context {args.context_len} tokens, "
          f"median of {args.repeats})\n{'=' * 78}")
    ids = torch.randint(1000, 5000, (1, args.context_len), device=base.device)

    time_a = time_sequential(base, ids, M, args.repeats)
    time_b, heads_time = time_medusa(medusa, ids, args.repeats)
    base_only = time_b - heads_time

    print(f"  Time A  {M} sequential base forwards : {time_a * 1000:8.1f} ms"
          f"   ({time_a * 1000 / M:.1f} ms/token)")
    print(f"  Time B  1 base forward + {M} heads    : {time_b * 1000:8.1f} ms")
    print(f"            of which base forward       : {base_only * 1000:8.1f} ms")
    print(f"            of which all {M} heads       : {heads_time * 1000:8.1f} ms "
          f"({heads_time / time_b:.1%} of Time B)")
    print()
    physics_speedup = time_a / time_b
    print(f"  PHYSICS CEILING  Time A / Time B     : {physics_speedup:8.2f}x")
    print(f"  tokens per base forward at 100% acc  : {1 + M}")
    print(f"  theoretical max throughput           : "
          f"{(1 + M) / time_b:8.1f} tok/s")

    # ---------------------------------------------------------------- ceiling
    print(f"\n{'=' * 78}\nACHIEVABLE CEILING  (what the AST restriction permits)\n{'=' * 78}")
    print(f"Generating {args.sample_prompts} real responses to measure how far "
          "structural\ntokens actually run consecutively...")
    texts = []
    for prompt_id, prompt in PROMPTS[: args.sample_prompts]:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(templated, return_tensors="pt").input_ids.to(base.device)
        with torch.no_grad():
            out = base.generate(
                encoded, max_new_tokens=args.sample_tokens, do_sample=False,
                repetition_penalty=1.0, top_k=0, pad_token_id=tokenizer.eos_token_id,
            )
        texts.append(tokenizer.decode(out[0, encoded.shape[1]:], skip_special_tokens=True))
        print(f"  {prompt_id:<11} {out.shape[1] - encoded.shape[1]:>4} tokens")

    stats = structural_runs(tokenizer, texts, cap=M)
    print(f"\n  tokens analysed              : {stats['tokens']}")
    print(f"  structural fraction          : {stats['structural_fraction']:.1%}")
    print(f"  mean consecutive run (cap {M}) : {stats['mean_run']:.2f}")
    print(f"  median run                   : {stats['median_run']:.1f}")
    print(f"  longest run observed         : {stats['max_run']}")
    print(f"  positions reaching all {M}    : {stats['reached_cap']:.2%}")
    print(f"\n  run-length histogram (how many head proposals could be accepted):")
    total = sum(stats["histogram"].values())
    for length, count in stats["histogram"].items():
        bar = "#" * max(1, round(60 * count / total))
        print(f"    {length:>2}: {count:>5} ({count / total:5.1%}) {bar}")

    achievable = 1 + stats["mean_run"]
    print(f"\n  tokens per base forward      : 1 + {stats['mean_run']:.2f} = "
          f"{achievable:.2f}")
    print(f"  ACHIEVABLE SPEEDUP           : {achievable:.2f}x")
    print(f"  achievable throughput        : {achievable / time_b:.1f} tok/s")

    # ---------------------------------------------------------------- verdict
    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    print(f"  physics ceiling      {physics_speedup:5.2f}x   "
          f"(perfect, unrestricted heads)")
    print(f"  achievable ceiling   {achievable:5.2f}x   "
          f"(perfect heads, AST-restricted)")
    print(f"  gap                  {physics_speedup / achievable:5.2f}x   "
          f"lost to the structural restriction alone")
    print()
    print("  The physics permits the target. The AST restriction does not: a head")
    print("  that can only emit keywords and punctuation cannot propose an")
    print(f"  identifier, and structural tokens run only {stats['mean_run']:.2f} deep on")
    print("  average in real code. Perfectly trained heads would still stop there.")
    print()
    print("  This is before any accuracy loss. Head i predicts t+i from h_t alone,")
    print("  with no knowledge of tokens t+1..t+i-1, so published Medusa accuracy")
    print("  decays sharply with i -- the achievable figure above is itself an")
    print("  optimistic bound on a trained system.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
