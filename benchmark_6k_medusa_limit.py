"""Achievable ceiling for Medusa heads restricted to the top-N frequent code tokens.

The AST-restricted variant capped at 1.87x because structural tokens interleave with
content: 54% of positions are immediately followed by an identifier a keyword-only
head cannot propose. This measures what happens when the mask is widened from
"syntax" to "the N most frequent tokens in code", so heads may also emit `self`,
`i`, `None`, `result`.

Methodology, and why it matters here
------------------------------------
Ranking token frequency on the same corpus the coverage is then measured on is
circular, and in this setting it is catastrophically so. A few hundred distinct token
types appear in ~1,000 tokens of generated code, all of which fit inside a 6,000-token
budget, so a top-6k set drawn from that corpus covers it perfectly, the run length
pins at the cap, and the script reports a ~16x ceiling that means nothing.

So the corpora are kept disjoint:

* **Ranking** -- this repository's own Python source (~94k tokens, 5,920 distinct
  types), plus generated code from the first half of the prompt set. Real code,
  independent of what is being evaluated.
* **Evaluation** -- freshly generated code from the *held-out* half of the prompts.
  This is the target workload: what the model actually emits.

The circular number is computed too, and printed as a warning, so the size of the
trap is visible rather than merely asserted.

What this can and cannot establish
----------------------------------
It measures *coverage*: how far consecutive tokens fall inside the allowed set, which
bounds how many head proposals could possibly be accepted. It assumes heads that,
within that set, predict perfectly.

That assumption gets weaker as the mask widens, and the report says so. An
AST-restricted head chooses among ~6k syntactically forced options; a frequency-
restricted head must pick the exact right token out of 6,000, ten positions ahead,
from a single hidden state, with no knowledge of the intervening tokens. Widening the
mask moves the binding constraint from coverage to head accuracy -- which this
simulation cannot measure, because it would require training the heads.
"""

from __future__ import annotations

import argparse
import glob
import io
import statistics
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmark import PROMPTS, TARGET_ID
from core.ast_medusa import is_structural, structural_token_ids

MASK_SIZES = (1000, 3000, 6000, 12000)


def repo_corpus(tokenizer) -> tuple[Counter, int]:
    """Token frequencies over this repository's own Python source."""
    counts: Counter = Counter()
    total = 0
    for path in sorted(glob.glob("core/*.py") + glob.glob("tests/*.py") + glob.glob("*.py")):
        ids = tokenizer(io.open(path, encoding="utf-8").read()).input_ids
        counts.update(ids)
        total += len(ids)
    return counts, total


def generate(model, tokenizer, prompts, max_new_tokens: int) -> list[str]:
    texts = []
    for prompt_id, prompt in prompts:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(templated, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.0, top_k=0, pad_token_id=tokenizer.eos_token_id,
            )
        texts.append(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        print(f"  {prompt_id:<11} {out.shape[1] - ids.shape[1]:>4} tokens")
    return texts


def run_lengths(token_ids: list[int], allowed: set[int], cap: int) -> dict:
    """How far consecutive allowed tokens run, i.e. how many proposals could stick.

    The final ``cap`` positions have fewer than ``cap`` tokens after them, so their
    runs truncate and both ``mean_run`` and ``at_cap`` are very slightly
    under-counted. On a corpus of a thousand-plus tokens that is well under 1%.
    """
    flags = [t in allowed for t in token_ids]
    runs = []
    for i in range(len(flags) - 1):
        length = 0
        while length < cap and i + 1 + length < len(flags) and flags[i + 1 + length]:
            length += 1
        runs.append(length)

    histogram: Counter = Counter(runs)
    return {
        "tokens": len(flags),
        "coverage": sum(flags) / max(1, len(flags)),
        "mean_run": statistics.fmean(runs) if runs else 0.0,
        "median_run": statistics.median(runs) if runs else 0.0,
        "max_run": max(runs) if runs else 0,
        "at_cap": sum(1 for r in runs if r == cap) / max(1, len(runs)),
        "zero": histogram.get(0, 0) / max(1, len(runs)),
        "histogram": dict(sorted(histogram.items())),
    }


def vram_for(hidden: int, out_features: int, num_heads: int) -> tuple[float, float]:
    params = hidden * out_features * num_heads
    return params / 1e6, params * 2 / 2**20      # M params, MiB fp16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-heads", type=int, default=10, help="M")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--holdout", type=int, default=5,
                        help="prompts reserved for evaluation; the rest rank frequencies")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this simulation needs a GPU to generate code.")
        return 1

    M = args.num_heads
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    print(f"Loading base (8-bit): {TARGET_ID}")
    base = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda:0", dtype=torch.float16,
    ).eval()
    hidden = int(base.config.hidden_size)
    vocab = int(base.config.vocab_size)

    # ------------------------------------------------------------- corpora
    split = len(PROMPTS) - args.holdout
    print(f"\n{'=' * 78}\nCORPORA  (kept disjoint -- see module docstring)\n{'=' * 78}")
    print(f"Ranking: repo source + generated code from prompts 1-{split}")
    rank_counts, rank_repo_total = repo_corpus(tokenizer)
    print(f"  repo source: {rank_repo_total:,} tokens, "
          f"{len(rank_counts):,} distinct types")
    rank_texts = generate(base, tokenizer, PROMPTS[:split], args.max_new_tokens)
    for text in rank_texts:
        rank_counts.update(tokenizer(text).input_ids)
    rank_total = rank_repo_total + sum(len(tokenizer(t).input_ids) for t in rank_texts)
    print(f"  combined   : {rank_total:,} tokens, {len(rank_counts):,} distinct types")

    print(f"\nEvaluation: generated code from the {args.holdout} held-out prompts")
    eval_texts = generate(base, tokenizer, PROMPTS[split:], args.max_new_tokens)
    eval_ids: list[int] = []
    for text in eval_texts:
        eval_ids.extend(tokenizer(text).input_ids)
    print(f"  {len(eval_ids):,} tokens for evaluation")

    ast_ids = set(structural_token_ids(tokenizer, vocab).tolist())

    # ------------------------------------------------------------- sweep
    print(f"\n{'=' * 78}\nACHIEVABLE CEILING vs MASK SIZE  (M={M} heads, held-out code)\n{'=' * 78}")
    header = (f"  {'mask':<22}{'out dim':>8}{'coverage':>10}{'mean run':>10}"
              f"{'max':>5}{'zero':>7}{'at cap':>8}{'ceiling':>9}")
    print(header)

    rows = []
    ast_stats = run_lengths(eval_ids, ast_ids, M)
    rows.append(("AST-only (previous)", len(ast_ids), ast_stats))
    print(f"  {'AST-only (previous)':<22}{len(ast_ids):>8}{ast_stats['coverage']:>9.1%}"
          f"{ast_stats['mean_run']:>10.2f}{ast_stats['max_run']:>5}"
          f"{ast_stats['zero']:>7.0%}{ast_stats['at_cap']:>8.1%}"
          f"{1 + ast_stats['mean_run']:>8.2f}x")

    for size in MASK_SIZES:
        top = {t for t, _ in rank_counts.most_common(size)}
        allowed = top | ast_ids           # frequent tokens *alongside* AST structure
        stats = run_lengths(eval_ids, allowed, M)
        rows.append((f"top-{size} + AST", len(allowed), stats))
        print(f"  {f'top-{size} + AST':<22}{len(allowed):>8}{stats['coverage']:>9.1%}"
              f"{stats['mean_run']:>10.2f}{stats['max_run']:>5}"
              f"{stats['zero']:>7.0%}{stats['at_cap']:>8.1%}"
              f"{1 + stats['mean_run']:>8.2f}x")

    full = run_lengths(eval_ids, set(range(vocab)), M)
    rows.append(("unrestricted", vocab, full))
    print(f"  {'unrestricted':<22}{vocab:>8}{full['coverage']:>9.1%}"
          f"{full['mean_run']:>10.2f}{full['max_run']:>5}"
          f"{full['zero']:>7.0%}{full['at_cap']:>8.1%}"
          f"{1 + full['mean_run']:>8.2f}x")

    # ------------------------------------------------------------- the 6k detail
    target = next(s for name, _, s in rows if name == "top-6000 + AST")
    size_6k = next(n for name, n, _ in rows if name == "top-6000 + AST")
    print(f"\n{'=' * 78}\nTOP-6K DETAIL  (output dim {size_6k})\n{'=' * 78}")
    print(f"  coverage of held-out tokens : {target['coverage']:.2%}")
    print(f"  mean consecutive run        : {target['mean_run']:.2f}  "
          f"(AST-only was {ast_stats['mean_run']:.2f})")
    print(f"  median run                  : {target['median_run']:.1f}")
    print(f"  longest run observed        : {target['max_run']} of {M}")
    print(f"  positions accepting nothing : {target['zero']:.1%}  "
          f"(AST-only was {ast_stats['zero']:.1%})")
    print(f"  positions reaching all {M}   : {target['at_cap']:.1%}")
    print(f"\n  run-length histogram:")
    total = sum(target["histogram"].values())
    for length, count in target["histogram"].items():
        print(f"    {length:>2}: {count:>5} ({count / total:5.1%}) "
              + "#" * max(1, round(56 * count / total)))
    print(f"\n  ACHIEVABLE CEILING          : {1 + target['mean_run']:.2f}x")

    # ------------------------------------------------------------- circularity
    circ_top = {t for t, _ in Counter(eval_ids).most_common(6000)}
    circ = run_lengths(eval_ids, circ_top | ast_ids, M)
    print(f"\n{'=' * 78}\nTHE CIRCULAR NUMBER  (do not use)\n{'=' * 78}")
    print(f"  Ranking on the evaluation corpus itself gives coverage "
          f"{circ['coverage']:.1%},")
    print(f"  mean run {circ['mean_run']:.2f} and a ceiling of "
          f"{1 + circ['mean_run']:.2f}x -- because the {len(set(eval_ids))} distinct")
    print(f"  types in {len(eval_ids)} tokens all fit inside a 6,000-token budget.")
    print(f"  That is {(1 + circ['mean_run']) / (1 + target['mean_run']):.1f}x the honest "
          "figure and measures nothing.")

    # ------------------------------------------------------------- VRAM
    print(f"\n{'=' * 78}\nVRAM BLUEPRINT  (hidden={hidden}, M={M})\n{'=' * 78}")
    print(f"  {'output dim':>12}{'params':>12}{'fp16':>12}{'+ 8.11 GiB base':>18}")
    for name, out_dim, _ in rows:
        params_m, mib = vram_for(hidden, out_dim, M)
        print(f"  {out_dim:>12}{params_m:>11.1f}M{mib:>10.0f} MiB"
              f"{8.11 + mib / 1024:>16.2f} GiB   {name}")
    print(f"\n  Card has 15.92 GiB. A {size_6k}-wide head set at M={M} costs "
          f"{vram_for(hidden, size_6k, M)[1] / 1024:.2f} GiB,")
    print(f"  leaving {15.92 - 8.11 - vram_for(hidden, size_6k, M)[1] / 1024:.2f} GiB "
          "for KV cache, activations and the tree.")

    # ------------------------------------------------------------- verdict
    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    ceiling = 1 + target["mean_run"]
    print(f"  AST-only ceiling   {1 + ast_stats['mean_run']:5.2f}x")
    print(f"  top-6k ceiling     {ceiling:5.2f}x   "
          f"({ceiling / (1 + ast_stats['mean_run']):.1f}x better)")
    print(f"  unrestricted       {1 + full['mean_run']:5.2f}x   "
          f"(= 1 + M, the coverage ceiling with no mask at all)")
    print()
    if ceiling >= 5.0:
        print("  Widening the mask does solve the interleaving problem: coverage is no")
        print("  longer the binding constraint.")
    else:
        print("  Widening the mask helps but does not clear 5x on coverage alone.")
    print()
    print("  What this does NOT establish: that trained heads reach this ceiling.")
    print("  Coverage only says a proposal is *permitted*. An AST head picked among")
    print("  syntactically forced options; a top-6k head must name the exact right")
    print(f"  token out of {size_6k}, up to {M} positions ahead, from one hidden state")
    print("  that has not seen the intervening tokens. Widening the mask moves the")
    print("  binding constraint from coverage to head accuracy, and only a training")
    print("  run can measure that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
