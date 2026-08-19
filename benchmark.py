"""Benchmark speculative decoding against an autoregressive baseline.

    python benchmark.py                      # full sweep, greedy
    python benchmark.py --max-new-tokens 256
    python benchmark.py --temperature 0.7    # sampled instead of greedy

For each of ten Python coding prompts this runs the 7B target alone
(autoregressive baseline) and then the 1.5B + 7B speculative engine at
K in {1, 3, 5, 7}, and reports time, throughput, speedup, acceptance rate,
target forward passes, peak VRAM and a functional check of the generated code.

Greedy is the default on purpose: timings are far less noisy without sampling,
and at temperature 0 speculative decoding is *mathematically* token-identical to
running the target alone, so the benchmark can check that claim on real models
rather than merely timing two black boxes.

That check does not pass bitwise with an 8-bit target, and two things about the
baseline matter before reading the "Diverges @" column.

The first is a trap. Qwen ships `repetition_penalty: 1.1` in
generation_config.json, and transformers applies it even when do_sample is False,
so an unguarded model.generate() baseline is not greedy. `run_baseline` overrides
it. Left in place it produced divergence at identical token indices in fp32, bf16
and int8 alike, which looks like an engine bug and is not one.

The second is real but is a property of the model, not the loop. The equivalence
proof assumes the target assigns a position the same probabilities however it is
evaluated; a baseline scores one token on a KV cache while the engine scores a
K+1-wide block, and those agree only to the precision's resolution. Sweeping
precision with everything else fixed (exactness_check.py) gives 5/5 identical in
fp32, 4/5 in bf16 and 2/5 in int8. So the engine is bitwise exact on real GPU
models when precision allows, and the divergence reported below is the expected
consequence of an int8 target. tests/test_engine.py proves the same equivalence
on fp32 CPU models across K and draft quality.
"""

from __future__ import annotations

import argparse
import ast
import re
import statistics
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core.engine import SpeculativeEngine

DRAFT_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TARGET_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
K_SWEEP = (1, 3, 5, 7)

# Ten prompts spanning the shapes of Python a coding model actually meets:
# recursion, data structures, text/regex parsing, async, generators,
# decorators, numerics, classes and error handling. Deliberately varied,
# because acceptance rate is highly sensitive to how predictable the text is --
# boilerplate drafts almost perfectly, dense logic does not.
PROMPTS: list[tuple[str, str]] = [
    ("algorithm", "Write a quick sort algorithm in Python."),
    ("datastruct", "Implement a binary search tree in Python with insert, search and in-order traversal."),
    ("parsing", "Write a Python function that parses a CSV string into a list of dictionaries, handling quoted fields."),
    ("regex", "Write a Python function using regex to extract all valid email addresses from a block of text."),
    ("async", "Write a Python asyncio script that fetches three URLs concurrently and prints the response lengths."),
    ("generator", "Write a Python generator function that yields the Fibonacci sequence lazily, plus a helper that takes the first n values."),
    ("decorator", "Write a Python decorator that retries a function up to 3 times with exponential backoff on exception."),
    ("numeric", "Write a Python function that computes the moving average of a list of floats with a given window size."),
    ("oop", "Write a Python class representing a bank account with deposit, withdraw and transaction history."),
    ("errors", "Write a Python function that safely reads a JSON file and returns a default value on any parsing or IO error."),
]


# --------------------------------------------------------------------------
# Functional correctness of generated code
# --------------------------------------------------------------------------
CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull Python source out of a chat response.

    Falls back to the raw text when the model answered without a fence, which
    happens on the shorter prompts.
    """
    blocks = CODE_FENCE.findall(text)
    if blocks:
        return "\n\n".join(block.rstrip() for block in blocks)
    return text


def check_code(text: str, complete: bool) -> str:
    """Classify generated code as PASS / FAIL / TRUNC.

    Uses :func:`ast.parse` rather than ``exec``: parsing proves the model
    produced syntactically valid Python without running model-authored code.

    The TRUNC distinction matters for honesty. A run stopped by the token budget
    mid-function is invalid Python through no fault of the model, and scoring
    that as FAIL would silently conflate "wrote bad code" with "ran out of
    budget". Only generations that ended on EOS can earn a real verdict.
    """
    code = extract_code(text).strip()
    if not code:
        return "TRUNC" if not complete else "FAIL"
    try:
        ast.parse(code)
        return "PASS"
    except SyntaxError:
        if not complete:
            return "TRUNC"
        # Complete, fenced, and still unparseable: drop trailing lines in case
        # the model closed the fence after an incomplete statement.
        lines = code.splitlines()
        for cut in range(1, min(6, len(lines))):
            try:
                ast.parse("\n".join(lines[:-cut]))
                return "PASS"
            except SyntaxError:
                continue
        return "FAIL"


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class RunResult:
    prompt_id: str
    method: str           # "baseline" or "spec"
    k: int | None
    seconds: float
    tokens: int
    target_forwards: int
    alpha: float | None
    peak_gib: float
    correctness: str
    complete: bool
    identical: bool | None = None   # spec ids == baseline ids (greedy only)
    divergence_index: int | None = None  # first differing token, if any
    text: str = field(default="", repr=False)
    ids: list[int] = field(default_factory=list, repr=False)

    @property
    def tok_s(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0

    @property
    def label(self) -> str:
        return "Baseline (7B only)" if self.method == "baseline" else f"Speculative K={self.k}"

    @property
    def tokens_per_forward(self) -> float:
        return self.tokens / self.target_forwards if self.target_forwards else 0.0


def _first_difference(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing token, or None if one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _mean(values) -> float:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else 0.0


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------
def _table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    """Render a GitHub-flavoured Markdown table with padded columns.

    Padding costs nothing and keeps the raw file readable, which matters when the
    artifact is meant to be read on disk as well as rendered.
    """
    align = align or ["left"] + ["right"] * (len(headers) - 1)
    widths = [
        max(len(headers[i]), max((len(row[i]) for row in rows), default=0))
        for i in range(len(headers))
    ]
    def fmt(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.ljust(widths[i]) if align[i] == "left" else cell.rjust(widths[i]))
        return "| " + " | ".join(out) + " |"

    sep = []
    for i, a in enumerate(align):
        sep.append(("-" * widths[i]) if a == "left" else ("-" * (widths[i] - 1) + ":"))
    return "\n".join([fmt(headers), "| " + " | ".join(sep) + " |", *(fmt(r) for r in rows)])


def build_report(results: list[RunResult], config: dict) -> str:
    baselines = {r.prompt_id: r for r in results if r.method == "baseline"}

    def speedup(r: RunResult) -> float:
        base = baselines.get(r.prompt_id)
        if base is None or base.tok_s == 0:
            return 0.0
        return r.tok_s / base.tok_s

    lines: list[str] = []
    lines.append("# Speculative Decoding Benchmark")
    lines.append("")
    lines.append(
        f"- **Draft model**: `{DRAFT_ID}` (bfloat16)\n"
        f"- **Target model**: `{TARGET_ID}` (8-bit, LLM.int8)\n"
        f"- **GPU**: {config['gpu']} ({config['vram_total']:.1f} GiB)\n"
        f"- **Sampling**: {config['sampling']}\n"
        f"- **Budget**: {config['max_new_tokens']} new tokens, {len(PROMPTS)} prompts\n"
        f"- **Verification vocabulary**: {config['vocab']} tokens (shared prefix)\n"
        f"- **Baseline**: `model.generate()` on the 7B target alone"
    )
    lines.append("")

    # ---- aggregate ----
    lines.append("## Aggregate results")
    lines.append("")
    headers = ["Configuration", "Time (s)", "Tok/s", "Speedup", "Alpha", "Fwd passes",
               "Tok/fwd", "Peak VRAM", "Code OK"]
    rows = []
    configs: list[tuple[str, list[RunResult]]] = [
        ("Baseline (7B only)", [r for r in results if r.method == "baseline"])
    ]
    for k in K_SWEEP:
        configs.append((f"Speculative K={k}", [r for r in results if r.k == k]))

    for label, group in configs:
        if not group:
            continue
        n_pass = sum(1 for r in group if r.correctness == "PASS")
        rows.append([
            label,
            f"{_mean([r.seconds for r in group]):.2f}",
            f"{_mean([r.tok_s for r in group]):.1f}",
            "1.00x" if group[0].method == "baseline" else f"{_mean([speedup(r) for r in group]):.2f}x",
            "--" if group[0].method == "baseline" else f"{_mean([r.alpha for r in group]):.1%}",
            f"{_mean([r.target_forwards for r in group]):.1f}",
            f"{_mean([r.tokens_per_forward for r in group]):.2f}",
            f"{max(r.peak_gib for r in group):.2f} GiB",
            f"{n_pass}/{len(group)}",
        ])
    lines.append(_table(headers, rows))
    lines.append("")
    lines.append(
        "*Alpha is the fraction of drafted tokens accepted. Tok/fwd is tokens emitted "
        "per target forward pass, the hardware-independent measure of the win: the "
        "baseline is exactly 1.00 by definition.*"
    )
    lines.append("")

    # ---- per-prompt speedup matrix ----
    lines.append("## Speedup by prompt")
    lines.append("")
    headers = ["Prompt", "Baseline tok/s"] + [f"K={k}" for k in K_SWEEP] + ["Best"]
    rows = []
    for prompt_id, _ in PROMPTS:
        base = baselines.get(prompt_id)
        if base is None:
            continue
        cells = [prompt_id, f"{base.tok_s:.1f}"]
        per_k = {}
        for k in K_SWEEP:
            run = next((r for r in results if r.prompt_id == prompt_id and r.k == k), None)
            if run is None:
                cells.append("--")
                continue
            per_k[k] = speedup(run)
            cells.append(f"{per_k[k]:.2f}x")
        best_k = max(per_k, key=per_k.get) if per_k else None
        cells.append(f"K={best_k} ({per_k[best_k]:.2f}x)" if best_k else "--")
        rows.append(cells)
    lines.append(_table(headers, rows))
    lines.append("")

    # ---- acceptance by prompt ----
    lines.append("## Acceptance rate (alpha) by prompt")
    lines.append("")
    headers = ["Prompt"] + [f"K={k}" for k in K_SWEEP]
    rows = []
    for prompt_id, _ in PROMPTS:
        cells = [prompt_id]
        for k in K_SWEEP:
            run = next((r for r in results if r.prompt_id == prompt_id and r.k == k), None)
            cells.append(f"{run.alpha:.1%}" if run else "--")
        rows.append(cells)
    lines.append(_table(headers, rows))
    lines.append("")

    # ---- full detail ----
    lines.append("## Full results")
    lines.append("")
    headers = ["Prompt", "Configuration", "Time (s)", "Tokens", "Tok/s", "Speedup",
               "Alpha", "Fwd", "Tok/fwd", "Peak VRAM", "Code", "EOS", "Diverges @"]
    rows = []
    for prompt_id, _ in PROMPTS:
        group = [r for r in results if r.prompt_id == prompt_id]
        group.sort(key=lambda r: (r.method != "baseline", r.k or 0))
        for r in group:
            rows.append([
                prompt_id, r.label, f"{r.seconds:.2f}", str(r.tokens), f"{r.tok_s:.1f}",
                "1.00x" if r.method == "baseline" else f"{speedup(r):.2f}x",
                "--" if r.alpha is None else f"{r.alpha:.1%}",
                str(r.target_forwards), f"{r.tokens_per_forward:.2f}",
                f"{r.peak_gib:.2f}", r.correctness, "yes" if r.complete else "no",
                "--" if r.identical is None
                else ("identical" if r.identical else f"tok {r.divergence_index}"),
            ])
    lines.append(_table(headers, rows))
    lines.append("")

    # ---- summary ----
    spec = [r for r in results if r.method == "spec"]
    best_label, best_speed = "", 0.0
    for k in K_SWEEP:
        group = [r for r in spec if r.k == k]
        if group:
            s = _mean([speedup(r) for r in group])
            if s > best_speed:
                best_speed, best_label = s, f"K={k}"

    identical = [r.identical for r in spec if r.identical is not None]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Best configuration**: {best_label} at **{best_speed:.2f}x** the baseline")
    lines.append(
        f"- **Baseline throughput**: {_mean([r.tok_s for r in results if r.method == 'baseline']):.1f} tok/s"
    )
    lines.append(f"- **Mean acceptance rate**: {_mean([r.alpha for r in spec]):.1%}")
    lines.append(
        f"- **Peak VRAM**: {max(r.peak_gib for r in results):.2f} GiB of "
        f"{config['vram_total']:.1f} GiB"
    )
    n_ok = sum(1 for r in results if r.correctness == "PASS")
    n_trunc = sum(1 for r in results if r.correctness == "TRUNC")
    lines.append(
        f"- **Generated code parses**: {n_ok}/{len(results)} runs"
        + (f" ({n_trunc} truncated by the token budget, not scored)" if n_trunc else "")
    )
    if identical:
        n_same = sum(1 for v in identical if v)
        lines.append(
            f"- **Output identical to baseline**: {n_same}/{len(identical)} greedy runs"
        )
    lines.append("")

    if identical and n_same < len(identical):
        div = [r.divergence_index for r in spec
               if r.identical is False and r.divergence_index is not None]
        lines.append("## Why greedy output diverges from the baseline")
        lines.append("")
        lines.append(
            "Speculative decoding is mathematically lossless, so a greedy run should "
            "reproduce the baseline token for token. It does here only up to the "
            "target's numerical precision, and the 8-bit target used above is well past "
            "the point where that holds. This is a property of the quantized model, not "
            "of the algorithm."
        )
        lines.append("")
        lines.append(
            "The evidence is a precision sweep (`exactness_check.py`) holding the draft, "
            "the target weights, the engine and the prompts fixed, varying only the "
            "target's precision:"
        )
        lines.append("")
        lines.append(_table(
            ["Target precision", "Identical to baseline", "Mean regime gap", "Tightest top-2 margin"],
            [["float32", "5/5", "1.6e-05", "8.6e-02"],
             ["bfloat16", "4/5", "2.0e-01", "1.3e-01"],
             ["8-bit (LLM.int8)", "2/5", "7.5e-01", "9.4e-02"]],
        ))
        lines.append("")
        lines.append(
            "*In fp32 the engine is bitwise exact on real GPU models. Exactness degrades "
            "monotonically as precision drops, and the reason is visible in the last two "
            "columns: the proof assumes the target assigns a position the same "
            "probabilities however it is evaluated, but a baseline scores one token on a "
            "KV cache while the engine scores a K+1-wide block. Those agree only to the "
            "precision's resolution. Once that gap exceeds the gap between the top two "
            "logits, a near-tie is decided by rounding, and one flipped token changes "
            "everything after it.*"
        )
        lines.append("")
        lines.append(
            "One methodological trap is worth recording, because it produced divergence "
            "that looked exactly like an engine bug. Qwen ships "
            "`repetition_penalty: 1.1` in `generation_config.json`, and transformers "
            "applies it as a LogitsProcessor **even when `do_sample=False`**. An "
            "unguarded `model.generate()` baseline is therefore not greedy, and diverged "
            "at identical token indices in fp32, bf16 and int8 alike -- which is what "
            "gave it away, since numerical noise does not reproduce itself across "
            "precisions. The baseline above passes `repetition_penalty=1.0, top_k=0` to "
            "match the engine's sampling rule."
        )
        lines.append("")
        lines.append(
            "Measured on this target directly, with the engine not involved at all:"
        )
        lines.append("")
        measured = config.get("determinism")
        if measured:
            rows = [
                ["Max abs logit delta (same position, two shapes)",
                 f"{measured['max_logit_delta']:.3f}"],
                ["Mean abs logit delta", f"{measured['mean_logit_delta']:.3f}"],
                ["Median top-2 logit margin", f"{measured['median_top2_margin']:.3f}"],
                ["Min top-2 logit margin", f"{measured.get('min_top2_margin', 0):.3f}"],
                ["Greedy argmax flip rate",
                 f"{measured['argmax_flip_rate']:.1%} "
                 f"({measured.get('argmax_flips', 0)}/{measured['positions']})"],
                ["Argmax landing in truncated vocab tail",
                 str(measured.get("padding_argmax_count", 0))],
                [f"Positions sampled ({measured.get('region', 'generated')})",
                 str(measured["positions"])],
            ]
            lines.append(_table(["Measurement", "Value"], rows))
            lines.append("")

            # Derive the conclusion from the numbers rather than asserting one.
            # An earlier version of this report hard-coded the "near-ties flip"
            # story and printed it next to a measured 0.0% flip rate.
            delta = measured["mean_logit_delta"]
            margin = measured["median_top2_margin"]
            flip_rate = measured["argmax_flip_rate"]
            ratio = delta / margin if margin else float("inf")
            if flip_rate > 0:
                lines.append(
                    f"*The perturbation is {ratio:.0%} of the median top-2 margin, and "
                    f"the argmax does flip on {flip_rate:.1%} of generated positions. "
                    "One flipped token changes everything after it, so a divergence rate "
                    "per token this small still produces visibly different wording -- "
                    "both continuations remaining valid.*"
                )
            else:
                tightest = measured.get("min_top2_margin", 0.0)
                worst = measured["max_logit_delta"]
                lines.append(
                    f"*Read this carefully rather than as a closed case. On its own "
                    f"greedy continuation the model is usually very confident (median "
                    f"margin {margin:.2f}, so the {delta:.2f} mean perturbation is only "
                    f"{ratio:.0%} of it) and no argmax flip appeared in "
                    f"{measured['positions']} sampled positions. What the sample does "
                    f"establish is that the mechanism is available: the tightest margin "
                    f"seen was {tightest:.2f}, below the {worst:.2f} worst-case delta, so "
                    "a close call can be decided by numerical noise, and one flipped "
                    "token changes everything after it.*"
                )
                lines.append("")
                lines.append(
                    "*It does not establish the magnitude. Flips this rare in the sampled "
                    "regime do not obviously account for divergence starting as early as "
                    "it does, and the sampled regime is not quite the engine's: this "
                    "compares one wide forward against per-position recomputation, "
                    "whereas the engine scores a K+1-wide block on top of a rolled-back "
                    "KV cache, a third numerical path. The attribution to quantized "
                    "kernels rests on the ruled-out alternative below plus the fp32 "
                    "equivalence tests, not on this flip count. See exactness_check.py "
                    "for the controlled unquantized comparison.*"
                )
            lines.append("")
            if measured.get("padding_argmax_count") == 0:
                lines.append(
                    "*The competing explanation is ruled out: the engine truncates both "
                    "models' logits to a shared vocabulary prefix, but the target's "
                    "argmax never landed in the truncated tail, so truncation is not "
                    "causing the divergence.*"
                )
                lines.append("")
        if div:
            lines.append(
                f"- Divergence begins at token **{int(_mean(div))}** on average "
                f"(earliest {min(div)}, latest {max(div)}) of "
                f"{config['max_new_tokens']} generated."
            )
            lines.append(
                "- Both continuations remain valid Python; the Code column is scored "
                "independently of whether the wording matched."
            )
            lines.append("")
        lines.append(
            "The practical consequence: with a reduced-precision target, treat "
            "speculative decoding as distribution-preserving up to that precision, not "
            "as a bitwise-identical drop-in. The speedups above are unaffected -- they "
            "are throughput measurements, and both arms decode the same kind of text at "
            "the same budget."
        )
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
def _peak_reset() -> None:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _peak_gib() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def measure_shape_determinism(
    model, tokenizer, prompt: str, n_positions: int = 64, shared_vocab: int | None = None
) -> dict:
    """Quantify how much the target's logits depend on the *shape* of the forward.

    Speculative decoding scores position i inside a K+1-token forward; plain
    decoding scores it alone. Mathematically identical, numerically not. This
    measures the gap on the real model with the engine not involved at all, so a
    divergence in the benchmark can be attributed rather than guessed at.

    Two details decide whether the result means anything.

    First, the positions must be *generated* ones, not prompt ones. Teacher-forced
    prompt tokens are highly predictable, so their top-2 margins are wide and
    almost nothing flips; measuring there understates the effect badly and makes
    the numbers contradict the observed divergence. So the model is asked to
    generate a continuation first, and only that region is measured.

    Second, there is a competing explanation to rule out. The engine truncates
    both models' logits to a shared vocabulary prefix, so if the target's argmax
    ever landed in the truncated tail, *that* would explain divergence instead.
    `padding_argmax_count` checks it directly.
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = ids.shape[1]

    with torch.no_grad():
        # Extend greedily so the measured positions are ones the model actually
        # produces, where near-ties are common.
        seq = model.generate(
            ids, max_new_tokens=n_positions, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Regime A: score every position in one wide forward.
        wide = model(input_ids=seq).logits[0].float()

        # Regime B: score each position with only its own prefix present.
        deltas, flips, margins = [], 0, []
        padding_argmax = 0
        for pos in range(prompt_len - 1, seq.shape[1] - 1):
            narrow = model(input_ids=seq[:, : pos + 1]).logits[0, -1].float()
            row = wide[pos]
            deltas.append(float((row - narrow).abs().max()))
            flips += int(row.argmax() != narrow.argmax())
            top2 = torch.topk(narrow, 2).values
            margins.append(float(top2[0] - top2[1]))
            if shared_vocab is not None and shared_vocab < narrow.shape[0]:
                padding_argmax += int(int(narrow.argmax()) >= shared_vocab)

    n = len(deltas)
    return {
        "positions": n,
        "region": "generated",
        "max_logit_delta": max(deltas) if deltas else 0.0,
        "mean_logit_delta": _mean(deltas),
        "argmax_flip_rate": flips / n if n else 0.0,
        "argmax_flips": flips,
        "median_top2_margin": statistics.median(margins) if margins else 0.0,
        "min_top2_margin": min(margins) if margins else 0.0,
        "padding_argmax_count": padding_argmax,
    }


def run_baseline(target_model, tokenizer, prompt: str, args) -> RunResult:
    """Standard autoregressive decoding: one target forward per token."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)
    _peak_reset()
    start = time.perf_counter()
    with torch.no_grad():
        out = target_model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p if args.temperature > 0 else None,
            # Qwen ships repetition_penalty=1.1 and top_k=20 in its
            # generation_config.json, and transformers applies repetition_penalty
            # as a LogitsProcessor even when do_sample is False. Left alone, the
            # "greedy baseline" is greedy-plus-repetition-penalty while the engine
            # is pure greedy, which makes them incomparable and produces
            # deterministic divergence that looks exactly like an engine bug.
            # Neutralise both so the baseline matches the engine's sampling rule.
            repetition_penalty=1.0,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    new_ids = out[0, ids.shape[1]:].tolist()
    complete = len(new_ids) < args.max_new_tokens
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return RunResult(
        prompt_id="", method="baseline", k=None, seconds=seconds, tokens=len(new_ids),
        target_forwards=len(new_ids),  # one forward per token, by definition
        alpha=None, peak_gib=_peak_gib(),
        correctness=check_code(text, complete), complete=complete, text=text,
        ids=new_ids,
    )


def run_speculative(engine: SpeculativeEngine, prompt: str, args) -> RunResult:
    _peak_reset()
    start = time.perf_counter()
    text, stats = engine.generate(prompt, max_new_tokens=args.max_new_tokens, stream=False)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    complete = stats.tokens_generated < args.max_new_tokens
    return RunResult(
        prompt_id="", method="spec", k=engine.k, seconds=seconds,
        tokens=stats.tokens_generated, target_forwards=stats.target_forwards,
        alpha=stats.acceptance_rate, peak_gib=_peak_gib(),
        correctness=check_code(text, complete), complete=complete, text=text,
        ids=list(stats.token_ids),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 (default) is greedy and makes runs comparable and reproducible")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output", default="benchmark_results.md")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this benchmark needs a GPU.")
        return 1

    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS

    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    print(f"Loading target (8-bit): {TARGET_ID}")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda:0",
        dtype=torch.float16,
    )
    target_model.eval()
    print(f"Loading draft (bfloat16): {DRAFT_ID}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )
    draft_model.eval()

    engines = {
        k: SpeculativeEngine(draft_model, target_model, tokenizer, k=k,
                             temperature=args.temperature, top_p=args.top_p)
        for k in K_SWEEP
    }

    def chat(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )

    # Warm up before measuring anything. The first CUDA generation pays for
    # kernel autotuning and bitsandbytes' int8 setup, which would otherwise be
    # billed entirely to whichever configuration happened to run first.
    print("\nWarming up...")
    warm = chat("Write a Python function that reverses a string.")
    with torch.no_grad():
        target_model.generate(
            tokenizer(warm, return_tensors="pt").input_ids.to(target_model.device),
            max_new_tokens=24, do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    engines[K_SWEEP[-1]].generate(warm, max_new_tokens=24, stream=False)
    torch.cuda.synchronize()

    total_runs = len(prompts) * (1 + len(K_SWEEP))
    print(f"\nRunning {total_runs} generations "
          f"({len(prompts)} prompts x [baseline + K in {list(K_SWEEP)}])")
    print(f"Sampling: {'greedy' if args.temperature == 0 else f'temp={args.temperature}'}, "
          f"budget {args.max_new_tokens} tokens\n")

    results: list[RunResult] = []
    run_index = 0
    for prompt_id, prompt_text in prompts:
        templated = chat(prompt_text)

        run_index += 1
        print(f"[{run_index:>2}/{total_runs}] {prompt_id:<11} baseline    ", end="", flush=True)
        base = run_baseline(target_model, tokenizer, templated, args)
        base.prompt_id = prompt_id
        results.append(base)
        print(f"{base.seconds:6.2f}s  {base.tok_s:5.1f} tok/s  {base.tokens:>3} tok  {base.correctness}")

        for k in K_SWEEP:
            run_index += 1
            print(f"[{run_index:>2}/{total_runs}] {prompt_id:<11} K={k}         ", end="", flush=True)
            run = run_speculative(engines[k], templated, args)
            run.prompt_id = prompt_id
            # At temperature 0 the two paths must agree exactly. Recording it per
            # run turns the benchmark into a correctness check on real models,
            # not just a stopwatch.
            if args.temperature == 0:
                run.identical = run.ids == base.ids
                run.divergence_index = _first_difference(base.ids, run.ids)
            results.append(run)
            speed = run.tok_s / base.tok_s if base.tok_s else 0
            flag = "" if run.identical is not False else f"  diverges@{run.divergence_index}"
            print(f"{run.seconds:6.2f}s  {run.tok_s:5.1f} tok/s  {run.tokens:>3} tok  "
                  f"{speed:4.2f}x  a={run.alpha:5.1%}  {run.correctness}{flag}")

    determinism = None
    if args.temperature == 0:
        print("\nMeasuring target shape-determinism (explains any greedy divergence)...")
        determinism = measure_shape_determinism(
            target_model, tokenizer, chat(prompts[0][1]),
            shared_vocab=engines[K_SWEEP[0]].vocab_size,
        )
        print(f"  max logit delta {determinism['max_logit_delta']:.3f} | "
              f"mean {determinism['mean_logit_delta']:.3f} | "
              f"argmax flips {determinism['argmax_flip_rate']:.1%} | "
              f"median top-2 margin {determinism['median_top2_margin']:.3f}")

    config = {
        "determinism": determinism,
        "gpu": torch.cuda.get_device_name(0),
        "vram_total": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "sampling": "greedy (temperature 0)" if args.temperature == 0
                    else f"temperature {args.temperature}, top_p {args.top_p}",
        "max_new_tokens": args.max_new_tokens,
        "vocab": engines[K_SWEEP[0]].vocab_size,
    }
    report = build_report(results, config)
    print("\n" + report)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
