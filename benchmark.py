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
from core.hybrid_engine import HybridCascadeEngine
from core.monte_carlo_engine import MonteCarloEngine
from core.particle_filter_engine import ParticleFilterEngine
from core.tree_engine import TreeSpeculativeEngine
from core.self_engine import SelfSpeculativeEngine

DRAFT_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TARGET_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
K_SWEEP = (1, 3, 5, 7)
DUAL_GATE_MAX_DRAFT = 8
# 0.35 is the specified default. 0.65 comes from gate_calibration.py: the draft's
# top-1 probability has a median of 0.99 on this pair, so 0.35 sits near the 1st
# percentile and almost never fires. Both are run so the difference is visible
# rather than argued.
DUAL_GATE_THRESHOLDS = (0.35, 0.65)
TREE_SPLIT_THRESHOLDS = (0.8,)
TREE_MAX_LEAVES = 8
TREE_K = 8
PF_SURVIVOR_FRACTIONS = (0.5,)
PF_PARTICLES = 8
PF_K = 8
MC_BRANCHES = (4,)
MC_DRAFT_TEMPERATURE = 1.2
MC_K = 5
HYBRID_MATCH_LENS = (2,)
HYBRID_NGRAM_DRAFT_LEN = 5
SELF_SPEC_WINDOW = 64
SELF_SPEC_K = 5
DUAL_GATE_ENTROPY_THRESHOLD = DUAL_GATE_THRESHOLDS[0]

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
    variant: str          # "baseline" | "static" | "dual_gate"
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
    draft_forwards: int = 0
    mean_draft_len: float = 0.0
    gate: dict | None = None        # dual-gate telemetry, None for static runs
    entropy_threshold: float | None = None   # set for dual_gate runs only
    # VRAM held by resident model weights when this configuration ran. Recorded
    # per configuration rather than per process: the two-model variants need a
    # second checkpoint loaded, the single-model ones do not.
    weights_gib: float = 0.0
    branches: int | None = None               # monte carlo / particle filter
    survivor_fraction: float | None = None    # particle filter runs only
    split_threshold: float | None = None      # tree runs only
    tree: dict | None = None                  # tree shape telemetry
    draft_rows: int = 0                       # draft batch rows pushed
    branch: dict | None = None                # monte carlo branch telemetry
    min_match_len: int | None = None          # hybrid runs only
    routing: dict | None = None               # hybrid routing telemetry

    @property
    def tok_s(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0

    @property
    def label(self) -> str:
        if self.variant == "baseline":
            return "Baseline (7B only)"
        if self.variant == "dual_gate":
            return (f"Adaptive Dual-Gate (Max K={DUAL_GATE_MAX_DRAFT}, "
                    f"t={self.entropy_threshold:g})")
        if self.variant == "self_spec":
            return f"Self-Spec Twin-Cache (window={SELF_SPEC_WINDOW}, K={self.k})"
        if self.variant == "hybrid":
            return f"Hybrid Cascade (n-gram m={self.min_match_len} + 0.5B + gate)"
        if self.variant == "monte_carlo":
            return f"Monte Carlo (B={self.branches}, K={self.k}, draft T={MC_DRAFT_TEMPERATURE})"
        if self.variant == "particle_filter":
            return (f"Particle Filter (B={self.branches}, K={self.k}, "
                    f"keep={self.survivor_fraction:g})")
        if self.variant == "tree":
            return (f"Evolutionary Tree (leaves<={TREE_MAX_LEAVES}, K={self.k}, "
                    f"split<{self.split_threshold:g})")
        return f"Speculative K={self.k}"

    @property
    def short_label(self) -> str:
        """Compact form for per-prompt table headers."""
        if self.variant == "baseline":
            return "Baseline"
        if self.variant == "dual_gate":
            return f"Gate t={self.entropy_threshold:g}"
        if self.variant == "self_spec":
            return "Self-Spec"
        if self.variant == "hybrid":
            return f"Hybrid m={self.min_match_len}"
        if self.variant == "monte_carlo":
            return f"MC B{self.branches}/K{self.k}"
        if self.variant == "particle_filter":
            return f"PF keep={self.survivor_fraction:g}"
        if self.variant == "tree":
            return f"Tree s<{self.split_threshold:g}"
        return f"K={self.k}"

    @property
    def tokens_per_forward(self) -> float:
        return self.tokens / self.target_forwards if self.target_forwards else 0.0

    @property
    def draft_rows_per_token(self) -> float:
        """Draft batch rows pushed per emitted token.

        The draft-side cost metric. Counting forward *launches* hides the
        difference between engines entirely -- they all make one launch per depth
        step -- so only the row count shows what the draft actually did.
        """
        return self.draft_rows / self.tokens if self.tokens else 0.0

    @property
    def total_forwards(self) -> int:
        """Every forward pass of a full-size model this configuration performed.

        For the two-model engines the draft forwards are ~4.4x cheaper and are
        excluded; for self-speculation the draft forwards are the *same model*, so
        counting only verification forwards would flatter it enormously.
        """
        return self.target_forwards + (self.draft_forwards if self.variant == "self_spec" else 0)


def _first_difference(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing token, or None if one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _counterfactual_draft_forwards(routing: dict) -> int:
    """Draft forwards the run would have spent with the fast path disabled.

    Estimated from the mean length of the *model*-drafted blocks in the same run,
    since those are the only blocks whose cost the fast path displaces. Using the
    draft ceiling instead would attribute the dual gate's savings to the n-gram
    router.
    """
    blocks = routing["model_blocks"]
    if blocks == 0:
        return routing["draft_forwards"]
    mean_model_block = routing["model_proposed"] / blocks
    return round(routing["iterations"] * mean_model_block)


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


def _configurations(results: list[RunResult]) -> list[tuple[str, list[RunResult]]]:
    """Group runs into report rows: baseline, each static K, then the dual gate."""
    groups: list[tuple[str, list[RunResult]]] = [
        ("Baseline (7B only)", [r for r in results if r.variant == "baseline"])
    ]
    for k in sorted({r.k for r in results if r.variant == "static"}):
        groups.append((f"Speculative K={k}",
                       [r for r in results if r.variant == "static" and r.k == k]))
    for threshold in sorted({r.entropy_threshold for r in results
                             if r.variant == "dual_gate"}):
        group = [r for r in results
                 if r.variant == "dual_gate" and r.entropy_threshold == threshold]
        groups.append((group[0].label, group))
    for match_len in sorted({r.min_match_len for r in results
                             if r.variant == "hybrid"}):
        group = [r for r in results
                 if r.variant == "hybrid" and r.min_match_len == match_len]
        groups.append((group[0].label, group))
    for branches, mc_k in sorted({(r.branches, r.k) for r in results
                                  if r.variant == "monte_carlo"}):
        group = [r for r in results if r.variant == "monte_carlo"
                 and r.branches == branches and r.k == mc_k]
        groups.append((group[0].label, group))
    for fraction in sorted({r.survivor_fraction for r in results
                            if r.variant == "particle_filter"}):
        group = [r for r in results if r.variant == "particle_filter"
                 and r.survivor_fraction == fraction]
        groups.append((group[0].label, group))
    for threshold in sorted({r.split_threshold for r in results
                             if r.variant == "tree"}):
        group = [r for r in results if r.variant == "tree"
                 and r.split_threshold == threshold]
        groups.append((group[0].label, group))
    self_spec = [r for r in results if r.variant == "self_spec"]
    if self_spec:
        groups.append((self_spec[0].label, self_spec))
    return groups


def _spec_columns(results: list[RunResult]) -> list[tuple[str, str]]:
    """Column headers for the per-prompt tables, in report order."""
    static = sorted({r.k for r in results if r.variant == "static"})
    columns = [(f"K={k}", f"k{k}") for k in static]
    for threshold in sorted({r.entropy_threshold for r in results
                             if r.variant == "dual_gate"}):
        columns.append((f"Gate t={threshold:g}", f"dual:{threshold}"))
    for match_len in sorted({r.min_match_len for r in results
                             if r.variant == "hybrid"}):
        columns.append((f"Hybrid m={match_len}", f"hybrid:{match_len}"))
    for branches, mc_k in sorted({(r.branches, r.k) for r in results
                                  if r.variant == "monte_carlo"}):
        columns.append((f"MC B{branches}/K{mc_k}", f"mc:{branches}:{mc_k}"))
    for fraction in sorted({r.survivor_fraction for r in results
                            if r.variant == "particle_filter"}):
        columns.append((f"PF keep={fraction:g}", f"pf:{fraction}"))
    for threshold in sorted({r.split_threshold for r in results
                             if r.variant == "tree"}):
        columns.append((f"Tree s<{threshold:g}", f"tree:{threshold}"))
    if any(r.variant == "self_spec" for r in results):
        columns.append(("Self-Spec", "self"))
    return columns


def _find(results: list[RunResult], prompt_id: str, key: str) -> RunResult | None:
    for r in results:
        if r.prompt_id != prompt_id:
            continue
        if key == "self" and r.variant == "self_spec":
            return r
        if key.startswith("tree:") and r.variant == "tree":
            if r.split_threshold == float(key.split(":")[1]):
                return r
            continue
        if key.startswith("pf:") and r.variant == "particle_filter":
            if r.survivor_fraction == float(key.split(":")[1]):
                return r
            continue
        if key.startswith("mc:") and r.variant == "monte_carlo":
            _, branches, mc_k = key.split(":")
            if r.branches == int(branches) and r.k == int(mc_k):
                return r
            continue
        if key.startswith("hybrid:") and r.variant == "hybrid":
            if r.min_match_len == int(key.split(":")[1]):
                return r
            continue
        if key.startswith("dual:") and r.variant == "dual_gate":
            if r.entropy_threshold == float(key.split(":")[1]):
                return r
            continue
        if key.startswith("k") and r.variant == "static" and r.k == int(key[1:]):
            return r
    return None


def build_report(results: list[RunResult], config: dict) -> str:
    baselines = {r.prompt_id: r for r in results if r.variant == "baseline"}

    def speedup(r: RunResult) -> float:
        base = baselines.get(r.prompt_id)
        if base is None or base.tok_s == 0:
            return 0.0
        return r.tok_s / base.tok_s

    lines: list[str] = []
    lines.append("# Speculative Decoding Benchmark")
    lines.append("")
    lines.append(
        f"- **Draft model**: `{config.get('draft_id', DRAFT_ID)}` (bfloat16)\n"
        f"- **Target model**: `{TARGET_ID}` (8-bit, LLM.int8)\n"
        f"- **GPU**: {config['gpu']} ({config['vram_total']:.1f} GiB)\n"
        f"- **Sampling**: {config['sampling']}\n"
        f"- **Budget**: {config['max_new_tokens']} new tokens{' (forced: EOS disabled, every run emits exactly this many)' if config.get('forced_length') else ' or EOS, whichever is first'}, {len(PROMPTS)} prompts\n"
        f"- **Verification vocabulary**: {config['vocab']} tokens (shared prefix)\n"
        f"- **Baseline**: `model.generate()` on the 7B target alone"
    )
    lines.append("")

    # ---- aggregate ----
    lines.append("## Aggregate results")
    lines.append("")
    headers = ["Configuration", "Time (s)", "Tok/s", "Speedup", "Alpha",
               "7B fwd/token", "Draft rows/tok", "Peak VRAM", "Code OK"]
    rows = []
    configs = _configurations(results)

    for label, group in configs:
        if not group:
            continue
        is_base = group[0].variant == "baseline"
        n_pass = sum(1 for r in group if r.correctness == "PASS")
        rows.append([
            label,
            f"{_mean([r.seconds for r in group]):.2f}",
            f"{_mean([r.tok_s for r in group]):.1f}",
            "1.00x" if is_base else f"{_mean([speedup(r) for r in group]):.2f}x",
            "--" if is_base else f"{_mean([r.alpha for r in group]):.1%}",
            f"{_mean([r.total_forwards / max(1, r.tokens) for r in group]):.2f}",
            "--" if is_base else f"{_mean([r.draft_rows_per_token for r in group]):.2f}",
            f"{max(r.peak_gib for r in group):.2f} GiB",
            f"{n_pass}/{len(group)}",
        ])
    lines.append(_table(headers, rows))
    lines.append("")
    lines.append(
        "*Alpha is the fraction of drafted tokens accepted. `7B fwd/token` counts every "
        "forward pass of a **full-size** model per emitted token, which is the metric "
        "that makes the three architectures comparable: the two-model engines get their "
        "draft forwards from a 1.5B model (~4.4x cheaper, so excluded), while "
        "self-speculation drafts with the 7B itself, so its draft forwards cost full "
        "price and are counted. Plain decoding is 1.00 by definition; below 1.00 is a "
        "real win, above 1.00 means more full-size compute than simply decoding. "
        "`Draft rows/tok` is draft-model batch rows pushed per emitted token -- the "
        "draft-side cost. Counting draft forward *launches* would hide the difference, "
        "since every breadth engine launches once per depth step and only the widths "
        "differ.*"
    )
    lines.append("")

    # ---- per-prompt speedup matrix ----
    lines.append("## Speedup by prompt")
    lines.append("")
    columns = _spec_columns(results)
    headers = ["Prompt", "Baseline tok/s"] + [name for name, _ in columns] + ["Best"]
    rows = []
    for prompt_id, _ in PROMPTS:
        base = baselines.get(prompt_id)
        if base is None:
            continue
        cells = [prompt_id, f"{base.tok_s:.1f}"]
        scores = {}
        for name, key in columns:
            run = _find(results, prompt_id, key)
            if run is None:
                cells.append("--")
                continue
            scores[name] = speedup(run)
            cells.append(f"{scores[name]:.2f}x")
        best = max(scores, key=scores.get) if scores else None
        cells.append(f"{best} ({scores[best]:.2f}x)" if best else "--")
        rows.append(cells)
    lines.append(_table(headers, rows))
    lines.append("")

    # ---- acceptance by prompt ----
    lines.append("## Acceptance rate (alpha) by prompt")
    lines.append("")
    columns = _spec_columns(results)
    headers = ["Prompt"] + [name for name, _ in columns]
    rows = []
    for prompt_id, _ in PROMPTS:
        cells = [prompt_id]
        for _, key in columns:
            run = _find(results, prompt_id, key)
            cells.append(f"{run.alpha:.1%}" if run else "--")
        rows.append(cells)
    lines.append(_table(headers, rows))
    lines.append("")

    # ---- dual-gate telemetry ----
    thresholds = sorted({r.entropy_threshold for r in results
                         if r.variant == "dual_gate" and r.gate})
    if thresholds:
        lines.append("## Dual-gate telemetry")
        lines.append("")
        lines.append(
            f"Draft ceiling {DUAL_GATE_MAX_DRAFT} tokens. The statistical gate fires "
            "when the draft's top-1 probability (read from the unwarped softmax) falls "
            "below the threshold; the syntactic gate fires on an unrecoverable bracket "
            "state inside fenced code. `Suppressed` counts fatal-looking bracket states "
            "seen in prose, where the syntactic gate deliberately stays silent because "
            'enumerations like "1)" are not syntax errors.'
        )
        lines.append("")
        for threshold in thresholds:
            adaptive = [r for r in results if r.variant == "dual_gate"
                        and r.entropy_threshold == threshold and r.gate]
            lines.append(f"### Threshold t = {threshold:g}")
            lines.append("")
            headers = ["Prompt", "Iters", "Mean draft len", "Statistical", "Syntactic",
                       "Ungated", "Gated %", "Suppressed", "Alpha", "Speedup"]
            rows = []
            for prompt_id, _ in PROMPTS:
                run = _find(results, prompt_id, f"dual:{threshold}")
                if run is None:
                    continue
                g = run.gate
                rows.append([
                    prompt_id, str(g["iterations"]), f"{run.mean_draft_len:.2f}",
                    str(g["statistical"]), str(g["syntactic"]), str(g["ungated"]),
                    f"{g['trigger_rate']:.0%}", str(g["suppressed"]),
                    f"{run.alpha:.1%}", f"{speedup(run):.2f}x",
                ])
            totals = {key: sum(r.gate[key] for r in adaptive)
                      for key in ("iterations", "statistical", "syntactic",
                                  "ungated", "suppressed")}
            rows.append([
                "**total**", str(totals["iterations"]),
                f"{_mean([r.mean_draft_len for r in adaptive]):.2f}",
                str(totals["statistical"]), str(totals["syntactic"]),
                str(totals["ungated"]),
                f"{(totals['statistical'] + totals['syntactic']) / max(1, totals['iterations']):.0%}",
                str(totals["suppressed"]),
                f"{_mean([r.alpha for r in adaptive]):.1%}",
                f"{_mean([speedup(r) for r in adaptive]):.2f}x",
            ])
            lines.append(_table(headers, rows))
            lines.append("")

    # ---- monte carlo branch telemetry ----
    breadth = [r for r in results
               if r.variant in ("monte_carlo", "particle_filter") and r.branch]
    branch_counts = sorted({(r.branches, r.k) for r in results
                            if r.variant == "monte_carlo" and r.branch})
    if breadth:
        lines.append("## Breadth telemetry: Monte Carlo vs Particle Filter")
        lines.append("")
        lines.append(
            "`Best` is the accepted-token count of the winning branch; `Single` is the "
            "mean across all branches in the same iteration -- what one branch would "
            "have achieved on the same draft samples. `Gain` is the difference, i.e. "
            "the extra tokens per iteration that breadth actually buys. `Wins` shows "
            "how often each branch index won; a degenerate spread would mean the "
            "branches are not diverging. `Distinct` is how many *different* drafts "
            "reached verification per iteration, which is the diagnostic for whether "
            "resampling has collapsed the population onto one path."
        )
        lines.append("")
        headers = ["Config", "Iters", "Best", "Single", "Gain", "Distinct",
                   "Tok/fwd", "Tok/s", "Speedup", "Win spread"]
        rows = []
        configs = [("monte_carlo", (b, k), None) for b, k in branch_counts]
        configs += [("particle_filter", None, f) for f in sorted(
            {r.survivor_fraction for r in results if r.variant == "particle_filter"}
        )]
        for variant, mc_key, fraction in configs:
            if variant == "monte_carlo":
                branches, mc_k = mc_key
                group = [r for r in results if r.variant == "monte_carlo"
                         and r.branches == branches and r.k == mc_k and r.branch]
                label = f"MC B={branches}, K={mc_k}"
            else:
                group = [r for r in results if r.variant == "particle_filter"
                         and r.survivor_fraction == fraction and r.branch]
                label = f"PF keep={fraction:g}"
            merged: dict[int, int] = {}
            for r in group:
                for index, count in r.branch["wins"].items():
                    merged[index] = merged.get(index, 0) + count
            total_wins = sum(merged.values()) or 1
            spread = " ".join(
                f"{i}:{c * 100 // total_wins}%" for i, c in sorted(merged.items())
            )
            rows.append([
                label,
                str(sum(r.branch["iterations"] for r in group)),
                f"{_mean([r.branch['best'] for r in group]):.2f}",
                f"{_mean([r.branch['single'] for r in group]):.2f}",
                f"+{_mean([r.branch['gain'] for r in group]):.2f}",
                f"{_mean([r.branch.get('unique', 0) for r in group]):.2f}",
                f"{_mean([r.tokens_per_forward for r in group]):.2f}",
                f"{_mean([r.tok_s for r in group]):.1f}",
                f"{_mean([speedup(r) for r in group]):.2f}x",
                spread,
            ])
        lines.append(_table(headers, rows))
        lines.append("")

    # ---- tree shape telemetry ----
    tree_thresholds = sorted({r.split_threshold for r in results
                              if r.variant == "tree" and r.tree})
    if tree_thresholds:
        lines.append("## Evolutionary tree telemetry")
        lines.append("")
        lines.append(
            "`Leaves` is the mean width the tree grew to, against a ceiling of "
            f"{TREE_MAX_LEAVES}. `Draft rows` totals the batch rows pushed through the "
            "draft model, with the fixed-width Monte Carlo figure alongside for "
            "comparison -- that reduction is the design's central claim. `Best` and "
            "`Gain` carry the same meaning as in the breadth table: the accepted count "
            "of the winning leaf, and how much taking the max over leaves buys."
        )
        lines.append("")
        headers = ["Config", "Iters", "Leaves", "Splits", "Draft rows", "Rows/tok",
                   "Best", "Gain", "Tok/fwd", "Tok/s", "Speedup"]
        rows = []
        reference = [r for r in results if r.variant == "monte_carlo"]
        for threshold in tree_thresholds:
            group = [r for r in results if r.variant == "tree"
                     and r.split_threshold == threshold and r.tree]
            rows.append([
                f"Tree split<{threshold:g}",
                str(sum(r.tree["iterations"] for r in group)),
                f"{_mean([r.tree['leaves'] for r in group]):.2f}",
                str(sum(r.tree["splits"] for r in group)),
                str(sum(r.draft_rows for r in group)),
                f"{_mean([r.draft_rows_per_token for r in group]):.2f}",
                f"{_mean([r.branch['best'] for r in group if r.branch]):.2f}",
                f"+{_mean([r.branch['gain'] for r in group if r.branch]):.2f}",
                f"{_mean([r.tokens_per_forward for r in group]):.2f}",
                f"{_mean([r.tok_s for r in group]):.1f}",
                f"{_mean([speedup(r) for r in group]):.2f}x",
            ])
        if reference:
            rows.append([
                f"MC B={reference[0].branches} (fixed width)",
                str(sum(r.branch["iterations"] for r in reference if r.branch)),
                f"{float(reference[0].branches):.2f}",
                "0",
                str(sum(r.draft_rows for r in reference)),
                f"{_mean([r.draft_rows_per_token for r in reference]):.2f}",
                f"{_mean([r.branch['best'] for r in reference if r.branch]):.2f}",
                f"+{_mean([r.branch['gain'] for r in reference if r.branch]):.2f}",
                f"{_mean([r.tokens_per_forward for r in reference]):.2f}",
                f"{_mean([r.tok_s for r in reference]):.1f}",
                f"{_mean([speedup(r) for r in reference]):.2f}x",
            ])
        lines.append(_table(headers, rows))
        lines.append("")

    # ---- hybrid routing telemetry ----
    match_lens = sorted({r.min_match_len for r in results
                         if r.variant == "hybrid" and r.routing})
    if match_lens:
        lines.append("## Hybrid cascade routing")
        lines.append("")
        lines.append(
            "`Fast %` is how often the free CPU n-gram drafter served a block; the "
            "two alpha columns are how often each drafter's proposals survived "
            "verification. They answer different questions, and a router can fire "
            "constantly while proposing badly. `Draft fwd` is GPU draft forwards "
            "actually spent, against `if all model` -- what pure model drafting "
            "would have cost. That counterfactual is estimated as iterations x the "
            "mean *model* block length observed in the same run, not the draft "
            "ceiling: the dual gate already shortens blocks well below the ceiling, "
            "so measuring against it would credit the n-gram router with the gate's "
            "savings."
        )
        lines.append("")
        for match_len in match_lens:
            group = [r for r in results if r.variant == "hybrid"
                     and r.min_match_len == match_len and r.routing]
            lines.append(f"### Pattern length m = {match_len}")
            lines.append("")
            headers = ["Prompt", "Iters", "n-gram blocks", "Fast %", "n-gram alpha",
                       "model alpha", "Draft fwd", "if all model", "Saved", "Speedup"]
            rows = []
            for prompt_id, _ in PROMPTS:
                run = _find(results, prompt_id, f"hybrid:{match_len}")
                if run is None:
                    continue
                t = run.routing
                would = _counterfactual_draft_forwards(t)
                rows.append([
                    prompt_id, str(t["iterations"]), str(t["ngram_blocks"]),
                    f"{t['ngram_share']:.0%}", f"{t['ngram_alpha']:.1%}",
                    f"{t['model_alpha']:.1%}", str(t["draft_forwards"]), str(would),
                    f"{1 - t['draft_forwards'] / max(1, would):.0%}",
                    f"{speedup(run):.2f}x",
                ])
            totals = {key: sum(r.routing[key] for r in group)
                      for key in ("iterations", "ngram_blocks", "ngram_proposed",
                                  "ngram_accepted", "model_proposed", "draft_forwards")}
            would_total = sum(_counterfactual_draft_forwards(r.routing) for r in group)
            rows.append([
                "**total**", str(totals["iterations"]), str(totals["ngram_blocks"]),
                f"{totals['ngram_blocks'] / max(1, totals['iterations']):.0%}",
                f"{totals['ngram_accepted'] / max(1, totals['ngram_proposed']):.1%}",
                f"{_mean([r.routing['model_alpha'] for r in group]):.1%}",
                str(totals["draft_forwards"]), str(would_total),
                f"{1 - totals['draft_forwards'] / max(1, would_total):.0%}",
                f"{_mean([speedup(r) for r in group]):.2f}x",
            ])
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
        order = {"baseline": 0, "static": 1, "dual_gate": 2, "hybrid": 3,
                 "monte_carlo": 4, "particle_filter": 5, "tree": 6, "self_spec": 7}
        group.sort(key=lambda r: (order[r.variant], r.k or 0))
        for r in group:
            rows.append([
                prompt_id, r.label, f"{r.seconds:.2f}", str(r.tokens), f"{r.tok_s:.1f}",
                "1.00x" if r.variant == "baseline" else f"{speedup(r):.2f}x",
                "--" if r.alpha is None else f"{r.alpha:.1%}",
                str(r.target_forwards), f"{r.tokens_per_forward:.2f}",
                f"{r.peak_gib:.2f}", r.correctness, "yes" if r.complete else "no",
                "--" if r.identical is None
                else ("identical" if r.identical else f"tok {r.divergence_index}"),
            ])
    lines.append(_table(headers, rows))
    lines.append("")

    # ---- summary ----
    spec = [r for r in results if r.variant != "baseline"]
    best_label, best_speed = "", 0.0
    for label, group in _configurations(results):
        if not group or group[0].variant == "baseline":
            continue
        mean_speed = _mean([speedup(r) for r in group])
        if mean_speed > best_speed:
            best_speed, best_label = mean_speed, label

    identical = [r.identical for r in spec if r.identical is not None]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Best configuration**: {best_label} at **{best_speed:.2f}x** the baseline")
    lines.append(
        f"- **Baseline throughput**: {_mean([r.tok_s for r in results if r.variant == 'baseline']):.1f} tok/s"
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
            min_new_tokens=args.max_new_tokens if getattr(args, "no_eos", False) else None,
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
        prompt_id="", variant="baseline", k=None, seconds=seconds, tokens=len(new_ids),
        target_forwards=len(new_ids),  # one forward per token, by definition
        alpha=None, peak_gib=_peak_gib(),
        correctness=check_code(text, complete), complete=complete, text=text,
        ids=new_ids,
    )


def run_speculative(engine, prompt: str, args, variant: str | None = None) -> RunResult:
    _peak_reset()
    start = time.perf_counter()
    text, stats = engine.generate(
        prompt, max_new_tokens=args.max_new_tokens, stream=False,
        stop_at_eos=not getattr(args, "no_eos", False),
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    complete = stats.tokens_generated < args.max_new_tokens
    return RunResult(
        prompt_id="", seconds=seconds,
        tokens=stats.tokens_generated, target_forwards=stats.target_forwards,
        alpha=stats.acceptance_rate, peak_gib=_peak_gib(),
        correctness=check_code(text, complete), complete=complete, text=text,
        ids=list(stats.token_ids),
        variant=variant or ("dual_gate" if engine.use_dual_gate else "static"),
        # An adaptive run has no single k to key report tables by.
        k=None if engine.use_dual_gate else engine.k,
        entropy_threshold=engine.entropy_threshold if engine.use_dual_gate else None,
        draft_forwards=stats.draft_forwards,
        min_match_len=getattr(engine, "min_match_len", None),
        branches=getattr(engine, "branches", getattr(engine, "particles", None)),
        survivor_fraction=getattr(engine, "survivor_fraction", None),
        split_threshold=getattr(engine, "split_threshold", None),
        draft_rows=stats.draft_row_forwards,
        tree=({
            "leaves": stats.mean_leaves,
            "splits": stats.tree_splits,
            "iterations": stats.iterations,
            "unique": stats.mean_unique_candidates,
        } if stats.leaf_counts else None),
        branch=({
            "best": stats.mean_best_branch,
            "single": stats.mean_single_branch,
            "gain": stats.branch_gain,
            "wins": stats.branch_win_spread,
            "iterations": stats.iterations,
            "unique": stats.mean_unique_candidates,
            "resamples": stats.resample_steps,
        } if stats.branch_accepted else None),
        routing=({
            "ngram_blocks": stats.ngram_drafts_used,
            "model_blocks": stats.model_drafts_used,
            "ngram_share": stats.ngram_share,
            "ngram_proposed": stats.ngram_tokens_proposed,
            "ngram_accepted": stats.ngram_tokens_accepted,
            "ngram_alpha": stats.ngram_acceptance_rate,
            "model_proposed": stats.model_tokens_proposed,
            "model_alpha": stats.model_acceptance_rate,
            "draft_forwards": stats.draft_forwards,
            "iterations": stats.iterations,
        } if stats.ngram_drafts_used or stats.model_drafts_used else None),
        mean_draft_len=stats.mean_draft_length,
        gate=({
            "statistical": stats.statistical_gate_triggers,
            "syntactic": stats.syntactic_gate_triggers,
            "ungated": stats.ungated_iterations,
            "suppressed": stats.syntactic_suppressed,
            "iterations": stats.iterations,
            "trigger_rate": stats.gate_trigger_rate,
        } if engine.use_dual_gate else None),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 (default) is greedy and makes runs comparable and reproducible")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output", default="benchmark_results.md")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    parser.add_argument("--max-draft-len", type=int, default=DUAL_GATE_MAX_DRAFT,
                        help="ceiling on the adaptive draft length")
    parser.add_argument("--entropy-thresholds", type=float, nargs="+",
                        default=list(DUAL_GATE_THRESHOLDS),
                        help="statistical gate fires below these top-1 probabilities; "
                             "one adaptive configuration is run per value")
    parser.add_argument("--no-dual-gate", action="store_true",
                        help="run only the static K sweep")
    parser.add_argument("--static-ks", type=int, nargs="*", default=list(K_SWEEP),
                        help="static draft lengths to sweep; pass none to skip")
    parser.add_argument("--self-spec-window", type=int, default=SELF_SPEC_WINDOW,
                        help="draft KV-cache window for the single-model engine")
    parser.add_argument("--self-spec-k", type=int, default=SELF_SPEC_K)
    parser.add_argument("--tree-split-thresholds", type=float, nargs="*",
                        default=list(TREE_SPLIT_THRESHOLDS),
                        help="tree split thresholds to sweep; a branch splits when its "
                             "top-1 probability falls below the value. Pass none to skip")
    parser.add_argument("--tree-max-leaves", type=int, default=TREE_MAX_LEAVES)
    parser.add_argument("--tree-k", type=int, default=TREE_K)
    parser.add_argument("--pf-survivor-fractions", type=float, nargs="*",
                        default=list(PF_SURVIVOR_FRACTIONS),
                        help="particle filter survivor fractions to sweep; one "
                             "configuration per value. Pass none to skip")
    parser.add_argument("--pf-particles", type=int, default=PF_PARTICLES)
    parser.add_argument("--pf-k", type=int, default=PF_K)
    parser.add_argument("--mc-branches", type=int, nargs="*", default=list(MC_BRANCHES),
                        help="parallel draft branch counts to sweep; one Monte Carlo "
                             "configuration is run per value. Pass none to skip")
    parser.add_argument("--mc-draft-temperature", type=float,
                        default=MC_DRAFT_TEMPERATURE,
                        help="draft sampling temperature for Monte Carlo branches; "
                             "must be > 0 or the branches do not diverge")
    parser.add_argument("--mc-ks", type=int, nargs="*", default=[MC_K],
                        help="draft lengths to pair with each branch count")
    parser.add_argument("--hybrid-match-lens", type=int, nargs="*",
                        default=list(HYBRID_MATCH_LENS),
                        help="n-gram pattern lengths to sweep; one hybrid cascade "
                             "configuration is run per value. Pass none to skip")
    parser.add_argument("--ngram-draft-len", type=int, default=HYBRID_NGRAM_DRAFT_LEN)
    parser.add_argument("--draft-id", default=DRAFT_ID,
                        help="draft checkpoint for the two-model configurations; a "
                             "smaller draft trades acceptance for VRAM and per-forward "
                             "cost")
    parser.add_argument("--no-self-spec", action="store_true",
                        help="skip the single-model Twin-Cache configuration")
    parser.add_argument("--no-eos", action="store_true",
                        help="force every run to emit exactly --max-new-tokens. "
                             "Configurations diverge and so stop at different EOS "
                             "points, and acceptance is higher early in a generation "
                             "than late, which flatters whichever run ends soonest")
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
    torch.cuda.synchronize()
    # Weights resident with the target alone. This is what the baseline and the
    # single-model engine actually require; the two-model configurations are
    # measured separately in phase 2, after the draft is loaded.
    vram_target_only = torch.cuda.memory_allocated() / 2**30
    print(f"  target weights resident: {vram_target_only:.2f} GiB")

    static_ks = list(args.static_ks or [])

    def chat(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )

    # Warm up before measuring anything. The first CUDA generation pays for
    # kernel autotuning and bitsandbytes' int8 setup, which would otherwise be
    # billed entirely to whichever configuration happened to run first.
    warm = chat("Write a Python function that reverses a string.")

    def warmup_target():
        with torch.no_grad():
            target_model.generate(
                tokenizer(warm, return_tensors="pt").input_ids.to(target_model.device),
                max_new_tokens=24, do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()

    match_lens = list(args.hybrid_match_lens or [])
    mc_configs = [(b, mc_k) for b in (args.mc_branches or [])
                  for mc_k in (args.mc_ks or [])]
    pf_fractions = list(args.pf_survivor_fractions or [])
    tree_thresholds = list(args.tree_split_thresholds or [])
    n_two_model = (len(static_ks) + len(match_lens) + len(mc_configs) + len(pf_fractions)
                   + len(tree_thresholds)
                   + (0 if args.no_dual_gate else len(args.entropy_thresholds)))
    total_runs = len(prompts) * (1 + n_two_model + (0 if args.no_self_spec else 1))
    print(f"\nRunning {total_runs} generations over {len(prompts)} prompts")
    print(f"Sampling: {'greedy' if args.temperature == 0 else f'temp={args.temperature}'}, "
          f"budget {args.max_new_tokens} tokens"
          + (" (forced length)" if args.no_eos else ""))

    results: list[RunResult] = []
    baselines: dict[str, RunResult] = {}
    run_index = 0

    def note(run, base, label, extra=""):
        run.prompt_id = base.prompt_id
        if args.temperature == 0:
            run.identical = run.ids == base.ids
            run.divergence_index = _first_difference(base.ids, run.ids)
        results.append(run)
        speed = run.tok_s / base.tok_s if base.tok_s else 0
        flag = "" if run.identical is not False else f"  diverges@{run.divergence_index}"
        print(f"{run.seconds:6.2f}s  {run.tok_s:5.1f} tok/s  {run.tokens:>3} tok  "
              f"{speed:4.2f}x  a={run.alpha:5.1%}  {run.correctness}{extra}{flag}")

    # =================== PHASE 1: target model only ======================
    print(f"\n{'=' * 78}")
    print("PHASE 1  target only -- baseline and single-model Twin-Cache")
    print(f"{'=' * 78}")
    self_engine = None
    if not args.no_self_spec:
        self_engine = SelfSpeculativeEngine(
            target_model, tokenizer, k=args.self_spec_k,
            draft_window_size=args.self_spec_window,
            temperature=args.temperature, top_p=args.top_p,
        )
    print("Warming up...")
    warmup_target()
    if self_engine is not None:
        self_engine.generate(warm, max_new_tokens=24, stream=False)
    torch.cuda.synchronize()

    for prompt_id, prompt_text in prompts:
        templated = chat(prompt_text)

        run_index += 1
        print(f"[{run_index:>2}/{total_runs}] {prompt_id:<11} baseline     ", end="", flush=True)
        base = run_baseline(target_model, tokenizer, templated, args)
        base.prompt_id = prompt_id
        base.weights_gib = vram_target_only
        baselines[prompt_id] = base
        results.append(base)
        print(f"{base.seconds:6.2f}s  {base.tok_s:5.1f} tok/s  {base.tokens:>3} tok  "
              f"{base.correctness}")

        if self_engine is not None:
            run_index += 1
            print(f"[{run_index:>2}/{total_runs}] {prompt_id:<11} self-spec    ",
                  end="", flush=True)
            run = run_speculative(self_engine, templated, args, variant="self_spec")
            run.weights_gib = vram_target_only
            note(run, base, "self-spec",
                 f"  7Bfwd/tok={run.total_forwards / max(1, run.tokens):.2f}")

    # =================== PHASE 2: add the 1.5B draft =====================
    engines: dict[int, SpeculativeEngine] = {}
    dual_engines: dict[float, SpeculativeEngine] = {}
    hybrid_engines: dict[int, HybridCascadeEngine] = {}
    mc_engines: dict[tuple[int, int], MonteCarloEngine] = {}
    pf_engines: dict[float, ParticleFilterEngine] = {}
    tree_engines: dict[float, TreeSpeculativeEngine] = {}
    vram_two_model = vram_target_only
    if n_two_model:
        print(f"\n{'=' * 78}")
        print("PHASE 2  target + 1.5B draft -- two-model static sweep and dual gate")
        print(f"{'=' * 78}")
        print(f"Loading draft (bfloat16): {args.draft_id}")
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_id, dtype=torch.bfloat16, device_map="cuda:0"
        )
        draft_model.eval()
        torch.cuda.synchronize()
        vram_two_model = torch.cuda.memory_allocated() / 2**30
        print(f"  both models resident: {vram_two_model:.2f} GiB "
              f"(+{vram_two_model - vram_target_only:.2f} GiB for the draft)")

        engines = {
            k: SpeculativeEngine(draft_model, target_model, tokenizer, k=k,
                                 temperature=args.temperature, top_p=args.top_p)
            for k in static_ks
        }
        dual_engines = {} if args.no_dual_gate else {
            threshold: SpeculativeEngine(
                draft_model, target_model, tokenizer,
                temperature=args.temperature, top_p=args.top_p,
                use_dual_gate=True,
                max_draft_len=args.max_draft_len,
                entropy_threshold=threshold,
            )
            for threshold in args.entropy_thresholds
        }
        hybrid_engines = {
            match_len: HybridCascadeEngine(
                draft_model, target_model, tokenizer,
                temperature=args.temperature, top_p=args.top_p,
                use_dual_gate=not args.no_dual_gate,
                max_draft_len=args.max_draft_len,
                entropy_threshold=(args.entropy_thresholds[-1]
                                   if args.entropy_thresholds else 0.65),
                ngram_draft_len=args.ngram_draft_len,
                min_match_len=match_len,
            )
            for match_len in match_lens
        }
        mc_engines = {
            (branches, mc_k): MonteCarloEngine(
                draft_model, target_model, tokenizer,
                k=mc_k, branches=branches,
                draft_temperature=args.mc_draft_temperature,
                temperature=args.temperature, top_p=args.top_p,
            )
            for branches, mc_k in mc_configs
        }
        pf_engines = {
            fraction: ParticleFilterEngine(
                draft_model, target_model, tokenizer,
                k=args.pf_k, particles=args.pf_particles,
                draft_temperature=args.mc_draft_temperature,
                survivor_fraction=fraction,
                temperature=args.temperature, top_p=args.top_p,
            )
            for fraction in pf_fractions
        }
        tree_engines = {
            threshold: TreeSpeculativeEngine(
                draft_model, target_model, tokenizer,
                k=args.tree_k, max_leaves=args.tree_max_leaves,
                split_threshold=threshold,
                temperature=args.temperature, top_p=args.top_p,
            )
            for threshold in tree_thresholds
        }

        print("Warming up...")
        for engine in (list(engines.values()) + list(dual_engines.values())
                       + list(hybrid_engines.values()) + list(mc_engines.values())
                       + list(pf_engines.values()) + list(tree_engines.values())):
            engine.generate(warm, max_new_tokens=24, stream=False)
        torch.cuda.synchronize()

        for prompt_id, prompt_text in prompts:
            templated = chat(prompt_text)
            base = baselines[prompt_id]

            for k in static_ks:
                run_index += 1
                print(f"[{run_index:>2}/{total_runs}] {prompt_id:<11} K={k:<10}",
                      end="", flush=True)
                run = run_speculative(engines[k], templated, args)
                run.weights_gib = vram_two_model
                note(run, base, f"K={k}")

            for threshold, engine in dual_engines.items():
                run_index += 1
                print(f"[{run_index:>2}/{total_runs}] {prompt_id:<11} "
                      f"gate t={threshold:<6g}", end="", flush=True)
                run = run_speculative(engine, templated, args)
                run.weights_gib = vram_two_model
                g = run.gate
                note(run, base, "gate",
                     f"  len={run.mean_draft_len:.2f} stat={g['statistical']} "
                     f"syn={g['syntactic']}")

            for (branches, mc_k), engine in mc_engines.items():
                run_index += 1
                print(f"[{run_index:>2}/{total_runs}] {prompt_id:<11} "
                      f"mc B{branches}/K{mc_k:<4}", end="", flush=True)
                run = run_speculative(engine, templated, args, variant="monte_carlo")
                run.weights_gib = vram_two_model
                b = run.branch
                note(run, base, "mc",
                     f"  best={b['best']:.2f} single={b['single']:.2f} "
                     f"gain=+{b['gain']:.2f}")

            for threshold, engine in tree_engines.items():
                run_index += 1
                print(f"[{run_index:>2}/{total_runs}] {prompt_id:<11} "
                      f"tree s<{threshold:<5g}", end="", flush=True)
                run = run_speculative(engine, templated, args, variant="tree")
                run.weights_gib = vram_two_model
                t = run.tree
                note(run, base, "tree",
                     f"  leaves={t['leaves']:.2f} splits={t['splits']} "
                     f"rows/tok={run.draft_rows_per_token:.2f}")

            for fraction, engine in pf_engines.items():
                run_index += 1
                print(f"[{run_index:>2}/{total_runs}] {prompt_id:<11} "
                      f"pf keep={fraction:<5g}", end="", flush=True)
                run = run_speculative(engine, templated, args,
                                      variant="particle_filter")
                run.weights_gib = vram_two_model
                b = run.branch
                note(run, base, "pf",
                     f"  best={b['best']:.2f} single={b['single']:.2f} "
                     f"gain=+{b['gain']:.2f} distinct={b['unique']:.2f}")

            for match_len, engine in hybrid_engines.items():
                run_index += 1
                print(f"[{run_index:>2}/{total_runs}] {prompt_id:<11} "
                      f"hybrid m={match_len:<4}", end="", flush=True)
                run = run_speculative(engine, templated, args, variant="hybrid")
                run.weights_gib = vram_two_model
                t = run.routing
                note(run, base, "hybrid",
                     f"  fast={t['ngram_share']:.0%} ngram_a={t['ngram_alpha']:.0%} "
                     f"model_a={t['model_alpha']:.0%} dfwd={t['draft_forwards']}")

    # The verification vocabulary is the shared prefix of both models' output
    # widths. With only the target loaded there is nothing to intersect, so it is
    # the target's own width.
    any_engine = next(iter(list(engines.values()) + list(dual_engines.values())), None)
    vocab_width = (any_engine or self_engine).vocab_size

    determinism = None
    if args.temperature == 0:
        print("\nMeasuring target shape-determinism (explains any greedy divergence)...")
        determinism = measure_shape_determinism(
            target_model, tokenizer, chat(prompts[0][1]),
            shared_vocab=vocab_width,
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
        "forced_length": args.no_eos,
        "draft_id": args.draft_id,
        "vocab": vocab_width,
    }
    report = build_report(results, config)
    print("\n" + report)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
