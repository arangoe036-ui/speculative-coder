"""Tests for the benchmark's reporting and code-checking logic.

These run on CPU with synthetic results.  The point is that a bug in the report
builder should not cost a full GPU sweep to discover, and that the correctness
classifier is doing real work rather than rubber-stamping everything PASS.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import (  # noqa: E402
    K_SWEEP,
    PROMPTS,
    RunResult,
    build_report,
    check_code,
    extract_code,
)

CONFIG = {
    "gpu": "Test GPU", "vram_total": 16.0, "sampling": "greedy (temperature 0)",
    "max_new_tokens": 384, "vocab": 151936,
}


# ---------------------------------------------------------------- prompts
def test_prompt_set_is_ten_and_distinct():
    assert len(PROMPTS) == 10
    ids = [p[0] for p in PROMPTS]
    assert len(set(ids)) == 10, "prompt ids must be unique to key the tables"
    assert all(text.strip().endswith((".", "?")) for _, text in PROMPTS)


# ---------------------------------------------------------------- code checks
def test_extract_code_prefers_fenced_blocks():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\nHope that helps."
    assert extract_code(text) == "def f():\n    return 1"


def test_extract_code_joins_multiple_blocks():
    text = "```python\nimport os\n```\nand\n```py\nx = 1\n```"
    assert extract_code(text) == "import os\n\nx = 1"


def test_extract_code_handles_unterminated_fence():
    """A generation cut off by the budget leaves the fence open."""
    assert extract_code("```python\ndef f():\n    return 1\n") == "def f():\n    return 1"


def test_extract_code_falls_back_to_raw_text():
    assert extract_code("x = 1") == "x = 1"


def test_check_code_passes_valid_python():
    text = "```python\ndef quick_sort(a):\n    return sorted(a)\n```"
    assert check_code(text, complete=True) == "PASS"


def test_check_code_fails_broken_python():
    text = "```python\ndef f(:\n    return\n```"
    assert check_code(text, complete=True) == "FAIL"


def test_check_code_reports_truncation_separately():
    """Budget-truncated code must not be scored as a model failure.

    Conflating "ran out of tokens" with "wrote invalid code" would make the
    correctness column meaningless at small budgets.
    """
    text = "```python\ndef f():\n    if x:\n"
    assert check_code(text, complete=False) == "TRUNC"
    assert check_code("", complete=False) == "TRUNC"


def test_check_code_tolerates_trailing_incomplete_line():
    """Complete responses sometimes close the fence after a dangling statement."""
    text = "```python\ndef f():\n    return 1\n\nresult = f(\n```"
    assert check_code(text, complete=True) == "PASS"


def test_check_code_does_not_execute_the_code():
    """ast.parse must be used, not exec: model-authored code is never run."""
    text = "```python\nraise SystemExit('boom')\nopen('/nonexistent/x','w')\n```"
    assert check_code(text, complete=True) == "PASS"


# ---------------------------------------------------------------- report
def _synthetic_results() -> list[RunResult]:
    results = []
    for prompt_id, _ in PROMPTS:
        results.append(RunResult(
            prompt_id=prompt_id, variant="baseline", k=None, seconds=10.0, tokens=200,
            target_forwards=200, alpha=None, peak_gib=11.0, correctness="PASS",
            complete=True, text="same",
        ))
        for k in K_SWEEP:
            results.append(RunResult(
                prompt_id=prompt_id, variant="static", k=k, seconds=10.0 / (1 + k / 10),
                tokens=200, target_forwards=200 // max(1, k), alpha=0.5 + k / 100,
                peak_gib=11.1, correctness="PASS", complete=True, identical=True,
                text="same",
            ))
    return results


def test_build_report_is_wellformed_markdown():
    report = build_report(_synthetic_results(), CONFIG)

    for heading in ("# Speculative Decoding Benchmark", "## Aggregate results",
                    "## Speedup by prompt", "## Acceptance rate (alpha) by prompt",
                    "## Full results", "## Summary"):
        assert heading in report, f"missing section: {heading}"

    # Every prompt and configuration must appear.
    for prompt_id, _ in PROMPTS:
        assert prompt_id in report
    for k in K_SWEEP:
        assert f"Speculative K={k}" in report

    # Table rows must all have a consistent column count within each table.
    for block in report.split("\n\n"):
        rows = [line for line in block.splitlines() if line.startswith("|")]
        if len(rows) < 2:
            continue
        widths = {row.count("|") for row in rows}
        assert len(widths) == 1, f"ragged table:\n{block}"


def test_aggregate_table_includes_every_configuration_row():
    """Each configuration must actually appear as a row, baseline included.

    Added after a real miss: `variant` and `method` briefly both encoded the
    baseline/spec distinction, and a baseline built without `variant` was
    classified as a static-K run. That silently dropped the baseline row from the
    aggregate table while every existing test still passed, because they only
    checked that section *headings* were present.
    """
    report = build_report(_synthetic_results(), CONFIG)
    aggregate = report.split("## Speedup by prompt")[0]
    assert "| Baseline (7B only)" in aggregate, "baseline row missing"
    for k in K_SWEEP:
        assert f"| Speculative K={k}" in aggregate, f"K={k} row missing"


def _with_dual_gate(results, threshold=0.35):
    """Append a dual-gate run per prompt to a synthetic result set."""
    for prompt_id, _ in PROMPTS:
        results.append(RunResult(
            prompt_id=prompt_id, variant="dual_gate", k=None, seconds=8.0,
            tokens=200, target_forwards=34, alpha=0.72, peak_gib=11.1,
            correctness="PASS", complete=True, identical=True, text="same",
            mean_draft_len=5.9, entropy_threshold=threshold,
            gate={"statistical": 7, "syntactic": 2, "ungated": 25,
                  "suppressed": 3, "iterations": 34, "trigger_rate": 9 / 34},
        ))
    return results


def test_dual_gate_appears_in_every_table():
    report = build_report(_with_dual_gate(_synthetic_results()), CONFIG)
    assert "Adaptive Dual-Gate (Max K=8, t=0.35)" in report
    # Per-prompt tables use the compact header.
    assert "Gate t=0.35" in report.split("## Speedup by prompt")[1]
    assert "## Dual-gate telemetry" in report


def test_dual_gate_telemetry_reports_counts_and_a_total():
    report = build_report(_with_dual_gate(_synthetic_results()), CONFIG)
    section = report.split("## Dual-gate telemetry")[1].split("## Full results")[0]
    assert "Statistical" in section and "Syntactic" in section
    assert "Suppressed" in section
    assert "**total**" in section, "a totals row makes the section readable at a glance"
    # 10 prompts x 7 statistical triggers, and x 34 iterations.
    assert "70" in section and "340" in section


def test_dual_gate_can_win_best_configuration():
    """The winner is whichever configuration is fastest, not hardcoded to a K."""
    results = _synthetic_results()
    for prompt_id, _ in PROMPTS:
        results.append(RunResult(
            prompt_id=prompt_id, variant="dual_gate", k=None, seconds=2.0,
            tokens=200, target_forwards=30, alpha=0.8, peak_gib=11.1,
            correctness="PASS", complete=True, text="same", mean_draft_len=6.0,
            entropy_threshold=0.65,
            gate={"statistical": 1, "syntactic": 0, "ungated": 29,
                  "suppressed": 0, "iterations": 30, "trigger_rate": 1 / 30},
        ))
    report = build_report(results, CONFIG)
    summary = report.split("## Summary")[1]
    assert "Adaptive Dual-Gate" in summary, "fastest configuration should win"


def test_report_omits_dual_gate_sections_when_not_run():
    """--no-dual-gate must not leave empty scaffolding behind."""
    report = build_report(_synthetic_results(), CONFIG)
    assert "## Dual-gate telemetry" not in report
    assert "Dual-Gate" not in report and "Gate t=" not in report


def test_multiple_thresholds_each_get_their_own_row_and_section():
    """Several adaptive configurations must stay distinguishable in every table.

    They share a baseline and static sweep within one run, so the only thing
    separating them in the report is the threshold.
    """
    results = _with_dual_gate(_synthetic_results(), threshold=0.35)
    results = _with_dual_gate(results, threshold=0.65)
    report = build_report(results, CONFIG)

    for threshold in ("0.35", "0.65"):
        assert f"Adaptive Dual-Gate (Max K=8, t={threshold})" in report
        assert f"Gate t={threshold}" in report
        assert f"### Threshold t = {threshold}" in report

    # Two adaptive rows in the aggregate table, not one collapsed row.
    aggregate = report.split("## Speedup by prompt")[0]
    assert aggregate.count("Adaptive Dual-Gate") == 2


def test_build_report_computes_speedup_against_matching_prompt():
    """Speedup must be per-prompt, not against a global mean.

    A slow prompt and a fast prompt in the same sweep would otherwise report
    wildly wrong per-row speedups.
    """
    results = [
        RunResult("slow", "baseline", None, 20.0, 200, 200, None, 11.0, "PASS", True, text="x"),
        RunResult("fast", "baseline", None, 5.0, 200, 200, None, 11.0, "PASS", True, text="x"),
        # Each is exactly 2x its own baseline.
        RunResult("slow", "static", 5, 10.0, 200, 40, 0.8, 11.1, "PASS", True, True, "x"),
        RunResult("fast", "static", 5, 2.5, 200, 40, 0.8, 11.1, "PASS", True, True, "x"),
    ]
    report = build_report(results, CONFIG)
    assert "2.00x" in report
    assert "4.00x" not in report and "0.50x" not in report


def test_build_report_locates_and_explains_divergence():
    """A greedy run that differs from the baseline must be located and attributed.

    A bare "not identical" flag reads as an engine bug.  The report has to say
    *where* it diverged and present the measured quantization noise that caused
    it, otherwise the artifact misrepresents its own correctness.
    """
    results = _synthetic_results()
    results[1].identical = False
    results[1].divergence_index = 17
    config = dict(CONFIG, determinism={
        "positions": 64, "region": "generated", "max_logit_delta": 0.223,
        "mean_logit_delta": 0.061, "argmax_flip_rate": 0.042, "argmax_flips": 3,
        "median_top2_margin": 0.35, "min_top2_margin": 0.01,
        "padding_argmax_count": 0,
    })
    report = build_report(results, config)

    assert "39/40 greedy runs" in report
    assert "tok 17" in report, "divergence point must be located in the table"
    assert "## Why greedy output diverges from the baseline" in report
    # The measured cause must appear, not just a claim about it.
    assert "0.223" in report and "0.350" in report
    assert "4.2%" in report
    assert "earliest 17" in report


def test_build_report_omits_divergence_section_when_all_identical():
    """No divergence, no section: the artifact should not raise a doubt it then
    has to walk back."""
    report = build_report(_synthetic_results(), CONFIG)
    assert "## Why greedy output diverges" not in report
    assert "40/40 greedy runs" in report


def test_build_report_survives_missing_determinism_data():
    """--temperature > 0 skips the diagnostic; the report must still render."""
    results = _synthetic_results()
    results[1].identical = False
    results[1].divergence_index = 3
    report = build_report(results, CONFIG)   # no "determinism" key
    assert "## Why greedy output diverges from the baseline" in report
    assert "earliest 3" in report


def test_run_result_derived_metrics():
    r = RunResult("x", "static", 5, 4.0, 200, 40, 0.75, 11.0, "PASS", True)
    assert r.tok_s == pytest.approx(50.0)
    assert r.tokens_per_forward == pytest.approx(5.0)
    assert r.label == "Speculative K=5"
    base = RunResult("x", "baseline", None, 4.0, 200, 200, None, 11.0, "PASS", True)
    assert base.tokens_per_forward == pytest.approx(1.0), "baseline is 1 token per forward"
    assert base.label == "Baseline (7B only)"
