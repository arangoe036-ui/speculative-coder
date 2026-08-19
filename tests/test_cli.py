"""Tests for the CLI's display logic (no GPU, no model loading).

Only the pure functions are covered: the HUD formatter and the colour wrapper.
The interactive loop itself is exercised by piping a script into `cli.py`.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import Ansi, format_hud  # noqa: E402
from core.engine import GenerationStats  # noqa: E402


def _stats(tokens=200, seconds=8.0, proposed=210, accepted=160, forwards=42):
    return GenerationStats(
        tokens_generated=tokens, seconds=seconds, draft_tokens_proposed=proposed,
        draft_tokens_accepted=accepted, target_forwards=forwards,
    )


PLAIN = Ansi(enabled=False)


def test_hud_has_the_requested_shape():
    """The HUD must match the specified [Alpha | Speedup | Tok/s] format."""
    hud = format_hud(_stats(), baseline_tok_s=12.5, ansi=PLAIN)
    assert hud == "[Alpha: 76% | Speedup: 2.0x | Tok/s: 25.0]"


def test_hud_speedup_is_relative_to_the_calibrated_baseline():
    stats = _stats(tokens=100, seconds=4.0)      # 25 tok/s
    assert "Speedup: 2.5x" in format_hud(stats, baseline_tok_s=10.0, ansi=PLAIN)
    assert "Speedup: 1.0x" in format_hud(stats, baseline_tok_s=25.0, ansi=PLAIN)


def test_hud_falls_back_to_tokens_per_forward_without_calibration():
    """With no baseline there is nothing to divide by, so report the
    hardware-independent figure instead of inventing a speedup."""
    hud = format_hud(_stats(tokens=200, forwards=40), baseline_tok_s=None, ansi=PLAIN)
    assert "Tok/fwd: 5.0" in hud
    assert "Speedup" not in hud


def test_hud_handles_zero_baseline_without_dividing_by_zero():
    hud = format_hud(_stats(), baseline_tok_s=0.0, ansi=PLAIN)
    assert "Tok/fwd" in hud


def test_hud_handles_an_empty_generation():
    """A run that produced nothing must not crash the HUD."""
    hud = format_hud(GenerationStats(), baseline_tok_s=12.5, ansi=PLAIN)
    assert "Alpha: 0%" in hud and "Tok/s: 0.0" in hud


@pytest.mark.parametrize(
    "accepted, proposed, colour_code",
    [(180, 200, "32"), (90, 200, "33"), (20, 200, "31")],  # green / yellow / red
)
def test_hud_colours_alpha_by_quality(accepted, proposed, colour_code):
    """Acceptance rate is the number worth glancing at, so it is colour-coded."""
    stats = _stats(proposed=proposed, accepted=accepted)
    hud = format_hud(stats, baseline_tok_s=12.5, ansi=Ansi(enabled=True))
    assert f"\033[{colour_code}m" in hud


def test_ansi_disabled_emits_no_escape_codes():
    """Piping to a file or a pager must not litter it with escape sequences."""
    hud = format_hud(_stats(), baseline_tok_s=12.5, ansi=PLAIN)
    assert "\033" not in hud
    assert PLAIN.bold("x") == "x" and PLAIN.red("y") == "y"


def test_ansi_enabled_wraps_and_resets():
    enabled = Ansi(enabled=True)
    assert enabled.green("ok") == "\033[32mok\033[0m"
