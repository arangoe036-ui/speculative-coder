"""Tests for the syntactic short-circuit gate.

Two things matter here. Real unrecoverable states must be caught, and the
constructs that actually appear in generated code -- regex literals, f-strings,
docstrings, comments -- must not trip it. A false trigger only costs throughput,
but a gate that fires constantly would shorten every draft block to 1 and undo
the whole point of adaptive drafting, so the false-positive cases are tested as
carefully as the true ones.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gates import SyntaxState  # noqa: E402

FENCE = "```python\n"


def feed_code(text: str) -> SyntaxState:
    """Track `text` as if it were inside a fenced code block."""
    state = SyntaxState()
    state.feed(FENCE)
    state.feed(text)
    return state


def feed_chunked(text: str, size: int = 3) -> SyntaxState:
    """Same, but delivered in small chunks, as streamed tokens arrive."""
    state = SyntaxState()
    state.feed(FENCE)
    for i in range(0, len(text), size):
        state.feed(text[i:i + size])
    return state


# ---------------------------------------------------------------- true positives
@pytest.mark.parametrize("code, why", [
    ("def f():\n    return 1)\n", "closes a paren that was never opened"),
    ("x = 1]\n", "closes a bracket that was never opened"),
    ("y = 2}\n", "closes a brace that was never opened"),
    ("foo(bar]\n", "mismatched: ] against ("),
    ("data[key)\n", "mismatched: ) against ["),
    ("d = {'a': 1)\n", "mismatched: ) against {"),
    ("def f(a, b)):\n", "one paren too many"),
])
def test_detects_unrecoverable_states(code, why):
    assert feed_code(code).fatal, f"should be fatal: {why}"


def test_reports_a_reason():
    state = feed_code("return 1)\n")
    assert state.fatal and "never opened" in state.reason


def test_fatal_is_terminal():
    """Once broken, later well-formed text must not clear the verdict."""
    state = feed_code("return 1)\n")
    assert state.feed("def g():\n    return 2\n") is True
    assert state.fatal


def test_detection_survives_token_sized_chunks():
    """Streamed tokens arrive a few characters at a time."""
    assert feed_chunked("def f():\n    return 1)\n", size=2).fatal
    assert feed_chunked("foo(bar]\n", size=1).fatal


# ---------------------------------------------------------------- true negatives
@pytest.mark.parametrize("code", [
    "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "left = [x for x in arr if x < pivot]\n",
    "d = {'a': [1, 2], 'b': (3, 4)}\n",
    "print(f'{x[0]} and {d[\"k\"]}')\n",
    "nested = f(g(h(1)))\n",
])
def test_accepts_wellformed_code(code):
    state = feed_code(code)
    assert not state.fatal, f"false positive on:\n{code}"


def test_unclosed_brackets_are_not_fatal():
    """A partial draft is the normal case, not an error.

    This is the single most important negative: every draft block ends
    mid-expression, and treating that as fatal would fire the gate on
    essentially every iteration.
    """
    for partial in ("def f(", "x = [1, 2,", "d = {'a':", "foo(bar(baz"):
        state = feed_code(partial)
        assert not state.fatal, f"unclosed should be recoverable: {partial!r}"
        assert state.depth > 0


@pytest.mark.parametrize("code", [
    r'pattern = r"[\w.-]+@[\w.-]+\.\w+"' + "\n",
    r'''re.findall(r"[^\]\[)(]+", text)''' + "\n",
    r'''s = "closing brace: }"''' + "\n",
    r"""t = 'unbalanced ( inside a string'""" + "\n",
])
def test_brackets_inside_strings_are_opaque(code):
    """Regex character classes are the obvious trap: `[^\\]\\[)(]` is balanced
    Python but wildly unbalanced as raw text."""
    assert not feed_code(code).fatal, f"false positive on:\n{code}"


def test_escapes_do_not_end_strings_early():
    assert not feed_code('s = "a \\" ) b"\n').fatal


def test_comments_are_opaque():
    assert not feed_code("x = 1  # closes nothing: )]}\n").fatal
    # ...but code after the comment is still tracked.
    assert feed_code("x = 1  # note )\ny = 2)\n").fatal


def test_docstrings_are_opaque():
    code = '''def f():
    """Return a tuple ) with a stray bracket ]."""
    return 1
'''
    assert not feed_code(code).fatal


def test_triple_quotes_split_across_chunks():
    """A ``\"\"\"`` delimiter can straddle a token boundary."""
    state = SyntaxState()
    state.feed(FENCE)
    for chunk in ('def f():\n    "', '"', '"stray ) here', '"', '""\n    return 1\n'):
        state.feed(chunk)
    assert not state.fatal
    state2 = feed_chunked('def f():\n    """stray ) here"""\n    return 1\n', size=1)
    assert not state2.fatal


def test_single_quote_string_ends_at_newline():
    """An apostrophe in prose must not swallow the rest of the response.

    Without this, "don't" would open a string that never closes, hiding every
    later bracket and disabling the gate entirely.
    """
    state = SyntaxState()
    state.feed("Here's the thing\n")
    state.feed(FENCE)
    state.feed("x = (1, 2)\n")
    assert not state.fatal
    assert state.depth == 0, "brackets after the apostrophe must still be tracked"


# ---------------------------------------------------------------- fence handling
def test_prose_brackets_do_not_trigger_the_gate():
    """Enumerations like "1)" are ordinary prose, not syntax errors."""
    state = SyntaxState()
    state.feed("This works as follows:\n1) pick a pivot\n2) partition\n")
    assert not state.fatal
    assert state.suppressed == 2, "prose detections should be counted, not ignored"


def test_prose_suppression_is_counted_not_silent():
    state = SyntaxState()
    state.feed("a smiley :) and a stray ]\n")
    assert not state.fatal
    assert state.suppressed == 2


def test_fence_transitions_reset_bracket_state():
    """Prose and code must not share a bracket stack."""
    state = SyntaxState()
    state.feed("Consider f(x  (unclosed in prose\n")
    state.feed(FENCE)
    assert state.depth == 0, "entering code must not inherit prose brackets"
    state.feed("return 1)\n")
    assert state.fatal, "and the fresh stack makes this stray close fatal"


def test_code_after_fence_close_is_prose_again():
    state = SyntaxState()
    state.feed(FENCE)
    state.feed("x = (1)\n")
    state.feed("```\n")
    assert not state.in_fence
    state.feed("Note that step 1) is optional.\n")
    assert not state.fatal and state.suppressed == 1


def test_backticks_split_across_chunks():
    state = SyntaxState()
    for chunk in ("``", "`py", "thon\n", "return 1)\n"):
        state.feed(chunk)
    assert state.fatal, "fence must be recognised even when split"


# ---------------------------------------------------------------- copy semantics
def test_copy_is_independent():
    """A speculative draft must not mutate the committed state."""
    committed = SyntaxState()
    committed.feed(FENCE)
    committed.feed("def f(")

    speculative = committed.copy()
    speculative.feed("x, y]\n")            # draft goes off the rails
    assert speculative.fatal
    assert not committed.fatal, "committed state must be untouched"
    assert committed.depth == 1

    # The committed state can still accept a good continuation.
    assert committed.feed("x, y):\n    return x\n") is False
    assert not committed.fatal


def test_copy_preserves_all_state():
    original = SyntaxState()
    original.feed(FENCE)
    original.feed('s = "abc\\')            # mid-string, mid-escape
    clone = original.copy()
    for slot in ("stack", "delim", "escaped", "in_comment", "in_fence",
                 "fatal", "reason", "suppressed"):
        assert getattr(clone, slot) == getattr(original, slot), slot


def test_empty_feed_is_a_noop():
    state = SyntaxState()
    assert state.feed("") is False
    assert not state.fatal and state.depth == 0


def test_repr_is_informative():
    state = feed_code("def f(")
    assert "depth=1" in repr(state)
