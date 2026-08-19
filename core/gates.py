"""Draft-phase short-circuit gates for adaptive speculative decoding.

With a fixed K the draft model always proposes K tokens, even when it has
visibly lost the thread. Every token after the first rejection is discarded, so
that work is pure waste. These gates stop drafting as soon as the draft looks
likely to be rejected, trading a shorter speculation window for less wasted
draft compute.

Why this cannot affect output correctness
----------------------------------------
`verify_tokens` is exact for *any* block length, and the emitted distribution
does not depend on how the length was chosen -- provided the choice is a stopping
rule over the *draft's* own state and never consults the target's distributions.
Both gates here satisfy that: they look only at the draft model's confidence and
at the text the draft has produced.

That independence is what licenses an otherwise odd-looking detail: the
confidence gate reads the draft's *raw* softmax while verification uses the
temperature-warped ``q``. If the gate used the warped distribution it would be
useless at temperature 0, where ``q`` is one-hot and the top-1 probability is
always exactly 1.0. The gate is a scheduling heuristic, not part of the sampling
rule, so it is free to look at the unwarped distribution.

Both gates are therefore allowed to be wrong. A false trigger costs a little
throughput; it can never change what the model emits.
"""

from __future__ import annotations

_PAIRS = {")": "(", "]": "[", "}": "{"}
_OPENERS = frozenset("([{")
_CLOSERS = frozenset(")]}")
_QUOTES = frozenset("\"'")
# Characters whose meaning depends on the next one or two characters, and which
# therefore cannot be interpreted at the very end of a chunk.
_NEEDS_LOOKAHEAD = frozenset("\"'`")


class SyntaxState:
    """Incremental bracket/quote tracker over generated source text.

    Detects states from which no continuation can recover -- closing a bracket
    that was never opened, or closing one against a mismatched opener. Unclosed
    brackets are *not* fatal: more tokens may still close them.

    Deliberately a character scanner rather than a parser. It runs on every
    drafted token inside the decode loop, so it has to cost microseconds; a real
    parse of a partial file would cost more than the draft token it is judging.

    Three properties make the approximation safe in this setting:

    * String and comment bodies are opaque, so brackets inside them are ignored.
      That is what keeps regex literals like ``r"[\\w.]+"`` from tripping it.
    * Fatality is only reported inside ``` fenced code, because prose has its own
      bracket conventions -- enumerations like "1)" and unmatched parentheses in
      English are common and are not syntax errors. Suppressed detections are
      counted so the cost of that decision is measurable rather than assumed.
    * Being wrong is cheap. See the module docstring.
    """

    __slots__ = ("stack", "delim", "escaped", "in_comment", "in_fence",
                 "fatal", "reason", "suppressed", "_carry")

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.delim: str = ""        # active string delimiter; "" means in code
        self.escaped: bool = False
        self.in_comment: bool = False
        self.in_fence: bool = False
        self.fatal: bool = False
        self.reason: str = ""
        self.suppressed: int = 0    # fatal-looking events seen outside code
        self._carry: str = ""       # held-back tail awaiting lookahead

    def copy(self) -> SyntaxState:
        """Cheap clone, so a draft block can be tracked speculatively.

        The draft may be rejected, so it must not mutate the committed state.
        """
        clone = SyntaxState.__new__(SyntaxState)
        clone.stack = self.stack.copy()
        clone.delim = self.delim
        clone.escaped = self.escaped
        clone.in_comment = self.in_comment
        clone.in_fence = self.in_fence
        clone.fatal = self.fatal
        clone.reason = self.reason
        clone.suppressed = self.suppressed
        clone._carry = self._carry
        return clone

    def feed(self, text: str) -> bool:
        """Consume ``text``; return True if the state is now fatally broken.

        Trailing quote and backtick characters are held back until the next call,
        since a single quote and a tripled one mean different things and a token
        boundary can fall between them. Detection therefore lags by at most one
        token, which for a heuristic gate is irrelevant -- and holding back only
        these characters keeps the common case lag-free.
        """
        if self.fatal:
            return True
        if not text:
            return False

        buf = self._carry + text
        n = len(buf)

        hold = 0
        while hold < 2 and buf[n - hold - 1] in _NEEDS_LOOKAHEAD:
            hold += 1
            if hold >= n:
                break
        limit = n - hold

        i = 0
        while i < limit:
            char = buf[i]

            if self.delim:                     # inside a string literal
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == self.delim[0] and buf.startswith(self.delim, i):
                    i += len(self.delim)
                    self.delim = ""
                    continue
                elif char == "\n" and len(self.delim) == 1:
                    # A single-quoted string cannot span a newline. Ending it
                    # here stops one stray apostrophe in prose ("don't") from
                    # swallowing the rest of the response.
                    self.delim = ""
                i += 1
                continue

            if self.in_comment:
                if char == "\n":
                    self.in_comment = False
                i += 1
                continue

            if char == "`" and buf.startswith("```", i):
                # Prose and code are independent bracket universes; carrying a
                # stack across the boundary would manufacture false fatals.
                self.in_fence = not self.in_fence
                self.stack.clear()
                self.delim = ""
                self.in_comment = False
                i += 3
                continue

            if char == "#":
                self.in_comment = True
            elif char in _QUOTES:
                triple = char * 3
                if buf.startswith(triple, i):
                    self.delim = triple
                    i += 3
                    continue
                self.delim = char
            elif char in _OPENERS:
                self.stack.append(char)
            elif char in _CLOSERS:
                if not self.stack:
                    if self.in_fence:
                        self.fatal = True
                        self.reason = f"closed {char!r} that was never opened"
                        self._carry = ""
                        return True
                    self.suppressed += 1
                elif self.stack[-1] != _PAIRS[char]:
                    opener = self.stack[-1]
                    if self.in_fence:
                        self.fatal = True
                        self.reason = f"closed {char!r} against open {opener!r}"
                        self._carry = ""
                        return True
                    self.suppressed += 1
                    self.stack.pop()
                else:
                    self.stack.pop()
            i += 1

        self._carry = buf[limit:]
        return False

    @property
    def depth(self) -> int:
        """Number of currently unclosed brackets."""
        return len(self.stack)

    def __repr__(self) -> str:
        where = "code" if self.in_fence else "prose"
        state = "fatal" if self.fatal else where
        return (f"SyntaxState({state}, depth={self.depth}, "
                f"delim={self.delim!r}, suppressed={self.suppressed})")
