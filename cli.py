"""Interactive terminal front-end for the speculative decoding engine.

    python cli.py
    python cli.py --k 7 --temperature 0.0
    echo "Write a bubble sort." | python cli.py    # piped input works too

Type a coding prompt and watch the tokens stream. After each response a HUD
reports acceptance rate, speedup and throughput:

    [Alpha: 78% | Speedup: 2.1x | Tok/s: 28.4]

The speedup figure is measured, not assumed. On startup the CLI times a short
autoregressive generation on the target alone and uses that as the reference, so
the number means "faster than this same 7B model decoding normally on this
machine". Without that calibration the honest thing to display would be tokens
per target forward pass, which is what --no-calibrate falls back to.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core.engine import SpeculativeEngine

DRAFT_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TARGET_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
CALIBRATION_PROMPT = "Write a Python function that reverses a string."
CALIBRATION_TOKENS = 48

HELP = """
Commands:
  /k <n>        set draft length K (currently {k})
  /temp <x>     set temperature (currently {temp})
  /tokens <n>   set max new tokens (currently {tokens})
  /stats        show the last response's full statistics
  /help         show this message
  /quit         exit
Anything else is sent to the model as a prompt.
"""


class Ansi:
    """Terminal colours, disabled automatically when stdout is not a tty."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, t: str) -> str:
        return self._wrap("2", t)

    def bold(self, t: str) -> str:
        return self._wrap("1", t)

    def cyan(self, t: str) -> str:
        return self._wrap("36", t)

    def green(self, t: str) -> str:
        return self._wrap("32", t)

    def yellow(self, t: str) -> str:
        return self._wrap("33", t)

    def red(self, t: str) -> str:
        return self._wrap("31", t)


def format_hud(stats, baseline_tok_s: float | None, ansi: Ansi) -> str:
    """The inline HUD shown after each response.

    Speedup is relative to the calibrated baseline when one exists. Falling back
    to tokens-per-forward when it does not is deliberate: inventing a speedup
    number with nothing to compare against would be the wrong kind of confident.
    """
    alpha = f"Alpha: {stats.acceptance_rate:.0%}"
    if baseline_tok_s:
        speed = f"Speedup: {stats.tokens_per_second / baseline_tok_s:.1f}x"
    else:
        speed = f"Tok/fwd: {stats.speedup_vs_autoregressive:.1f}"
    rate = f"Tok/s: {stats.tokens_per_second:.1f}"

    colour = ansi.green if stats.acceptance_rate >= 0.6 else (
        ansi.yellow if stats.acceptance_rate >= 0.35 else ansi.red
    )
    return ansi.dim("[") + colour(alpha) + ansi.dim(" | ") + ansi.cyan(speed) \
        + ansi.dim(" | ") + ansi.bold(rate) + ansi.dim("]")


def calibrate_baseline(target_model, tokenizer, ansi: Ansi) -> float:
    """Time plain autoregressive decoding on the target, for the HUD's reference.

    Runs twice and keeps the second: the first generation on a fresh CUDA context
    pays for kernel autotuning and int8 setup, which would inflate the baseline
    and flatter every speedup shown afterwards.
    """
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": CALIBRATION_PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)

    tok_s = 0.0
    for _ in range(2):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = target_model.generate(
                ids, max_new_tokens=CALIBRATION_TOKENS, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tok_s = (out.shape[1] - ids.shape[1]) / elapsed
    return tok_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--no-calibrate", action="store_true",
                        help="skip baseline timing; the HUD shows tokens per target forward")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    ansi = Ansi(sys.stdout.isatty() and not args.no_color)

    if not torch.cuda.is_available():
        print("CUDA is not available; this CLI needs a GPU.")
        return 1

    print(ansi.bold("Speculative Coder"), ansi.dim(f"| {torch.cuda.get_device_name(0)}"))
    print(ansi.dim(f"Loading target (8-bit): {TARGET_ID}"))
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda:0",
        dtype=torch.float16,
    )
    target_model.eval()
    print(ansi.dim(f"Loading draft (bfloat16): {DRAFT_ID}"))
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )
    draft_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)

    engine = SpeculativeEngine(
        draft_model, target_model, tokenizer,
        k=args.k, temperature=args.temperature, top_p=args.top_p,
    )

    baseline_tok_s = None
    if not args.no_calibrate:
        print(ansi.dim("Calibrating autoregressive baseline..."), end="", flush=True)
        baseline_tok_s = calibrate_baseline(target_model, tokenizer, ansi)
        print(ansi.dim(f" {baseline_tok_s:.1f} tok/s"))

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(ansi.dim(
        f"Ready. K={engine.k}, temperature={engine.temperature}, "
        f"VRAM {peak:.2f} GiB. /help for commands."
    ))

    last_stats = None
    max_new_tokens = args.max_new_tokens

    while True:
        try:
            print()
            line = input(ansi.cyan(">>> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            command, _, argument = line.partition(" ")
            command, argument = command.lower(), argument.strip()
            if command in ("/quit", "/exit", "/q"):
                break
            if command == "/help":
                print(ansi.dim(HELP.format(
                    k=engine.k, temp=engine.temperature, tokens=max_new_tokens
                )))
                continue
            if command == "/k":
                try:
                    engine.k = max(1, int(argument))
                    print(ansi.dim(f"K = {engine.k}"))
                except ValueError:
                    print(ansi.red(f"expected an integer, got {argument!r}"))
                continue
            if command == "/temp":
                try:
                    engine.temperature = max(0.0, float(argument))
                    print(ansi.dim(f"temperature = {engine.temperature}"))
                except ValueError:
                    print(ansi.red(f"expected a number, got {argument!r}"))
                continue
            if command == "/tokens":
                try:
                    max_new_tokens = max(1, int(argument))
                    print(ansi.dim(f"max_new_tokens = {max_new_tokens}"))
                except ValueError:
                    print(ansi.red(f"expected an integer, got {argument!r}"))
                continue
            if command == "/stats":
                if last_stats is None:
                    print(ansi.dim("no generation yet"))
                else:
                    print(ansi.dim(last_stats.summary()))
                    print(ansi.dim(f"accepted per iteration: {last_stats.accepted_per_iteration}"))
                continue
            print(ansi.red(f"unknown command {command!r}; try /help"))
            continue

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": line}], tokenize=False, add_generation_prompt=True
        )
        print()
        try:
            _, stats = engine.generate(prompt, max_new_tokens=max_new_tokens, stream=True)
        except KeyboardInterrupt:
            # Interrupting mid-generation should return to the prompt, not unwind
            # the whole session and force a two-minute model reload.
            print(ansi.yellow("\n[interrupted]"))
            continue
        last_stats = stats
        print(format_hud(stats, baseline_tok_s, ansi))

    print(ansi.dim("bye"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
