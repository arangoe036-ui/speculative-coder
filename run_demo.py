"""End-to-end demo: Qwen2.5-Coder 1.5B drafting for Qwen2.5-Coder 7B.

    python run_demo.py
    python run_demo.py --k 7 --max-new-tokens 400 --temperature 0.0

Draft : Qwen/Qwen2.5-Coder-1.5B-Instruct  in bfloat16
Target: Qwen/Qwen2.5-Coder-7B-Instruct    in 8-bit (LLM.int8 via bitsandbytes)

Model loading lives here, not in the engine, so the engine stays testable
without a GPU.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core.engine import SpeculativeEngine

DRAFT_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TARGET_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
PROMPT = "Write a quick sort algorithm in Python."


def vram(label: str) -> None:
    """Print allocated / reserved / total VRAM, plus what other processes hold."""
    if not torch.cuda.is_available():
        print(f"[vram] {label}: no CUDA device")
        return
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"[vram] {label:<24} allocated {allocated:5.2f} GiB | reserved {reserved:5.2f} GiB "
        f"| peak {peak:5.2f} GiB | free {free / 2**30:5.2f} of {total / 2**30:5.2f} GiB"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="draft length (gamma)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument(
        "--device-map",
        default="cuda:0",
        help="'cuda:0' pins everything to the GPU and OOMs honestly if it does "
        "not fit; 'auto' lets accelerate spill layers to CPU (slow, but it runs).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also time plain autoregressive decoding on the target for comparison",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this demo needs a GPU.")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram("startup")

    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)

    print(f"\nLoading target (8-bit): {TARGET_ID}")
    t0 = time.perf_counter()
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map=args.device_map,
        # LLM.int8 does its matmuls in fp16. Left at the checkpoint's bfloat16,
        # bitsandbytes casts every single matmul input bf16 -> fp16 and warns
        # about it each time, which both costs time and floods the stream we are
        # trying to watch. Asking for fp16 up front removes the cast entirely.
        dtype=torch.float16,
    )
    target_model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    vram("after target")

    print(f"\nLoading draft (bfloat16): {DRAFT_ID}")
    t0 = time.perf_counter()
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_ID,
        dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    draft_model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    vram("after draft")

    # These two checkpoints pad their embedding matrices to different widths
    # (151936 vs 152064) despite sharing a tokenizer, so report what the engine
    # settled on rather than assuming they agree.
    d_vocab = draft_model.get_output_embeddings().weight.shape[0]
    t_vocab = target_model.get_output_embeddings().weight.shape[0]
    print(f"\nDraft vocab {d_vocab} | target vocab {t_vocab} | tokenizer {len(tokenizer)}")

    engine = SpeculativeEngine(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"Verifying over the shared prefix: {engine.vocab_size} tokens")
    assert engine.vocab_size >= len(tokenizer), (
        "shared vocab is narrower than the tokenizer; real token ids would be "
        "truncated away"
    )

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    print(f"\n{'=' * 76}\nPROMPT: {args.prompt}\n{'=' * 76}")
    text, stats = engine.generate(
        prompt, max_new_tokens=args.max_new_tokens, stream=True
    )
    print(f"{'=' * 76}")
    print(stats.summary())
    print(f"accepted per iteration: {stats.accepted_per_iteration}")
    vram("after generation")

    if args.baseline:
        print(f"\n{'=' * 76}\nBaseline: plain autoregressive decoding on the 7B target")
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(engine.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = target_model.generate(
                ids,
                max_new_tokens=stats.tokens_generated,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n = out.shape[1] - ids.shape[1]
        print(f"{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s)")
        print(f"speculative was {(n / elapsed and stats.tokens_per_second / (n / elapsed)):.2f}x")

    print(f"\nGenerated {len(text)} characters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
