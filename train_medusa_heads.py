"""Phase 2: train the Medusa heads and measure the per-head accuracy curve.

Phase 1 left exactly one unknown. A top-6k mask permits 5.69x on coverage grounds,
but coverage only says a proposal is *allowed*; whether a head names the right token
ten positions ahead, from a single hidden state that never saw the intervening
tokens, is an empirical question that needs training to answer.

    python train_medusa_heads.py                      # low-rank heads, the default
    python train_medusa_heads.py --head-rank 0        # full linear, as first specified
    python train_medusa_heads.py --max-tokens 800000  # more data

Pipeline
--------
1. Extract ``(h_t, g_{t+i-1})`` pairs from real Python with one teacher-forced
   forward per chunk. ``g`` is the base model's own greedy token, so the heads are
   trained against precisely what acceptance is decided against.
2. Free the base model. Training only touches the cached features, so the 8.1 GiB
   base need not stay resident.
3. Fit the heads, then report per-head top-1 accuracy on two held-out sets: unseen
   library Python (in-domain) and freshly generated chat-style code (the real target
   workload).
4. Convert the accuracy curve into a projected speedup.

Head 1 is the built-in control. It predicts ``g_t`` from ``h_t``, which is exactly
what the base ``lm_head`` computes from the same vector -- a linear function of the
input. So head 1 must reach near-ceiling accuracy; if it does not, the pipeline is
wrong rather than the idea.
"""

from __future__ import annotations

import argparse
import gc
import glob
import io
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmark import PROMPTS, TARGET_ID
from core.ast_medusa import MedusaAST, structural_token_ids
from core.medusa_train import (
    evaluate_heads,
    extract_features,
    projected_speedup,
    train_heads,
)

CORPUS_GLOBS = (
    ".venv/Lib/site-packages/transformers/models/**/*.py",
    ".venv/Lib/site-packages/transformers/*.py",
    ".venv/Lib/site-packages/torch/nn/**/*.py",
)


def corpus_files(seed: int = 0) -> list[str]:
    files: list[str] = []
    for pattern in CORPUS_GLOBS:
        files += glob.glob(pattern, recursive=True)
    files = sorted(set(f for f in files if "__pycache__" not in f))
    random.Random(seed).shuffle(files)
    return files


def read(paths: list[str]) -> list[str]:
    texts = []
    for path in paths:
        try:
            texts.append(io.open(path, encoding="utf-8").read())
        except (OSError, UnicodeDecodeError):
            continue
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-heads", type=int, default=10, help="M")
    parser.add_argument("--top-k", type=int, default=6000,
                        help="mask size: top-K frequent tokens, unioned with AST tokens")
    parser.add_argument("--head-rank", type=int, default=256,
                        help="bottleneck rank; 0 means a plain full linear head")
    parser.add_argument("--max-tokens", type=int, default=400_000)
    parser.add_argument("--eval-tokens", type=int, default=60_000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-files", type=int, default=400)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this needs a GPU.")
        return 1

    head_rank = None if args.head_rank == 0 else args.head_rank
    M = args.num_heads

    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    print(f"Loading base (8-bit, frozen): {TARGET_ID}")
    base = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda:0", dtype=torch.float16,
    ).eval()

    # ---------------------------------------------------------------- mask
    print(f"\n{'=' * 78}\nMASK\n{'=' * 78}")
    files = corpus_files()
    train_files = files[: args.train_files]
    eval_files = files[args.train_files: args.train_files + 60]

    from collections import Counter
    counts: Counter = Counter()
    for text in read(train_files[:200]):
        counts.update(tokenizer(text).input_ids)
    ast_ids = set(structural_token_ids(tokenizer, base.config.vocab_size).tolist())
    allowed = sorted({t for t, _ in counts.most_common(args.top_k)} | ast_ids)
    allowed_tensor = torch.tensor(allowed, dtype=torch.long)
    print(f"  top-{args.top_k} frequent (ranked on training files) + "
          f"{len(ast_ids)} AST = {len(allowed)} outputs")

    medusa = MedusaAST(base, tokenizer, num_heads=M, head_mode="factored",
                       head_rank=head_rank)
    # Replace the AST-derived mask with the frequency-based one measured above.
    medusa.structural_ids = allowed_tensor.to(medusa.structural_ids.device)
    out_features = len(allowed)
    if head_rank is None:
        medusa.medusa_heads = torch.nn.ModuleList(
            torch.nn.Linear(medusa.hidden_size, out_features, bias=False)
            for _ in range(M)
        ).to("cuda:0")
    else:
        medusa.medusa_heads = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(medusa.hidden_size, head_rank, bias=False),
                torch.nn.Linear(head_rank, out_features, bias=False),
            )
            for _ in range(M)
        ).to("cuda:0")
    params = sum(p.numel() for p in medusa.medusa_heads.parameters())
    print(f"  heads: M={M}, rank={head_rank or 'full'} -> {params / 1e6:.1f}M params "
          f"({params * 4 / 2**30:.2f} GiB fp32, {params * 16 / 2**30:.2f} GiB with Adam)")

    # ---------------------------------------------------------------- extract
    print(f"\n{'=' * 78}\nFEATURE EXTRACTION\n{'=' * 78}")
    print(f"  training corpus: {len(train_files)} files, target "
          f"{args.max_tokens:,} tokens")
    hidden, targets, consumed = extract_features(
        base, tokenizer, read(train_files), M, max_tokens=args.max_tokens
    )
    print(f"  train: {hidden.shape[0]:,} examples from {consumed:,} tokens "
          f"({hidden.numel() * 2 / 2**30:.2f} GiB fp16, host RAM)")

    print(f"  held-out library Python: {len(eval_files)} unseen files")
    eval_hidden, eval_targets, eval_consumed = extract_features(
        base, tokenizer, read(eval_files), M, max_tokens=args.eval_tokens,
        progress_every=0,
    )
    print(f"  eval:  {eval_hidden.shape[0]:,} examples from {eval_consumed:,} tokens")

    print("  held-out generated code: the 10 benchmark prompts")
    generated = []
    for _, prompt in PROMPTS:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(templated, return_tensors="pt").input_ids.to(base.device)
        with torch.no_grad():
            out = base.generate(ids, max_new_tokens=256, do_sample=False,
                                repetition_penalty=1.0, top_k=0,
                                pad_token_id=tokenizer.eos_token_id)
        generated.append(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    gen_hidden, gen_targets, gen_consumed = extract_features(
        base, tokenizer, generated, M, progress_every=0
    )
    print(f"  gen:   {gen_hidden.shape[0]:,} examples from {gen_consumed:,} tokens")

    # Free the base: training only touches cached features, so 8.1 GiB of weights
    # need not compete with the optimiser state.
    del base
    medusa.base_model = None
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  base released; VRAM now {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    # ---------------------------------------------------------------- train
    print(f"\n{'=' * 78}\nTRAINING  ({args.epochs} epochs, batch {args.batch_size}, "
          f"lr {args.lr})\n{'=' * 78}")
    before = evaluate_heads(medusa, eval_hidden, eval_targets)
    print(f"  untrained head-1 accuracy (sanity, should be ~0): {before[0]:.4%}")
    train_heads(medusa, hidden, targets, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr)

    # ---------------------------------------------------------------- results
    print(f"\n{'=' * 78}\nPER-HEAD ACCURACY\n{'=' * 78}")
    train_acc = evaluate_heads(medusa, hidden[: eval_hidden.shape[0]],
                               targets[:, : eval_hidden.shape[0]])
    lib_acc = evaluate_heads(medusa, eval_hidden, eval_targets)
    gen_acc = evaluate_heads(medusa, gen_hidden, gen_targets)

    print(f"  {'head':>5}{'predicts':>10}{'train':>10}{'lib held-out':>14}"
          f"{'generated':>12}{'chain (gen)':>13}")
    chain = 1.0
    for i in range(M):
        chain *= gen_acc[i]
        print(f"  {i + 1:>5}{'t+' + str(i + 1):>10}{train_acc[i]:>9.1%}"
              f"{lib_acc[i]:>13.1%}{gen_acc[i]:>11.1%}{chain:>12.1%}")

    print(f"\n{'=' * 78}\nPROJECTED SPEEDUP\n{'=' * 78}")
    for name, acc in (("held-out library Python", lib_acc),
                      ("generated code (target workload)", gen_acc)):
        projection = projected_speedup(acc)
        print(f"  {name}")
        print(f"    expected accepted tokens : {projection['expected_accepted']:.2f}")
        print(f"    tokens per base forward  : {projection['tokens_per_forward']:.2f}")
        print(f"    projected speedup        : {projection['speedup']:.2f}x")

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    gen = projected_speedup(gen_acc)
    print(f"  Phase 1 coverage ceiling   5.69x   (perfect heads, top-6k mask)")
    print(f"  Phase 2 measured           {gen['speedup']:.2f}x   "
          f"(trained heads, same mask)")
    print(f"  Evolutionary Tree (built)  2.39x   (measured end-to-end, no training)")
    print()
    print(f"  head 1 accuracy {gen_acc[0]:.1%} -- this head distils the base lm_head,")
    print("  a linear map from the same hidden state, so it bounds how well the")
    print("  pipeline can possibly work. Heads 2+ must predict without seeing the")
    print("  intervening tokens, and the chain column shows how fast that compounds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
