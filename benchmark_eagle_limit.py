"""Chain accuracy of latent extrapolation (EAGLE) against the Medusa collapse.

Medusa's per-head accuracy collapsed because every head projected from the same
`h_t` with no autoregressive feedback: 27.7% at t+2, 10.9% at t+3, and a chain
product that was finished by head 4. This trains a single latent extrapolation head
and measures the same curve, so the two are directly comparable.

    python benchmark_eagle_limit.py
    python benchmark_eagle_limit.py --steps 8 --max-tokens 800000

Method
------
1. Extract ``(h_t, greedy tokens)`` from real Python with one teacher-forced forward
   per chunk. Targets are the base model's own greedy tokens, which is what acceptance
   is decided against, and the next hidden state ``h_{t+1}``, which the head regresses.
2. Train the head on both losses -- smooth L1 on the predicted hidden state and cross
   entropy on the logits it produces through the frozen ``lm_head``.
3. Evaluate **free-running**: start at ``h_t``, run K extrapolation steps feeding each
   step the token the previous step predicted, and compare against the base model's
   greedy continuation. Errors compound exactly as they would while drafting. A
   teacher-forced evaluation would flatter the head badly.

Accounting note
---------------
Position t+1 is ``argmax lm_head(h_t)`` -- the base model's own output, exact and
free. It contributes a fixed 1.0 to expected accepted length and belongs to neither
architecture's head. The Medusa figures are recomputed here under that same
convention: the previously reported 1.86x double-counted position t+1, since Medusa's
head 1 predicts the very token ``lm_head`` already gives, at strictly worse accuracy.
Corrected, Medusa is 1.31x. Both architectures are scored by
:func:`core.latent_extrapolator.chain_speedup`.
"""

from __future__ import annotations

import argparse
import glob
import io
import random
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmark import PROMPTS, TARGET_ID
from core.latent_extrapolator import LatentExtrapolator, chain_speedup

# Measured in Phase 2, per-position accuracy on generated code for positions t+1..t+10.
MEDUSA_ACCURACY = [0.655, 0.277, 0.109, 0.053, 0.037, 0.022, 0.021, 0.018, 0.016, 0.011]

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


@torch.no_grad()
def extract(base, tokenizer, texts, chunk_len, max_tokens, progress=50):
    """Collect per-chunk hidden states and the base model's greedy tokens.

    Kept chunk-shaped rather than flattened: the extrapolator attends over a sequence,
    so training needs contiguous runs, and a chunk boundary is a legitimate sequence
    boundary.
    """
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
    consumed = 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        for start in range(0, ids.numel() - 2, chunk_len):
            chunk = ids[start: start + chunk_len]
            if chunk.numel() < 16:
                continue
            out = base(input_ids=chunk.unsqueeze(0).to(base.device),
                       output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1][0].to(torch.float16).cpu()   # [T, H]
            greedy = out.logits[0].argmax(dim=-1).cpu()                 # [T]
            chunks.append((hidden, greedy))
            consumed += chunk.numel()
            if progress and len(chunks) % progress == 0:
                print(f"    {len(chunks)} chunks, {consumed:,} tokens", flush=True)
            if consumed >= max_tokens:
                return chunks, consumed
    return chunks, consumed


def train(model, chunks, epochs, lr, ce_weight, device, log_every=100):
    """Fit the head with smooth L1 on hidden states plus CE through lm_head."""
    optimiser = torch.optim.Adam(model.head.parameters(), lr=lr)
    step = 0
    start = time.perf_counter()

    for epoch in range(epochs):
        random.shuffle(chunks)
        running_l1 = running_ce = 0.0
        seen = 0
        for hidden_cpu, greedy_cpu in chunks:
            usable = hidden_cpu.shape[0] - 1
            if usable < 2:
                continue
            hidden = hidden_cpu[:usable].to(device=device, dtype=torch.float32).unsqueeze(0)
            target_hidden = hidden_cpu[1: usable + 1].to(
                device=device, dtype=torch.float32
            ).unsqueeze(0)
            # Token consumed at step t is the base's greedy token for position t+1.
            consumed_tokens = greedy_cpu[:usable].to(device).unsqueeze(0)
            # The logits from the predicted state should name position t+2's token.
            target_tokens = greedy_cpu[1: usable + 1].to(device)

            with torch.no_grad():
                embeddings = model.embed(consumed_tokens).to(torch.float32)

            predicted = model(embeddings, hidden)
            l1 = nn.functional.smooth_l1_loss(predicted, target_hidden)
            logits = model.project(predicted[0]).float()
            ce = nn.functional.cross_entropy(logits, target_tokens)
            loss = l1 + ce_weight * ce

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            optimiser.step()

            running_l1 += l1.detach().item()
            running_ce += ce.detach().item()
            seen += 1
            step += 1
            if log_every and step % log_every == 0:
                print(f"    epoch {epoch + 1} step {step:>5}  "
                      f"L1 {running_l1 / seen:.4f}  CE {running_ce / seen:.4f}",
                      flush=True)
        print(f"  epoch {epoch + 1}/{epochs}  L1 {running_l1 / max(1, seen):.4f}  "
              f"CE {running_ce / max(1, seen):.4f}  "
              f"({time.perf_counter() - start:.0f}s)", flush=True)


@torch.no_grad()
def chain_accuracy(model, chunks, steps, device, sample_stride=4):
    """Free-running per-step accuracy: does step i name the base's greedy token?

    Each step is fed the token the previous step predicted, so a mistake at step 2
    poisons step 3. That compounding is the point -- it is what a draft actually
    experiences, and what a teacher-forced measurement would hide.
    """
    correct = [0] * steps
    total = 0

    for hidden_cpu, greedy_cpu in chunks:
        length = hidden_cpu.shape[0]
        if length < steps + 2:
            continue
        positions = range(0, length - steps - 1, sample_stride)
        index = torch.tensor(list(positions))
        if index.numel() == 0:
            continue

        state = hidden_cpu[index].to(device=device, dtype=torch.float32).unsqueeze(1)
        token = model.project(state).argmax(dim=-1)          # [N, 1] exact, position t+1
        cache = None
        total += index.numel()

        for depth in range(steps):
            embedding = model.embed(token).to(torch.float32)
            state, cache = model.head(embedding, state, cache)
            token = model.project(state).argmax(dim=-1)
            truth = greedy_cpu[index + depth + 1].to(device)
            correct[depth] += int((token[:, 0] == truth).sum())

    return [c / max(1, total) for c in correct]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5, help="K extrapolation steps")
    parser.add_argument("--chunk-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=400_000)
    parser.add_argument("--eval-tokens", type=int, default=40_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--train-files", type=int, default=400)
    parser.add_argument("--save-head", default=None,
                        help="write the trained head here so the latent tree engine "
                             "can load it instead of retraining")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this needs a GPU.")
        return 1

    device = "cuda:0"
    K = args.steps
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    print(f"Loading base (8-bit, frozen): {TARGET_ID}")
    base = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map=device, dtype=torch.float16,
    ).eval()

    print(f"\n{'=' * 78}\nARCHITECTURE\n{'=' * 78}")
    model = LatentExtrapolator(base, intermediate_size=args.intermediate_size)
    print(f"  {model.parameter_report()}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable: {trainable / 1e6:.1f}M "
          f"({trainable * 16 / 2**30:.2f} GiB with Adam)")
    print(f"  embedding and lm_head are the base model's, frozen and reused -- the")
    print(f"  head learns feature dynamics, never a vocabulary projection.")

    # ---------------------------------------------------------------- data
    print(f"\n{'=' * 78}\nFEATURE EXTRACTION\n{'=' * 78}")
    files = corpus_files()
    train_chunks, consumed = extract(
        base, tokenizer, read(files[: args.train_files]), args.chunk_len, args.max_tokens
    )
    print(f"  train: {len(train_chunks)} chunks, {consumed:,} tokens")

    lib_chunks, lib_consumed = extract(
        base, tokenizer, read(files[args.train_files: args.train_files + 60]),
        args.chunk_len, args.eval_tokens, progress=0
    )
    print(f"  held-out library: {len(lib_chunks)} chunks, {lib_consumed:,} tokens")

    print("  held-out generated code: the 10 benchmark prompts")
    generated = []
    for _, prompt in PROMPTS:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(templated, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = base.generate(ids, max_new_tokens=256, do_sample=False,
                                repetition_penalty=1.0, top_k=0,
                                pad_token_id=tokenizer.eos_token_id)
        generated.append(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    gen_chunks, gen_consumed = extract(
        base, tokenizer, generated, args.chunk_len, 10**9, progress=0
    )
    print(f"  generated: {len(gen_chunks)} chunks, {gen_consumed:,} tokens")

    # ---------------------------------------------------------------- train
    print(f"\n{'=' * 78}\nTRAINING  ({args.epochs} epochs, lr {args.lr}, "
          f"ce_weight {args.ce_weight})\n{'=' * 78}")
    cold = chain_accuracy(model, gen_chunks, K, device)
    print(f"  untrained step-1 accuracy (sanity, should be ~0): {cold[0]:.3%}")
    train(model, train_chunks, args.epochs, args.lr, args.ce_weight, device)

    # ---------------------------------------------------------------- results
    print(f"\n{'=' * 78}\nCHAIN ACCURACY  (free-running, errors compound)\n{'=' * 78}")
    lib = chain_accuracy(model, lib_chunks, K, device)
    gen = chain_accuracy(model, gen_chunks, K, device)

    print(f"  {'position':>10}{'EAGLE lib':>12}{'EAGLE gen':>12}"
          f"{'Medusa gen':>12}{'EAGLE chain':>13}")
    print(f"  {'t+1':>10}{'exact':>12}{'exact':>12}{'exact':>12}{1.0:>12.1%}")
    running = 1.0
    for i in range(K):
        running *= gen[i]
        medusa = MEDUSA_ACCURACY[i + 1] if i + 1 < len(MEDUSA_ACCURACY) else float("nan")
        print(f"  {'t+' + str(i + 2):>10}{lib[i]:>11.1%}{gen[i]:>11.1%}"
              f"{medusa:>11.1%}{running:>12.1%}")

    print(f"\n{'=' * 78}\nPROJECTED SPEEDUP  (position t+1 exact, counted once)\n{'=' * 78}")
    eagle_gen = chain_speedup(gen)
    eagle_lib = chain_speedup(lib)
    medusa_gen = chain_speedup(MEDUSA_ACCURACY[1: K + 1])
    medusa_full = chain_speedup(MEDUSA_ACCURACY[1:])

    print(f"  EAGLE  (library held-out)  {eagle_lib['speedup']:.2f}x")
    print(f"  EAGLE  (generated code)    {eagle_gen['speedup']:.2f}x")
    print(f"  Medusa (generated, K={K})   {medusa_gen['speedup']:.2f}x")
    print(f"  Medusa (generated, M=10)   {medusa_full['speedup']:.2f}x   "
          f"<- corrects the 1.86x previously reported, which double-counted t+1")

    if args.save_head:
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.head.state_dict().items()},
                "hidden_size": model.hidden_size,
                "num_attention_heads": model.head.num_heads,
                "intermediate_size": model.head.gate_proj.out_features,
                "steps_trained": K,
                "chain_accuracy_generated": gen,
                "chain_accuracy_library": lib,
            },
            args.save_head,
        )
        print("\n  head saved to " + args.save_head
              + f" ({model.head_parameters * 4 / 2**20:.0f} MiB fp32)")

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    improvement = eagle_gen["speedup"] / medusa_gen["speedup"]
    print(f"  latent feedback vs unconditional heads: {improvement:.2f}x better")
    print(f"  step t+2 accuracy: EAGLE {gen[0]:.1%} vs Medusa "
          f"{MEDUSA_ACCURACY[1]:.1%}")
    print()
    print(f"  Evolutionary Tree (measured end-to-end)  2.39x")
    print(f"  EAGLE  (projected from chain accuracy)   {eagle_gen['speedup']:.2f}x")
    print()
    print("  The projection is an upper bound on throughput: it counts tokens per base")
    print("  forward and charges nothing for the K extrapolation steps or the")
    print("  verification pass. The tree's 2.39x is a measured wall-clock number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
