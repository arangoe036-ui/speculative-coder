# Speculative-Coder: High-Throughput Local Inference via Modified Rejection Sampling

A from-scratch PyTorch implementation of speculative decoding — **2.06x faster
local inference on a 7B code model, with the output distribution provably
unchanged.**

No `vllm`, no `assistant_model=`, no wrapper around someone else's scheduler.
The rejection sampler, the two-model generation loop, and the KV-cache rollback
are all implemented and tested here directly.

---

## 1. The Hook

A 1.5B draft model proposes tokens; a 7B target model verifies a whole block of
them in a single forward pass. Tokens the target would not have produced are
rejected and replaced by sampling from the residual distribution
`norm(max(0, p - q))` — the modified rejection sampling scheme of
[Leviathan, Kalman & Matias (2023)](https://arxiv.org/abs/2211.17192),
Algorithm 1. The guarantee is not an approximation: every emitted token is
distributed exactly as the target model would have sampled it, no matter how bad
the draft model is. A weak draft costs throughput, never quality.

| | |
|---|---|
| **Throughput** | **28.4 tok/s**, up from 13.8 tok/s autoregressive — **2.06x** |
| **Target forward passes** | **61.7** per response, down from 299 — **4.78 tokens per forward** |
| **Acceptance rate (α)** | 76.7% mean, peaking at 97.2% |
| **Peak VRAM** | 11.08 GiB of 15.9 — both models resident, no offload |
| **Hardware** | Single RTX 5080 (16 GB, Blackwell sm_120) |
| **Target** | `Qwen2.5-Coder-7B-Instruct`, 8-bit (LLM.int8) |
| **Draft** | `Qwen2.5-Coder-1.5B-Instruct`, bfloat16 |
| **Tests** | 83 passing, including a bitwise-equivalence oracle |

The measured win comes from restructuring compute, not from spending memory:
peak VRAM is within 0.02 GiB across every configuration benchmarked.

### How it works

Each iteration:

1. **Draft** — the 1.5B model autoregressively proposes `K` tokens (`K` cheap
   sequential forwards), recording its distribution `q` at each step.
2. **Verify** — the 7B target scores the entire block in **one** forward pass,
   yielding `p` for all `K` positions plus a free `K+1`-th "bonus" distribution.
3. **Accept / reject** — each drafted token is accepted with probability
   `min(1, p/q)`; the first rejection truncates the block and its replacement is
   drawn from the normalised residual.
4. **Roll back** — rejected tokens are already in both KV caches, so both are
   cropped to the accepted prefix. This is a slice, not a recomputation.

One expensive forward therefore yields up to `K+1` tokens instead of one.

### Repository layout

| Path | Role |
|---|---|
| `core/verifier.py` | The mathematics. Modified rejection sampling, model-free. |
| `core/engine.py` | `SpeculativeEngine` — the draft/verify loop and KV rollback. |
| `benchmark.py` | 10-prompt × K-sweep benchmark, emits `benchmark_results.md`. |
| `exactness_check.py` | Controlled precision sweep proving bitwise exactness. |
| `cli.py` | Interactive streaming REPL with a live acceptance HUD. |
| `run_demo.py` | Minimal single-prompt demo. |
| `tests/` | 83 tests: statistical, equivalence, structural. |

---

## 2. The Benchmark

Ten Python coding prompts spanning algorithms, data structures, parsing, regex,
async, generators, decorators, numerics, classes and error handling. Each run
against the 7B target alone and then the speculative engine at `K ∈ {1,3,5,7}`.
Greedy sampling, 384-token budget, timings synchronised on the CUDA stream, with
a warm-up generation discarded so kernel autotuning is not billed to whichever
configuration happened to run first.

| Configuration | Time (s) | Tok/s | Speedup | Alpha | Fwd passes | Tok/fwd | Peak VRAM | Code OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (7B only) | 21.68 | 13.8 | 1.00x | — | 299.0 | 1.00 | 11.07 GiB | 9/10 |
| Speculative K=1 | 15.04 | 20.0 | 1.45x | 91.7% | 157.4 | 1.91 | 11.07 GiB | 9/10 |
| Speculative K=3 | 10.71 | 26.0 | 1.89x | 82.6% | 80.9 | 3.45 | 11.08 GiB | 9/10 |
| **Speculative K=5** | **10.37** | **28.4** | **2.06x** | **76.7%** | **61.7** | **4.78** | **11.08 GiB** | **9/10** |
| Speculative K=7 | 10.44 | 28.1 | 2.03x | 69.1% | 51.1 | 5.74 | 11.08 GiB | 9/10 |

`Tok/fwd` — tokens emitted per target forward pass — is the hardware-independent
measure of the win. The baseline is exactly 1.00 by definition. `Code OK` counts
runs whose generated Python parses via `ast.parse`. There were **zero `FAIL`s**
across all 50 runs; the missing row in each column is `datastruct` — the longest
task, which hits the 384-token budget in every configuration — scored `TRUNC`
rather than `FAIL` so that "ran out of budget" is not conflated with "wrote
invalid code".

### Why K=5 is the Pareto knee

The two terms move in opposite directions, and `K=5` is where they cross.

**Tokens per target forward keeps improving with K** — 1.91 → 3.45 → 4.78 → 5.74.
On the target side, larger `K` is monotonically better. If the draft model were
free, `K=7` would win outright.

**But acceptance decays** — 91.7% → 82.6% → 76.7% → 69.1%. Acceptance is
multiplicative down the block: surviving to position `i` requires every earlier
token to have been accepted, so the expected accepted run length grows
sub-linearly in `K` while the draft cost grows exactly linearly.

That linear cost is what closes the gap. Each iteration spends `K` *sequential*
draft forwards. Going from `K=5` to `K=7` cuts target forwards from 61.7 to 51.1
— saving ~10 — while raising draft forwards from ≈308 to ≈358, adding ~50. With
a 1.5B draft against an 8-bit 7B target, those roughly cancel, and the measured
throughput duly plateaus (28.4 → 28.1 tok/s). Past this point the draft becomes
the bottleneck and throughput declines.

`K=5` is therefore the best *fixed* choice for this pair, but the frontier is
prompt-dependent, which is the more interesting result:

| Prompt | α at K=7 | Best K | Speedup |
|---|---:|---|---:|
| `datastruct` | 84.4% | **K=7** | 2.46x |
| `oop` | 83.6% | **K=7** | 2.36x |
| `regex` | 53.4% | **K=5** | 1.90x (K=7 drops to 1.67x) |

Predictable, boilerplate-heavy text — class scaffolding, tree traversals — drafts
almost perfectly and rewards a long speculation window. Dense, information-rich
text like regex construction does not, and there `K=7` is actively *worse* than
`K=5`: the draft burns seven forwards to have four rejected. A fixed `K` leaves
this on the table, which is the argument for adapting `K` to recent acceptance.

Full per-prompt tables: [`benchmark_results.md`](benchmark_results.md).

---

## 3. The Engineering Falsification

At temperature 0, speculative decoding should reproduce plain greedy decoding on
the target **token for token**. It did not. Chasing that down took two wrong
hypotheses, and the wrong ones are the instructive part.

**Hypothesis 1: int8 kernels.** `bitsandbytes` selects kernel paths by input
shape, so scoring a token inside a `K+1`-wide forward could plausibly differ from
scoring it alone. Measured on the target directly: a real gap, up to 1.2 logits.
Case closed — except it wasn't. A controlled sweep holding the draft, target
weights, engine and prompts fixed and varying **only** the target's precision
showed **bfloat16 diverging too**. Not quantization.

**Hypothesis 2: reduced precision generally.** Also wrong, and the refutation was
decisive: **fp32 diverged as well — at the *same token indices* as bf16**, with a
regime gap of `3.4e-05` against a tightest top-2 logit margin of `8.6e-02`. That
gap is ~2500x too small to flip an argmax. And there it was: *identical
divergence points across three different precisions is not what numerical noise
looks like.* Noise does not reproduce itself. The cause had to be systematic.

**The actual bug was in the baseline.** Qwen ships `repetition_penalty: 1.1` in
its `generation_config.json`, and `transformers` applies `repetition_penalty` as
a `LogitsProcessor` **even when `do_sample=False`**. The "greedy baseline" was
greedy-plus-repetition-penalty; the engine was pure greedy. Two different
sampling rules, compared as though they were one. Overriding
`repetition_penalty=1.0, top_k=0`:

| Target precision | Identical to baseline | Mean regime gap | Tightest top-2 margin |
|---|---:|---:|---:|
| **float32** | **5/5** | 1.6e-05 | 8.6e-02 |
| bfloat16 | 4/5 | 2.0e-01 | 1.3e-01 |
| 8-bit (LLM.int8) | 2/5 | 7.5e-01 | 9.4e-02 |

**In fp32 the engine is bitwise exact on real GPU models**, and the residual
bf16/int8 divergence is genuine precision sensitivity — monotonic in precision,
and predicted each time by whether the regime gap exceeds the top-2 margin. The
proof needs the target to score a position identically however it is evaluated;
a baseline scores one token on a KV cache while the engine scores a `K+1`-wide
block, and those agree only to the precision's resolution. Where the target has a
near-tie tighter than that gap, rounding decides it, and one flipped token
changes everything after it.

So: the loop was correct all along; a *premise* fails at low precision, and a
config file quietly broke the measurement. Reproduce with `python
exactness_check.py`. Both dead hypotheses are documented in that file rather than
deleted, because the falsification is the evidence.

### What the tests actually verify

- **The mathematics** — 10,000-run Monte Carlo goodness-of-fit (per-category
  z-test, Pearson χ², total variation) confirming emitted tokens follow the
  target distribution, plus separate checks that accepted tokens follow
  `norm(min(p,q))`, rejections follow `norm(max(0, p−q))`, and acceptance equals
  `1 − TV(p,q)`.
- **The loop** — at temperature 0 the engine must equal an *independent*
  no-cache full-recompute greedy oracle, token for token, across
  `K ∈ {1,2,3,5,8}` and three draft qualities (~68%/~26%/~1% acceptance).
- **The tests themselves** — every load-bearing invariant was mutation-tested.
  Resampling from `p` instead of the residual is caught at 20σ. A KV cache one
  position too long fails 22 tests; one position too *short* — still correct,
  merely wasteful — fails 1, which is why cache length is asserted by equality
  rather than inequality.

```bash
python -m pytest tests/ -v      # 83 passed
```

---

## 4. Usage

### Setup

Requires Python 3.12+ and a CUDA GPU with ≈12 GiB free. Blackwell cards
(RTX 50-series, sm_120) need a CUDA 12.8+ build of PyTorch.

```bash
git clone <repo> && cd speculative-coder
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; use bin/activate on Linux
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install numpy transformers accelerate bitsandbytes pytest
python download_models.py            # pre-fetch both checkpoints (~18 GB)
```

### Interactive REPL

```bash
python cli.py
```

Tokens stream in as each block is verified, and every response ends with a HUD:

````text
Speculative Coder | NVIDIA GeForce RTX 5080
Loading target (8-bit): Qwen/Qwen2.5-Coder-7B-Instruct
Loading draft (bfloat16): Qwen/Qwen2.5-Coder-1.5B-Instruct
Calibrating autoregressive baseline... 13.9 tok/s
Ready. K=5, temperature=0.7, VRAM 11.06 GiB. /help for commands.

>>> Write a Python function that checks if a string is a palindrome.

```python
def is_palindrome(s):
    normalized_string = s.replace(" ", "").lower()
    return normalized_string == normalized_string[::-1]
```
[Alpha: 87% | Speedup: 2.1x | Tok/s: 29.2]
````

The speedup is **measured, not assumed**: at startup the CLI times a short
autoregressive generation on the target alone — twice, discarding the first so
CUDA autotuning is not billed to the baseline — and uses that as the reference.
With `--no-calibrate` there is nothing to divide by, so the HUD reports tokens
per target forward pass instead of inventing a ratio. Alpha is colour-coded
green/amber/red, since it is the number that explains a disappointing speedup.

| Command | Effect |
|---|---|
| `/k <n>` | Change draft length `K` live |
| `/temp <x>` | Change temperature (`0` = greedy) |
| `/tokens <n>` | Change the token budget |
| `/stats` | Full statistics for the last response, including per-iteration acceptance |
| `/help` | List commands |
| `/quit` | Exit |

Flags: `--k`, `--temperature`, `--top-p`, `--max-new-tokens`, `--no-calibrate`,
`--no-color`.

### Benchmark, demo, and verification

```bash
python benchmark.py                       # full sweep -> benchmark_results.md
python benchmark.py --limit 3 --max-new-tokens 128
python benchmark.py --temperature 0.7     # sampled instead of greedy

python run_demo.py --baseline             # single prompt vs autoregressive
python exactness_check.py                 # fp32/bf16/int8 exactness sweep
python -m pytest tests/ -v
```

---

## Notes and limitations

- **Sampling semantics.** `temperature` and `top_p` are applied *identically* to
  draft and target. The guarantee is relative to whatever distribution the target
  is sampled from, so warping one and not the other silently breaks it.
- **Vocabulary reconciliation.** These sibling checkpoints pad their embedding
  matrices to different widths (1.5B → 151936, 7B → 152064) despite sharing a
  tokenizer. Both logits tensors are sliced to the shared prefix *before* the
  softmax; without this, `p` and `q` index different token spaces.
- **Bitwise exactness needs fp32.** With a quantized target, treat the engine as
  distribution-preserving up to that precision, not as a bitwise drop-in. The
  throughput numbers are unaffected.
- **Batch size 1.** The engine handles one sequence at a time. Batched
  speculative decoding needs per-sequence acceptance lengths and ragged cache
  rollback, which is a materially different design.
- **KV growth.** Combined 84 KiB/token (draft 28 + target 56, both 28 layers with
  GQA). At the ~2.7 GiB of headroom above, contexts beyond roughly 32k tokens
  will not fit alongside the weights.
- **Fixed K.** The per-prompt table in §2 shows the optimum moving between K=5
  and K=7 with acceptance. Adapting `K` to a running estimate of α is the obvious
  next step.

## Reference

Yaniv Leviathan, Matan Kalman, Yossi Matias. *Fast Inference from Transformers
via Speculative Decoding.* ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
