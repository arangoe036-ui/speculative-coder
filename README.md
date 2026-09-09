# Speculative-Coder — 2.40× faster local inference, provably lossless

**13.6 → 32.7 tok/s on a 7B code model on a single RTX 5080, emitting exactly the tokens
the target model would have sampled.** No `vllm`, no `assistant_model=`. The modified
rejection sampler, the generation loop, the KV-cache rollback, twelve speculation
strategies, two training runs and five diagnostic harnesses are implemented from scratch
and covered by 451 tests.

The winning design drafts a **single** branch and splits it only where the drafter is
unsure — certainty costs one batch row, uncertainty costs two. It reaches the highest
measured throughput here on **3.1× less draft compute** than fixed-width parallel
drafting, at higher acceptance.

**Losslessness is verified, not asserted.** A 10,000-run Monte Carlo goodness-of-fit on
the rejection sampler, an *independent* full-recompute greedy oracle that every engine
must match token for token, and a precision sweep reported as measured — fp32 5/5
bitwise identical, bf16 4/5, int8 2/5. The standard implementation bug in this
algorithm, resampling from `p` instead of the residual, is caught at **20σ** — so the
test is proven able to fail.

Twelve architectures were built and measured; eight of them lost. Each is
[mapped with the mechanism that ruled it out](#the-design-space-what-does-not-work),
which is *why* the 2.40× is trustworthy: the champion won against every alternative I
could build, not against no alternative at all.

---

## Results

Ten Python coding prompts, forced 256 tokens per run, greedy target, single RTX 5080
(16 GB). Target `Qwen2.5-Coder-7B-Instruct` in 8-bit; drafter, where used,
`Qwen2.5-Coder-0.5B-Instruct` in bfloat16.

| Architecture | Tok/s | Speedup | Lossless | Weights VRAM | Verdict |
|---|---:|---:|:-:|---:|---|
| Baseline (7B alone) | 13.6 | 1.00x | — | 8.11 GiB | reference |
| **Evolutionary Tree** (0.5B, KV-mitosis) | **32.7** | **2.40x** | yes | 9.05 GiB | **throughput champion** |
| Monte Carlo (0.5B, B=8 fixed width) | 32.0 | 2.32x | yes | 9.05 GiB | superseded by the tree |
| **Adaptive Dual-Gate** (0.5B) | **30.4** | **2.21x** | yes | 9.05 GiB | **deployment champion** |
| Hybrid n-gram Cascade | 30.4 | 2.21x | yes | 9.05 GiB | falsified (+1%, noise) |
| Static K=5 (0.5B) | 28.3 | 2.08x | yes | 9.05 GiB | baseline speculation |
| Particle Filter (SMC) | 27.3 | 1.98x | yes | 9.05 GiB | falsified |
| EAGLE latent chain *(projected)* | — | 1.35x | yes | 8.55 GiB | falsified |
| Unconditional Medusa *(projected)* | — | 1.31x | yes | 8.51 GiB | falsified |
| Speculative Jacobi (block=5) | 14.0 | 1.02x | yes | **8.11 GiB** | falsified |
| Latent Mitosis (EAGLE tree) | 12.4 | 0.91x | yes | 9.23 GiB | falsified |
| Twin-Cache self-speculation | 9.0 | 0.65x | yes | **8.11 GiB** | falsified |
| Cascade early-exit (layer 14) | — | ~1.00x | **no** | 8.11 GiB | falsified |

Every configuration except the last is **distribution-preserving**: emitted tokens are
exactly what the target model would have produced. That is verified, not asserted —
see [Correctness and method](#correctness-and-method).

---

## The Dual Champions

### 1. Evolutionary Tree — 2.40x, 32.7 tok/s *(throughput champion)*

Fixed-width parallel drafting pays for B branches from the first token, including at
the many positions where the drafter is completely certain. This grows the tree
instead.

Drafting starts as a **single** branch taking argmax tokens. At each depth step the
engine inspects each branch's top-1 probability, and where it falls **below 0.80** —
where the drafter is genuinely unsure — the branch undergoes **KV-mitosis**: its cache
and token history are cloned into a new batch row, one child takes the top-1 token and
its sibling the top-2, capped at 8 leaves. Certainty costs one row; uncertainty costs
two. All leaves are verified in one batched target forward and the longest-accepted
leaf is committed.

| Config | Iters | Leaves | Splits | **Draft rows** | Rows/tok | Best leaf | Tok/fwd |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Tree (split<0.80)** | 367 | **4.72** | 1,401 | **7,620** | **2.98** | **6.12** | **7.09** |
| Tree (split<0.95) | 394 | 6.25 | 2,148 | 11,386 | 4.45 | 5.91 | 6.88 |
| Monte Carlo B=8 (fixed) | 378 | 8.00 | 0 | **23,848** | 9.32 | 5.91 | 6.88 |

**3.1x less draft compute than fixed-width Monte Carlo while producing strictly
superior sequences** — higher best-leaf acceptance (6.12 vs 5.91), higher alpha (77.1%
vs 74.9%), more tokens per expensive forward (7.09 vs 6.88).

Two mechanisms. The trunk is the drafter's **greedy** path, already far better than any
sampled branch — Monte Carlo's *average* branch accepted 3.49 tokens. And splitting
only at uncertainty places the recovery option exactly where errors occur.
`split<0.95` shows the other side: more leaves, 1.5x the draft rows, a *lower* best
leaf, because it splits where there was no real uncertainty.

**Honest limits.** The +2.8% edge over Monte Carlo is inside run-to-run variation
(repeated Monte Carlo runs span 30.7–32.0 tok/s). Against the dual gate (+8.2%, same
run) the gap is solid. The tree is the *fastest measured* configuration, not
established as faster than Monte Carlo. And the 3.1x compute saving barely converts to
wall clock, because `batch_scaling.py` measures a batch-8 forward at **1.06x** the cost
of batch-1 — narrowing a batch reclaims compute that was nearly free. The acceptance
improvement is the real contribution; the compute saving is what would let it scale to
larger leaf counts.

### 2. Adaptive Dual-Gate — 2.21x, 30.4 tok/s *(deployment champion)*

Fixed-K speculation discards every token drafted after the first rejection. This
abandons fixed K and stops early on two signals.

**Statistical gate.** Stop when the drafter's top-1 probability falls below a
threshold. The threshold was *calibrated*, not guessed (`gate_calibration.py`, 1,893
drafted tokens): the drafter's top-1 probability has a **median of 0.99**, so the
intuitive 0.35 sits near the 1st percentile and fires on only 8% of blocks. Acceptance
rises monotonically with confidence (36% → 77% across buckets), and **τ = 0.65** cuts
33% of draft forwards while the tokens it skips had only 30% acceptance against 71%
overall.

**Syntactic gate.** A bracket/quote state machine halts a draft that closes a bracket
it never opened. It fires **zero times** across 20 prompt-runs — a real negative
result, not a broken tracker: instrumentation confirms 1,780 chars fed, code fences
entered, brackets balanced 8/8. Kept behind a flag; not claimed as a win.

**Why this is the deployment choice.** It needs no batch memory (**+0.94 GiB** over
running the 7B alone, and 1.01 draft rows/token — the leanest here), and it is the
**only accelerated configuration that works at any temperature.** All five breadth
engines are greedy-only.

One subtlety worth stealing: the gate reads the drafter's **unwarped** softmax. At
temperature 0 the warped `q` is one-hot with top-1 exactly 1.0, so a gate built on it
would never fire — indistinguishable from a gate that found nothing.

---

## The Design Space: What Does Not Work

Eight of the twelve architectures lost, in the six post-mortems below. Each stays in the
repository with its telemetry, because the mechanism is the reusable part — together they
map where batch-1 local inference actually binds. Two of them took full training runs to
rule out.

### 1. Twin-Cache Self-Speculation — 0.65x
**Failed: batch-1 weight-bandwidth dominance.**

One model drafting for itself, made "cheap" by windowing its own KV cache to 64 tokens.
Cropping the cache makes draft *attention* cheap, but attention is not the cost: the
forward reads **~8.1 GiB of weights against ~17 MB of KV cache** at 300 tokens — 0.2%
of the traffic. Measured, the windowed draft forward cost **76.4 ms against the
baseline's 72.7 ms** — slightly *more*. KV traffic would not rival weight traffic until
roughly **152k tokens** of context.

Since the drafter *is* the target, K draft forwards plus one verification buys at most
K+1 tokens for K+1 full-price forwards, so forwards-per-token is ≥1.00 always, with
equality only at 100% acceptance. It measured **1.48**. The VRAM saving was real
(**2.89 GiB**) but it is dominated by simply decoding normally.

### 2. N-Gram Hybrid Cascade — +1% (inside noise)
**Failed: generation vs. editing domain mismatch.**

A free CPU prompt-lookup drafter routed ahead of the model. Prompt-lookup wins where
output copies input — code *editing*, RAG, summarisation. Generating from a short
instruction gives it almost nothing to copy.

It is squeezed from both sides: at `m=2` it fires on **16%** of blocks with only
**31.5%** acceptance; at `m=6` acceptance reaches **53.6%** but it fires on **4%**.
Never both. Break-even needs ~68%. It genuinely saved 18% of draft forwards, but
tokens-per-target-forward *fell* in every configuration (4.59 → 4.21–4.48), and each
extra iteration costs a 73 ms target forward against a 6 ms draft forward — a **12x
exchange rate**. Trading cheap draft compute for expensive target compute loses.

### 3. Particle Filter (Sequential Monte Carlo) — 1.98x
**Failed: diversity collapse.**

Prune the weakest half of draft particles each step, clone the strongest. The
hypothesis was right — resampling *did* produce statistically superior sequences,
lifting mean per-particle acceptance from **3.49 to 4.73** (+36%).

But it destroyed the thing that was paying. Breadth `Gain` fell from **+2.27 to +0.17**,
and distinct candidates reaching verification fell from **5.64 to 2.10** of 8
particles, with one branch winning 87–91% of iterations against Monte Carlo's 44%.
Best leaf dropped 5.76 → 4.90. The resampling weights are the drafter's *own*
log-probabilities with no observation of the target, making it a greedy search toward
the drafter's mode — exactly where a single greedy draft already goes. **Eight copies of
the drafter's favourite continuation are worth less than eight different guesses**,
because acceptance is decided by the target. Overhead was not the problem:
per-iteration cost was identical (215 vs 216 ms).

### 4. Unconditional & Latent Lookahead — Medusa 1.31x, EAGLE 1.35x, Latent Mitosis 0.91x
**Failed: chain-product accuracy decay and compounding autoregressive error.**

The most expensive entry, pursued through three architectures and two training runs.

**Medusa (M=10 heads, 1.31x).** Heads bolted on a frozen 7B, each predicting token
*t+i* directly from the same hidden state `h_t`. Trained on 400k tokens against the
base model's own greedy output:

| Position | t+1 | t+2 | t+3 | t+4 | t+5…t+10 |
|---|---:|---:|---:|---:|---:|
| Accuracy | exact | **27.7%** | 10.9% | 5.3% | 3.7% → 1.1% |
| **Chain** | **100%** | **27.7%** | **3.0%** | **0.2%** | ~0% |

A step commits the longest **all-correct prefix**, so accuracies multiply. The chain
falls below 1% by t+4: **effective M is 3, not 10**, and heads 4–10 are 320 MiB of
parameters contributing nothing.

**Capacity is provably not the constraint.** A rank-256 bottleneck head (**38M
params**) matched the full linear head (**402M**) to within **0.01x** on both eval
sets. Ten times the parameters bought nothing — the ceiling is the information content
of `h_t`.

Two sub-findings worth keeping. A full-vocabulary head is `3584 × 152064` = **545M
parameters**, so M=15 is **15.2 GiB and cannot be instantiated** alongside the 8.1 GiB
base; the output restriction's real payoff was making the projection *factorable* (26x
smaller, verified mathematically identical to 2.26e-08). And the AST restriction that
motivated it capped acceptance at **coverage**: structural tokens interleave rather than
cluster, giving a mean consecutive run of **0.79** with 54% of positions accepting
nothing. Widening to a top-6k frequency mask fixed coverage (run 0.79 → **4.69**,
ceiling 5.69x) and moved the binding constraint to accuracy, where it died.

**EAGLE latent chain (1.35x).** A single 115M transformer layer extrapolating the
*hidden state* — `h_{t+1} = Ext(emb(y_{t+1}), h_t)` — with the frozen `lm_head` turning
it into logits, so each step conditions on the previous step's prediction. The
conditioning deficit was real and fixing it helped: **t+2 accuracy 32.1% vs Medusa's
27.7%**, with 115M parameters against 402M. But the chain still collapsed one position
later (32.1% → 7.9% → 2.6%), and EAGLE is *worse* than Medusa from t+3 onward (7.9% vs
10.9%) because free-running compounds errors — a wrong token at t+2 poisons t+3, where
Medusa's independent heads each get a fresh shot from the true `h_t`.

**Latent Mitosis (0.91x).** Grafting the tree's KV-mitosis onto the EAGLE head produced
the *worst* breadth result, and the telemetry says exactly why:

| | Leaves | Splits | Draft rows/tok | Best leaf | Tok/fwd |
|---|---:|---:|---:|---:|---:|
| Tree (0.5B) | **4.72** | 1,401 | **2.98** | **6.12** | **7.09** |
| Latent Mitosis | **15.85** / 16 | 25,932 | **85.80** | **0.48** | 1.47 |

The tree's advantage is *selective* branching, which requires a **confident** drafter.
The head's top-1 confidence almost never clears 0.85, so it splits at nearly every step
and saturates the leaf budget immediately — "adaptive" degenerates to "always maximally
wide," at **29x the draft cost** for half an accepted token.

### 5. Speculative Jacobi (Fixed-Point Relaxation) — 1.02x
**Failed: no source of lookahead.**

Greedy decoding is the solution of a triangular non-linear system; Jacobi guesses the
whole block and relaxes it in parallel. It is **exact by construction with no rejection
sampling** — position 1's logits condition only on the true context, so after *k*
sweeps the first *k* tokens are provably the greedy sequence. The nicest mathematics in
the project.

But convergence is ~**1.1 tokens per sweep**, barely above the theoretical floor of
1.0, and each sweep costs a full target forward:

| Block | Sweeps/block | Tokens/block | Tok/fwd | **Converged** |
|---:|---:|---:|---:|---:|
| 5 | 4.59 | 5.28 | 1.15 | **32%** |
| 10 | 4.87 | 5.29 | 1.09 | **4%** |
| 16 | 4.94 | 5.30 | 1.08 | **1%** |

Larger blocks are strictly worse — `Tokens/block` pins at ~5.3 regardless of width, so
B=16 pays for a wider forward and gets the same yield. It has perfect conditioning and
no lookahead: the n-gram seed fires on 19–24% of blocks, so most start from "last token
repeated," and relaxation can only fix one position per full-price forward.

The commit rule needed a correction worth recording: taking only the longest common
prefix between consecutive sweeps commits **zero** tokens whenever the guess is wrong at
position 1, which hangs the loop. Committing `max(lcp, sweeps_performed)` is provably
sound and guarantees progress.

### 6. Cascade Early-Exit Verification — ~1.00x, and **lossy**
**Failed: late logit-lens resolution and KV-debt deferral.**

Project an intermediate hidden state through `lm_head` and accept the drafted token when
that distribution is confident. Measured before building the engine, because two claims
had to hold first — and neither did.

| Layer | Fires (τ=0.95) | **Agree ǀ fired** | Agree (all) | layers/tok | vs full |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.4% | 0.0% | 0.1% | 27.9 | 1.00x |
| **14** | **0.4%** | **50.0%** | **2.5%** | **27.9** | **1.00x** |
| 18 | 0.8% | 45.5% | 5.6% | 27.9 | 1.00x |
| 22 | 10.3% | 74.8% | 29.1% | 27.4 | 0.98x |
| 25 | 39.8% | 90.4% | 63.9% | 26.8 | 0.96x |

At layer 14 the gate fires on **0.4%** of positions and is right **half** the time; its
overall agreement with the full model is **2.5%**. Agreement only becomes useful in the
**last 3 layers** (63.9% at layer 25) — precisely where there is nothing left to skip.
The residual stream does not enter the output basis until the end.

Even at the best operating point the trade is bad: **4.3% nominal traffic saving for
9.6% of accepts corrupted.** And the nominal saving is not real — exiting at layer *L*
leaves layers *L*..27 with no KV entries for that position, so a later full pass must
supply them. Early exit **defers** work, it does not delete it; `CascadeBudget` accounts
for this explicitly, and with every exit outstanding `layers_per_token_with_debt` equals
the full 28.

This would also have been the project's **only lossy engine**. Everything else emits
exactly what the target would.

### Also measured and rejected

- **Static K sweep.** K=5 is the best fixed choice (2.08x), but the per-prompt optimum
  moves between K=5 and K=7 with acceptance — the argument for adapting K.
- **1.5B drafter.** Identical throughput to the 0.5B (30.2 vs 30.1 tok/s) for **1.95
  GiB more VRAM**. The smaller drafter is strictly better here.
- **M=10 Medusa heads.** Heads 4–10 contribute under 0.1% to the acceptance chain.

---

## What the study shows

Sorted by what each one supplies, the architectures triangulate a single conclusion:

| Architecture | Lookahead | Conditioning | Result |
|---|---|---|---|
| Jacobi | none | perfect | 1.02x |
| Medusa | cheap, unconditional | none | 1.31x |
| EAGLE chain | cheap, latent | latent, free-running | 1.35x |
| Latent Mitosis | cheap, latent, branched | latent, free-running | 0.91x |
| Twin-Cache | same model, windowed | real but full-price | 0.65x |
| **Evolutionary Tree** | **0.5B model, 6 ms** | **real** | **2.40x** |

**A small model that actually runs beats every attempt to approximate one.** Perfect
conditioning without lookahead fails; cheap lookahead without conditioning fails;
latent conditioning buys 1.03x; branching a weak drafter is worse than not branching.
The 0.5B drafter's 6 ms buys conditioning on *real* tokens, and nothing cheaper
reproduced it.

Three hardware facts govern all of it, each measured rather than assumed:

1. **Batch-1 decoding is weight-bound.** A batch-8 forward costs **1.06x** batch-1
   (81.7 → 86.7 ms). Breadth is nearly free; narrowing a batch reclaims almost nothing.
   This is why the tree's 3.1x compute saving yields ~1% of wall clock, and why
   Twin-Cache's cache windowing was worthless.
2. **The exchange rate is ~12:1.** A target forward is ~73 ms against a 0.5B draft
   forward at ~6 ms. Any scheme trading cheap draft compute for extra target forwards
   loses — exactly how the n-gram cascade failed.
3. **Acceptance multiplies.** A step commits the longest all-correct prefix, so one weak
   early position caps everything behind it. This single fact killed Medusa, EAGLE and
   Latent Mitosis.

---

## Correctness and method

**451 tests pass.** The guarantee is that emitted tokens are distributed exactly as the
target would have sampled them, checked at three levels.

**The mathematics.** 10,000-run Monte Carlo goodness-of-fit (per-category z-test,
Pearson χ², total variation) on `verify_tokens`, plus separate checks that accepted
tokens follow `norm(min(p,q))`, rejections follow `norm(max(0, p−q))`, and acceptance
equals `1 − TV(p,q)`. Substituting the common naive bug — resampling from `p` instead of
the residual — is caught at **20σ**.

**The loops.** At temperature 0 every engine must reproduce an *independent* no-cache
full-recompute greedy oracle, token for token. Swept across K, drafter quality,
`max_leaves`, `split_threshold`, `branches`, `survivor_fraction`, `draft_window_size`,
`block_size`, `max_iterations` and draft temperature.

**Real models.** `exactness_check.py` sweeps target precision with everything else
fixed: **fp32 5/5 identical, bf16 4/5, int8 2/5.** So the engine is bitwise exact on
real GPU models when precision allows, and residual divergence under quantization is a
property of the model, not the algorithm.

### Why aggressive tricks stay lossless

The breadth engines do things that *should* break speculative decoding — committing the
longest-accepted branch conditions on the acceptance outcome; particle resampling
destroys `q` as a proposal density; the tree's and n-gram's drafts are fully
deterministic. All are safe at target temperature 0 for one reason:

> `p` is one-hot at the target's argmax `a`. A proposal is accepted only if it
> **equals** `a`, and if rejected the residual `max(0, p − q)` is **also** one-hot at
> `a`. Either way the branch emits `a`. So every branch emits a *prefix of the same
> greedy string*, and choosing the longest prefix of one string cannot change its
> content — only how much is committed per iteration.

Tests assert this by checking the candidate leaves form a **prefix chain**. Above
temperature 0 these selections would bias generation, so those engines **refuse
`temperature > 0`** rather than silently biasing. A deterministic proposal is handled by
a one-hot `q`, which is exact rather than a fudge: acceptance becomes
`min(1, p(x)/1) = p(x)` and the residual normalises to `p(y)/(1−p(x))`, giving back
exactly `p` — verified by a 10,000-run Monte Carlo.

### Measurement discipline

Four measurement errors were caught and are documented, because each would have
produced a confidently wrong result.

**`repetition_penalty` in the baseline.** Qwen ships `repetition_penalty: 1.1` in
`generation_config.json`, and `transformers` applies it **even when `do_sample=False`**
— so an unguarded `model.generate()` "greedy baseline" is not greedy. Finding it
required falsifying two other hypotheses: int8 kernels (killed because bf16 diverged
too) and reduced precision generally (killed because fp32 diverged *at identical token
indices*, with a regime gap a thousand times too small to flip an argmax). Identical
divergence points across three precisions is not noise. Every baseline now passes
`repetition_penalty=1.0, top_k=0`.

**Circular frequency ranking.** Ranking token frequency on the evaluation corpus reports
a **10.95x** coverage ceiling instead of **5.69x**, because 1,000 tokens of generated
code hold only 351 distinct types and all fit inside a 6,000-token budget. Ranking and
evaluation corpora are kept disjoint, and a test pins the trap.

**Double-counted position t+1.** Medusa head 1 predicts position *t+1*, which
`lm_head(h_t)` already gives exactly. Crediting both inflated Medusa from **1.31x to
1.86x**. `chain_speedup` now takes accuracies starting at *t+2* and treats *t+1* as a
fixed 1.0; a regression test pins both figures.

**Length-confounded throughput.** Configurations diverge under int8 and stop at
different EOS points, and acceptance is higher early in a generation than late — which
flatters whichever run ends soonest. `--no-eos` forces every configuration to emit
exactly `--max-new-tokens`.

### Testing methodology

Every load-bearing invariant was mutation-tested. Reusable findings:

- A KV cache one position **too long** fails 22 tests; one position **too short** —
  still correct, merely wasteful — fails 1. Hence cache length is asserted by
  *equality*.
- **RoPE position bugs are invisible to output tests.** Numbering windowed-draft
  positions by cache index rather than absolutely fails exactly 1 test while all 6
  output-correctness tests pass, because verification cleans up after any draft. So
  positions are asserted white-box.
- Two independently-initialised tiny models agree on ~1% of tokens, which would make an
  equivalence test exercise only the rejection path. Test drafters are *noise-perturbed
  copies* of the target tuned to ~68% / ~26% / ~1% acceptance, with a guard test that
  fails if acceptance drifts out of the mixed range.
- **Bugs that destroy acceptance while preserving output are the recurring hazard.** The
  Latent Mitosis seed anchored on a rejected token's hidden state, giving alpha 1.3%
  with perfectly correct output; only an acceptance assertion caught it.

```bash
python -m pytest tests/ -v      # 451 passed
```

---

## Usage

Requires Python 3.12+ and a CUDA GPU with ≈10 GiB free. Blackwell cards (RTX 50-series,
sm_120) need a CUDA 12.8+ build of PyTorch.

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; bin/activate on Linux
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install numpy transformers accelerate bitsandbytes pytest
python download_models.py
```

### Interactive REPL

```bash
python cli.py
```

Tokens stream as each block is verified; every response ends with a live HUD:

````text
>>> Write a Python function that checks if a string is a palindrome.

```python
def is_palindrome(s):
    normalized_string = s.replace(" ", "").lower()
    return normalized_string == normalized_string[::-1]
```
[Alpha: 87% | Speedup: 2.1x | Tok/s: 29.2]
````

The speedup is **measured**: the CLI times a short autoregressive generation on the
target at startup (twice, discarding the first so CUDA autotuning is not billed to the
baseline) and uses that as the reference.

### Benchmarks and diagnostics

```bash
python benchmark.py --help                                  # every config is sweepable
python benchmark.py --no-eos --tree-split-thresholds 0.8     # throughput champion
python benchmark.py --no-eos --entropy-thresholds 0.65       # deployment champion

python exactness_check.py             # fp32/bf16/int8 bitwise-exactness sweep
python gate_calibration.py            # calibrate the entropy gate against acceptance
python batch_scaling.py               # is widening the batch actually free?
python benchmark_medusa_limit.py      # 15 tokens sequentially vs through M heads
python benchmark_6k_medusa_limit.py   # coverage ceiling vs mask size
python benchmark_eagle_limit.py       # latent chain accuracy; --save-head to checkpoint
python benchmark_cascade_limit.py     # logit-lens agreement by layer
python train_medusa_heads.py          # trains Medusa heads, prints the accuracy curve
```

---

## Architecture

| Path | Role |
|---|---|
| `core/verifier.py` | Leviathan modified rejection sampling. Model-free. |
| `core/engine.py` | `SpeculativeEngine` — draft/verify loop, KV rollback, dual gate. |
| `core/gates.py` | Statistical + syntactic draft short-circuit gates. |
| **`core/tree_engine.py`** | **Evolutionary Tree — dynamic KV-mitosis (champion).** |
| `core/monte_carlo_engine.py` | Fixed-width parallel branch speculation. |
| `core/particle_filter_engine.py` | Resampled particles *(falsified)*. |
| `core/hybrid_engine.py`, `core/ngram.py` | CPU n-gram fast path *(falsified)*. |
| `core/self_engine.py` | Twin-Cache self-speculation *(falsified)*. |
| `core/jacobi_engine.py` | Fixed-point relaxation *(falsified)*. |
| `core/ast_medusa.py`, `core/medusa_train.py` | Factored Medusa heads *(falsified)*. |
| `core/latent_extrapolator.py` | EAGLE latent head *(falsified)*. |
| `core/latent_tree_engine.py` | Latent Mitosis *(falsified)*. |
| `core/cascade_verifier.py` | Early-exit logit-lens measurement *(falsified)*. |
| `benchmark.py` | 10-prompt sweep across every configuration. |
| `cli.py` | Interactive streaming REPL. |

Each iteration, in all speculative engines: draft → verify in one target forward →
modified rejection sampling → roll both KV caches back to the accepted prefix. The
rollback is a cache **slice**, never a recomputation. `attention_mask` and
`cache_position` are always passed explicitly, because after a rollback the cache is
shorter than the emitted text and letting `transformers` infer the offset is the classic
source of silent off-by-one corruption.

## Notes and limitations

- **Breadth engines are greedy-only.** Monte Carlo, particle filter, tree, Jacobi and
  Latent Mitosis all refuse `temperature > 0`. The dual gate works at any temperature,
  which is why it is the deployment recommendation.
- **Batch size 1.** Batched serving needs per-sequence acceptance lengths and ragged
  rollback — a materially different design.
- **Vocabulary reconciliation.** Sibling Qwen checkpoints pad embeddings to different
  widths (0.5B/1.5B → 151936, 7B → 152064) despite sharing a tokenizer. Both logits
  tensors are sliced to the shared prefix *before* the softmax.
- **Bitwise exactness needs fp32.** With a quantized target, treat the engine as
  distribution-preserving up to that precision. Throughput is unaffected.
- **The drafter is the remaining bottleneck for breadth.** Monte Carlo spends ~23.8k
  draft rows to save ~160 target forwards; the tree cuts that 3.1x. Cheaper drafters,
  not smarter proposals, is where further work should go — every attempt at a *smarter*
  proposal is in the graveyard above.

## Reference

Yaniv Leviathan, Matan Kalman, Yossi Matias. *Fast Inference from Transformers via
Speculative Decoding.* ICML 2023.
[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
