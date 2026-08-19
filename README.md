# Speculative-Coder: High-Throughput Inference via Adaptive Entropy & Evolutionary Branching

A from-scratch PyTorch speculative decoding engine. **2.39x faster local inference
on a 7B code model, with the output distribution provably unchanged** — and a
documented record of the four ideas that didn't work and the mechanism that killed
each one.

No `vllm`, no `assistant_model=`. The rejection sampler, the generation loop, the
KV-cache rollback, and six speculation strategies are implemented and tested here.

---

## Results

Ten Python coding prompts, forced 256 tokens per run, greedy target, single
RTX 5080 (16 GB). Target `Qwen2.5-Coder-7B-Instruct` in 8-bit; draft
`Qwen2.5-Coder-0.5B-Instruct` in bfloat16.

| Configuration | Tok/s | Speedup | Alpha | 7B fwd/token | Draft rows/token | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (7B alone) | 13.8 | 1.00x | — | 1.00 | — | 8.19 GiB |
| **Adaptive Dual-Gate** *(deployment)* | 30.4 | **2.21x** | 78.4% | 0.22 | **1.01** | **9.12 GiB** |
| Monte Carlo (B=8, K=8) | 32.0 | 2.32x | 74.9% | 0.15 | 9.32 | 9.39 GiB |
| **Evolutionary Tree** *(research)* | **32.9** | **2.39x** | 77.1% | **0.14** | 2.98 | 9.42 GiB |

`Alpha` is the fraction of drafted tokens accepted. `7B fwd/token` counts forward
passes of the *expensive* model per emitted token — plain decoding is 1.00 by
definition. `Draft rows/token` counts draft-model batch rows, the draft-side cost.

Every configuration is **distribution-preserving**: emitted tokens are exactly
what the target model would have produced. That is verified, not asserted — see
[Correctness](#correctness).

---

## 1. The Baseline

`Qwen2.5-Coder-7B-Instruct` in 8-bit, decoding autoregressively at **13.8 tok/s**,
one forward pass per token. The forward is memory-bound: it reads ~8.1 GiB of
weights to produce a single token, which is the inefficiency every strategy below
attacks.

The measurement discipline matters more than it sounds. Qwen ships
`repetition_penalty: 1.1` in its `generation_config.json`, and `transformers`
applies it **even when `do_sample=False`** — so an unguarded `model.generate()`
"greedy baseline" is not greedy. Finding that took falsifying two other hypotheses
(§5). Every baseline here passes `repetition_penalty=1.0, top_k=0`.

## 2. The Deployment Champion — Adaptive Dual-Gate (2.21x)

Fixed-K speculation wastes draft compute: every token drafted after the eventual
first rejection is discarded. The dual gate abandons fixed K and stops drafting
early on either of two signals.

**Statistical gate.** Stop when the draft's top-1 probability falls below a
threshold — the draft is unsure, so the block is unlikely to survive verification.
The threshold was *calibrated*, not guessed (`gate_calibration.py`, 1,893 drafted
tokens): the draft's top-1 probability has a **median of 0.99**, so the intuitive
0.35 sits near the 1st percentile and fires on only 8% of blocks. Acceptance rises
monotonically with confidence (36% → 77% across buckets), confirming the premise,
and **τ = 0.65** cuts 33% of draft forwards while the tokens it skips had only 30%
acceptance versus 71% overall.

**Syntactic gate.** A bracket/quote state machine (`core/gates.py`) halts a draft
that closes a bracket it never opened. It fires **zero times** across 20 prompt-runs.
That is a real negative result, not a broken tracker — instrumentation confirms
1,780 chars fed, code fences entered, brackets balanced 8/8. A competent code model
drafting ≤8 tokens from a valid state simply doesn't produce unrecoverable syntax.
Kept behind a flag; not claimed as a win.

**Why this is the deployment choice.** It needs no batch memory (**+0.94 GiB** over
running the 7B alone, and 1.01 draft rows/token — the leanest of anything here), and
it is the only accelerated configuration that **works at any temperature**. All three
breadth engines are greedy-only.

One subtlety worth stealing: the gate reads the draft's **unwarped** softmax. At
temperature 0 the warped `q` is one-hot with top-1 exactly 1.0, so a gate built on it
would never fire — indistinguishable from a gate that found nothing. There's a test
asserting it fires at temperature 0.

## 3. The Research Champion — Evolutionary Tree Speculation (2.39x)

Monte Carlo speculation buys breadth by drafting B independent continuations and
committing whichever got furthest. It works (2.32x), but it pays for eight
full-width branches from the first token — including at the many positions where the
draft is completely certain what comes next.

**The mitosis mechanism.** Drafting starts as a *single* branch taking argmax tokens.
At each depth step the engine inspects each branch's top-1 probability; where it
falls **below 0.80** — where the draft is genuinely uncertain — the branch undergoes
mitosis:

- its KV cache and token history are cloned into a new batch row,
- one child takes the **top-1** token, its sibling takes the **top-2**,
- the leaf count increments, capped at `max_leaves = 8`.

Certainty costs one row; uncertainty costs two. The trunk is drafted once and breadth
is spent exactly where the draft might be wrong. All leaves are then verified against
the 7B in one batched forward, and the longest-accepted leaf is committed.

### What it achieved

| Config | Iters | Leaves | Splits | **Draft rows** | Rows/tok | Best leaf | Gain | Tok/fwd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Tree (split<0.80)** | 367 | **4.72** | 1,401 | **7,620** | **2.98** | **6.12** | +2.16 | **7.09** |
| Tree (split<0.95) | 394 | 6.25 | 2,148 | 11,386 | 4.45 | 5.91 | +2.78 | 6.88 |
| Monte Carlo B=8 (fixed) | 378 | 8.00 | 0 | **23,848** | 9.32 | 5.91 | +2.26 | 6.88 |

**A 3.1x reduction in draft compute (7,620 vs 23,848 rows) while producing strictly
superior sequences** — higher best-leaf acceptance (6.12 vs 5.91), higher alpha
(77.1% vs 74.9%), and more tokens per expensive forward (7.09 vs 6.88). It grows to
4.72 leaves on average instead of paying for 8 throughout.

Two mechanisms explain the acceptance win. The trunk is the draft's **greedy** path,
already a far better single draft than any T=1.2 sample — Monte Carlo's average
branch accepted 3.49 tokens. And splitting *only* at uncertainty places the recovery
option precisely where errors occur. `split<0.95` demonstrates the tradeoff from the
other side: it splits on positions that weren't really uncertain, paying 1.5x the
draft rows for a *lower* best leaf.

### Honest limits

**The throughput edge over Monte Carlo is within noise.** 32.9 vs 32.0 tok/s is
+2.8%, and repeated Monte Carlo runs span 30.7–32.0 (~4%). Versus the dual gate
(+8.2%, same run) the gap is solid. The tree is the **fastest measured**
configuration; it is not established as faster than Monte Carlo.

**The compute saving barely converts to wall clock, by design of the hardware.**
`batch_scaling.py` measures a batch-8 forward at **1.06x** the cost of batch-1
(81.7 → 86.7 ms) because decoding is weight-bound. Narrowing a batch reclaims
compute that was nearly free. The tree's real contribution is the *acceptance*
improvement; the 3.1x compute reduction is what would let it scale to much larger
leaf counts, not what makes it fast today.

**No tree attention mask.** Each leaf is materialised as its own cache row rather
than sharing a trunk behind a 4D mask. That costs memory (9.42 vs 9.12 GiB) but not
time, for the same reason. It also means all leaves are exactly K deep by
construction, so the verification batch is rectangular with **no padding required**.

## 4. The Falsified Graveyard

Four ideas were built, measured, and rejected. Each is kept in the repository with
its telemetry, because the mechanism that killed it is the useful part.

### Twin-Cache Self-Speculation — 0.65x *(35% slower than baseline)*

One model drafting for itself, made "cheap" by windowing its own KV cache to 64
tokens. **Killed by the batch-1 memory wall.** Cropping the cache makes draft
*attention* cheap, but attention isn't the cost: the forward reads ~8.1 GiB of
weights against ~17 MB of KV cache at 300 tokens — 0.2% of the traffic. Measured, the
windowed draft forward cost **76.4 ms against the baseline's 72.7 ms** — slightly
*more*. KV traffic wouldn't even rival weight traffic until ~152k tokens of context.
Since the draft *is* the target, K draft forwards + 1 verification buys at most K+1
tokens for K+1 full-price forwards, so forwards-per-token is ≥1.00 always, with
equality only at 100% acceptance. It measured 1.48. VRAM saving was real (**2.89 GiB**),
but it is dominated by simply decoding normally.

### N-Gram Hybrid Cascade — +1% *(within noise)*

A free CPU prompt-lookup drafter routed in front of the 0.5B model. **Killed by a
generation-vs-editing mismatch.** Prompt-lookup wins where output copies input — code
editing, RAG, summarisation. Generating from a short instruction gives it almost
nothing to copy. It is squeezed from both sides: at `m=2` it fires on 16% of blocks
with only **31.5% acceptance**; at `m=6` acceptance reaches 53.6% but it fires on 4%.
Never both. Break-even needs ~68%. It genuinely saved 18% of draft forwards, but
tokens-per-target-forward *fell* in every configuration (4.59 → 4.21–4.48), and each
extra iteration costs a 73 ms target forward against a 6 ms draft forward — a 12x
exchange rate. Trading cheap draft compute for expensive target compute loses.

### Particle Filter (Sequential Monte Carlo) — 1.98x *(12% slower than Monte Carlo)*

Prune the weakest half of draft particles each step and clone the strongest.
**Killed by diversity collapse.** The hypothesis was right: resampling *did* produce
statistically superior sequences, lifting mean per-particle acceptance from **3.49 to
4.73** (+36%). But it destroyed the thing that was paying — `Gain` fell from **+2.27
to +0.17**, and distinct candidates reaching verification fell from **5.64 to 2.10**
of 8 particles, with one branch winning 87–91% of iterations versus Monte Carlo's 44%.
Best leaf dropped 5.76 → 4.90. The resampling weights are the draft's *own* log-probs
with no observation of the target, so it is a greedy search toward the draft's mode —
exactly where a single greedy draft already goes. **Eight copies of the draft's
favourite continuation are worth less than eight different guesses**, because
acceptance is decided by the target. Overhead was not the problem: per-iteration cost
was identical (215 vs 216 ms); the entire deficit was fewer tokens committed per
iteration.

### Also measured and rejected

- **Static K sweep.** K=5 is the best fixed choice (2.08x), but the per-prompt
  optimum moves between K=5 and K=7 with acceptance — the argument for adapting K.
- **1.5B draft.** Identical throughput to the 0.5B draft (30.2 vs 30.1 tok/s) for
  **1.95 GiB more VRAM**. The smaller draft is strictly better here.

---

## Correctness

**281 tests pass.** The guarantee is that emitted tokens are distributed exactly as
the target model would have sampled them, and it is checked at three levels.

**The mathematics.** 10,000-run Monte Carlo goodness-of-fit (per-category z-test,
Pearson χ², total variation) on `verify_tokens`, plus separate checks that accepted
tokens follow `norm(min(p,q))`, rejections follow `norm(max(0, p−q))`, and acceptance
equals `1 − TV(p,q)`. Substituting the common naive bug — resampling from `p` instead
of the residual — is caught at **20σ**.

**The loop.** At temperature 0 every engine must reproduce an *independent* no-cache
full-recompute greedy oracle, token for token. Swept across K, draft quality,
`max_leaves`, `split_threshold`, `branches`, `survivor_fraction`, `draft_window_size`
and draft temperature.

**Real models.** `exactness_check.py` sweeps target precision with everything else
fixed: **fp32 gives 5/5 identical, bf16 4/5, int8 2/5**. So the engine is bitwise
exact on real GPU models when precision allows, and residual divergence under
quantization is a property of the model, not the algorithm.

### Why aggressive tricks stay lossless

The breadth engines do things that *should* break speculative decoding — selecting
the longest-accepted branch conditions on the acceptance outcome; particle resampling
destroys `q` as an honest proposal density; the tree's draft is fully deterministic.
All are safe at target temperature 0 for one specific reason:

> `p` is one-hot at the target's argmax `a`. A proposal is accepted only if it
> **equals** `a`, and if rejected the residual `max(0, p − q)` is **also** one-hot at
> `a`. Either way the branch emits `a`. So every branch emits a *prefix of the same
> greedy string*, and choosing the longest prefix of one string cannot change its
> content — only how much is committed per iteration.

Tests assert this directly by checking the candidate leaves form a **prefix chain**.
Above temperature 0 these selections would bias generation, so those engines
**refuse `temperature > 0`** rather than silently biasing.

### Testing methodology

Every load-bearing invariant was mutation-tested. Some findings:

- A KV cache one position **too long** fails 22 tests; one position **too short** —
  still correct, merely wasteful — fails 1. Hence cache length is asserted by
  *equality*, not inequality.
- Numbering windowed-draft positions by cache index rather than absolutely fails
  exactly **1** test while all 6 output-correctness tests still pass. RoPE bugs are
  invisible to output tests because verification cleans up after any draft, so
  positions are asserted white-box.
- Two independently-initialised tiny models agree on ~1% of tokens, which would make
  an equivalence test exercise only the rejection path. Draft models in tests are
  *noise-perturbed copies* of the target, tuned to ~68% / ~26% / ~1% acceptance, with
  a guard test that fails if acceptance drifts out of the mixed range.

```bash
python -m pytest tests/ -v      # 281 passed
```

---

## Usage

Requires Python 3.12+ and a CUDA GPU with ≈10 GiB free. Blackwell cards (RTX 50xx,
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
target at startup (twice, discarding the first so CUDA autotuning isn't billed to the
baseline) and uses that as the reference.

### Benchmarks and diagnostics

```bash
python benchmark.py --help                    # every configuration is sweepable
python benchmark.py --no-eos --tree-split-thresholds 0.8   # the research champion
python benchmark.py --entropy-thresholds 0.65              # the deployment champion

python exactness_check.py     # fp32/bf16/int8 bitwise-exactness sweep
python gate_calibration.py    # calibrate the entropy gate against measured acceptance
python batch_scaling.py       # measure whether widening the batch is actually free
```

`--no-eos` forces every configuration to emit exactly `--max-new-tokens`. Without it,
configurations diverge under int8 and stop at different EOS points, and acceptance is
higher early in a generation than late — which flatters whichever run ends soonest.

---

## Architecture

| Path | Role |
|---|---|
| `core/verifier.py` | Leviathan modified rejection sampling. Model-free, 15 tests. |
| `core/engine.py` | `SpeculativeEngine` — draft/verify loop, KV rollback, dual gate. |
| `core/gates.py` | Statistical + syntactic draft short-circuit gates. |
| `core/tree_engine.py` | **Evolutionary tree** — dynamic KV-cache mitosis. |
| `core/monte_carlo_engine.py` | Fixed-width parallel branch speculation. |
| `core/particle_filter_engine.py` | Resampled particles *(falsified)*. |
| `core/hybrid_engine.py`, `core/ngram.py` | CPU n-gram fast path *(falsified)*. |
| `core/self_engine.py` | Twin-Cache self-speculation *(falsified)*. |
| `benchmark.py` | 10-prompt sweep across every configuration. |
| `cli.py` | Interactive streaming REPL. |
| `exactness_check.py`, `gate_calibration.py`, `batch_scaling.py` | Diagnostics. |

Each iteration, in all engines: draft → verify in one target forward → modified
rejection sampling → roll both KV caches back to the accepted prefix. The rollback is
a cache **slice**, never a recomputation. `attention_mask` and `cache_position` are
always passed explicitly, because after a rollback the cache is shorter than the
emitted text and letting `transformers` infer the offset is the classic source of
silent off-by-one corruption.

## Notes and limitations

- **Breadth engines are greedy-only.** Monte Carlo, particle filter and tree all
  refuse `temperature > 0`. The dual gate works at any temperature.
- **Batch size 1.** Batched serving needs per-sequence acceptance lengths and ragged
  rollback — a materially different design.
- **Vocabulary reconciliation.** Sibling Qwen checkpoints pad embeddings to different
  widths (0.5B/1.5B → 151936, 7B → 152064) despite sharing a tokenizer. Both logits
  tensors are sliced to the shared prefix *before* the softmax; otherwise `p` and `q`
  index different token spaces.
- **Bitwise exactness needs fp32.** With a quantized target, treat the engine as
  distribution-preserving up to that precision. Throughput is unaffected.
- **The draft is now the bottleneck for breadth.** Monte Carlo spends ~23.8k draft
  rows to save ~160 target forwards. The tree cuts that 3.1x, which is the direction
  further work should go: cheaper drafts, not smarter proposals.

## Reference

Yaniv Leviathan, Matan Kalman, Yossi Matias. *Fast Inference from Transformers via
Speculative Decoding.* ICML 2023.
[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
