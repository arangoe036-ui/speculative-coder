# Speculative Decoding Benchmark

- **Draft model**: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (bfloat16)
- **Target model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (8-bit, LLM.int8)
- **GPU**: NVIDIA GeForce RTX 5080 (15.9 GiB)
- **Sampling**: greedy (temperature 0)
- **Budget**: 256 new tokens (forced: EOS disabled, every run emits exactly this many), 10 prompts
- **Verification vocabulary**: 151936 tokens (shared prefix)
- **Baseline**: `model.generate()` on the 7B target alone

## Aggregate results

| Configuration                                 | Time (s) | Tok/s | Speedup |  Alpha | 7B fwd/token | Draft rows/tok | Peak VRAM | Code OK |
| --------------------------------------------- | -------: | ----: | ------: | -----: | -----------: | -------------: | --------: | ------: |
| Baseline (7B only)                            |    18.62 |  13.8 |   1.00x |     -- |         1.00 |             -- |  8.19 GiB |   10/10 |
| Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.92 |  32.9 |   2.39x |  77.1% |         0.14 |           2.98 |  9.42 GiB |    9/10 |
| Speculative Jacobi (block=5, max_iter=5)      |    18.39 |  14.0 |   1.02x | 100.0% |         0.87 |           0.00 |  8.19 GiB |    7/10 |
| Speculative Jacobi (block=10, max_iter=5)     |    19.90 |  12.9 |   0.94x |  53.6% |         0.92 |           0.00 |  8.19 GiB |    8/10 |
| Speculative Jacobi (block=16, max_iter=5)     |    20.24 |  12.7 |   0.92x |  34.2% |         0.93 |           0.00 |  8.20 GiB |    8/10 |

*Alpha is the fraction of drafted tokens accepted. `7B fwd/token` counts every forward pass of a **full-size** model per emitted token, which is the metric that makes the three architectures comparable: the two-model engines get their draft forwards from a 1.5B model (~4.4x cheaper, so excluded), while self-speculation drafts with the 7B itself, so its draft forwards cost full price and are counted. Plain decoding is 1.00 by definition; below 1.00 is a real win, above 1.00 means more full-size compute than simply decoding. `Draft rows/tok` is draft-model batch rows pushed per emitted token -- the draft-side cost. Counting draft forward *launches* would hide the difference, since every breadth engine launches once per depth step and only the widths differ.*

## Speedup by prompt

| Prompt     | Baseline tok/s | Tree s<0.8 | Jacobi B=5 | Jacobi B=10 | Jacobi B=16 |               Best |
| ---------- | -------------: | ---------: | ---------: | ----------: | ----------: | -----------------: |
| algorithm  |           13.7 |      2.82x |      1.00x |       0.99x |       0.87x | Tree s<0.8 (2.82x) |
| datastruct |           13.7 |      2.79x |      1.16x |       0.98x |       1.01x | Tree s<0.8 (2.79x) |
| parsing    |           13.8 |      1.92x |      0.97x |       0.88x |       0.91x | Tree s<0.8 (1.92x) |
| regex      |           13.6 |      1.99x |      1.03x |       1.00x |       0.96x | Tree s<0.8 (1.99x) |
| async      |           13.9 |      2.30x |      0.94x |       0.87x |       0.87x | Tree s<0.8 (2.30x) |
| generator  |           13.7 |      2.53x |      1.06x |       0.95x |       0.88x | Tree s<0.8 (2.53x) |
| decorator  |           13.7 |      1.99x |      1.05x |       1.00x |       0.97x | Tree s<0.8 (1.99x) |
| numeric    |           13.8 |      2.65x |      1.00x |       0.87x |       0.91x | Tree s<0.8 (2.65x) |
| oop        |           13.8 |      2.57x |      1.01x |       0.95x |       0.93x | Tree s<0.8 (2.57x) |
| errors     |           13.9 |      2.37x |      0.95x |       0.91x |       0.91x | Tree s<0.8 (2.37x) |

## Acceptance rate (alpha) by prompt

| Prompt     | Tree s<0.8 | Jacobi B=5 | Jacobi B=10 | Jacobi B=16 |
| ---------- | ---------: | ---------: | ----------: | ----------: |
| algorithm  |      91.1% |     100.0% |       54.5% |       32.3% |
| datastruct |      87.5% |     100.0% |       55.6% |       37.0% |
| parsing    |      60.2% |     100.0% |       52.1% |       34.1% |
| regex      |      61.6% |     100.0% |       54.9% |       35.2% |
| async      |      75.3% |     100.0% |       51.0% |       32.6% |
| generator  |      82.9% |     100.0% |       54.4% |       33.1% |
| decorator  |      63.1% |     100.0% |       56.1% |       35.7% |
| numeric    |      88.2% |     100.0% |       50.9% |       33.7% |
| oop        |      83.5% |     100.0% |       53.7% |       34.3% |
| errors     |      77.8% |     100.0% |       52.6% |       34.1% |

## Jacobi convergence

Every relaxation sweep costs one full target forward, so `sweeps/block` against `tokens/block` is the entire economics: the method wins only when a block freezes in fewer sweeps than it yields tokens. `Converged` is the share of blocks that reached a true fixed point rather than hitting the iteration cap. `Seeds` splits the initial guess between the n-gram lookup and the fallback of repeating the last token.

| Config   | Blocks | Sweeps/block | Tokens/block | Tok/fwd | Converged | n-gram seeds | Tok/s | Speedup |
| -------- | -----: | -----------: | -----------: | ------: | --------: | -----------: | ----: | ------: |
| block=5  |    485 |         4.59 |         5.28 |    1.15 |       32% |          24% |  14.0 |   1.02x |
| block=10 |    484 |         4.87 |         5.29 |    1.09 |        4% |          23% |  12.9 |   0.94x |
| block=16 |    483 |         4.94 |         5.30 |    1.08 |        1% |          19% |  12.7 |   0.92x |

## Evolutionary tree telemetry

`Leaves` is the mean width the tree grew to, against a ceiling of 8. `Draft rows` totals the batch rows pushed through the draft model, with the fixed-width Monte Carlo figure alongside for comparison -- that reduction is the design's central claim. `Best` and `Gain` carry the same meaning as in the breadth table: the accepted count of the winning leaf, and how much taking the max over leaves buys.

| Config         | Iters | Leaves | Splits | Draft rows | Rows/tok | Best |  Gain | Tok/fwd | Tok/s | Speedup |
| -------------- | ----: | -----: | -----: | ---------: | -------: | ---: | ----: | ------: | ----: | ------: |
| Tree split<0.8 |   367 |   4.72 |   1401 |       7620 |     2.98 | 6.12 | +2.16 |    7.09 |  32.9 |   2.39x |

## Full results

| Prompt     |                                 Configuration | Time (s) | Tokens | Tok/s | Speedup |  Alpha | Fwd | Tok/fwd | Peak VRAM |  Code | EOS | Diverges @ |
| ---------- | --------------------------------------------: | -------: | -----: | ----: | ------: | -----: | --: | ------: | --------: | ----: | --: | ---------: |
| algorithm  |                            Baseline (7B only) |    18.71 |    256 |  13.7 |   1.00x |     -- | 256 |    1.00 |      8.18 |  PASS |  no |         -- |
| algorithm  | Evolutionary Tree (leaves<=8, K=8, split<0.8) |     6.64 |    256 |  38.6 |   2.82x |  91.1% |  31 |    8.26 |      9.39 |  PASS |  no |  identical |
| algorithm  |      Speculative Jacobi (block=5, max_iter=5) |    18.66 |    256 |  13.7 |   1.00x | 100.0% | 226 |    1.13 |      8.18 |  PASS |  no |    tok 224 |
| algorithm  |     Speculative Jacobi (block=10, max_iter=5) |    18.88 |    256 |  13.6 |   0.99x |  54.5% | 224 |    1.14 |      8.18 |  PASS |  no |  identical |
| algorithm  |     Speculative Jacobi (block=16, max_iter=5) |    21.40 |    256 |  12.0 |   0.87x |  32.3% | 252 |    1.02 |      8.18 |  PASS |  no |  identical |
| datastruct |                            Baseline (7B only) |    18.73 |    256 |  13.7 |   1.00x |     -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| datastruct | Evolutionary Tree (leaves<=8, K=8, split<0.8) |     6.72 |    256 |  38.1 |   2.79x |  87.5% |  32 |    8.00 |      9.30 |  PASS |  no |  identical |
| datastruct |      Speculative Jacobi (block=5, max_iter=5) |    16.19 |    256 |  15.8 |   1.16x | 100.0% | 196 |    1.31 |      8.19 |  PASS |  no |  identical |
| datastruct |     Speculative Jacobi (block=10, max_iter=5) |    19.13 |    256 |  13.4 |   0.98x |  55.6% | 228 |    1.12 |      8.19 |  PASS |  no |  identical |
| datastruct |     Speculative Jacobi (block=16, max_iter=5) |    18.49 |    256 |  13.8 |   1.01x |  37.0% | 218 |    1.17 |      8.19 |  PASS |  no |  identical |
| parsing    |                            Baseline (7B only) |    18.49 |    256 |  13.8 |   1.00x |     -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| parsing    | Evolutionary Tree (leaves<=8, K=8, split<0.8) |     9.63 |    256 |  26.6 |   1.92x |  60.2% |  44 |    5.82 |      9.42 |  PASS |  no |     tok 19 |
| parsing    |      Speculative Jacobi (block=5, max_iter=5) |    19.00 |    256 |  13.5 |   0.97x | 100.0% | 231 |    1.11 |      8.19 |  PASS |  no |     tok 20 |
| parsing    |     Speculative Jacobi (block=10, max_iter=5) |    20.98 |    256 |  12.2 |   0.88x |  52.1% | 248 |    1.03 |      8.19 | TRUNC |  no |     tok 19 |
| parsing    |     Speculative Jacobi (block=16, max_iter=5) |    20.41 |    256 |  12.5 |   0.91x |  34.1% | 240 |    1.07 |      8.19 |  PASS |  no |     tok 19 |
| regex      |                            Baseline (7B only) |    18.77 |    256 |  13.6 |   1.00x |     -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| regex      | Evolutionary Tree (leaves<=8, K=8, split<0.8) |     9.42 |    256 |  27.2 |   1.99x |  61.6% |  44 |    5.82 |      9.41 | TRUNC |  no |     tok 97 |
| regex      |      Speculative Jacobi (block=5, max_iter=5) |    18.30 |    256 |  14.0 |   1.03x | 100.0% | 221 |    1.16 |      8.19 |  PASS |  no |     tok 97 |
| regex      |     Speculative Jacobi (block=10, max_iter=5) |    18.86 |    256 |  13.6 |   1.00x |  54.9% | 222 |    1.15 |      8.19 |  PASS |  no |    tok 133 |
| regex      |     Speculative Jacobi (block=16, max_iter=5) |    19.52 |    256 |  13.1 |   0.96x |  35.2% | 230 |    1.11 |      8.19 |  PASS |  no |     tok 76 |
| async      |                            Baseline (7B only) |    18.44 |    256 |  13.9 |   1.00x |     -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| async      | Evolutionary Tree (leaves<=8, K=8, split<0.8) |     8.03 |    256 |  31.9 |   2.30x |  75.3% |  37 |    6.92 |      9.42 |  PASS |  no |  identical |
| async      |      Speculative Jacobi (block=5, max_iter=5) |    19.69 |    256 |  13.0 |   0.94x | 100.0% | 239 |    1.07 |      8.19 |  PASS |  no |    tok 159 |
| async      |     Speculative Jacobi (block=10, max_iter=5) |    21.29 |    256 |  12.0 |   0.87x |  51.0% | 253 |    1.01 |      8.19 |  PASS |  no |    tok 129 |
| async      |     Speculative Jacobi (block=16, max_iter=5) |    21.18 |    256 |  12.1 |   0.87x |  32.6% | 251 |    1.02 |      8.19 |  PASS |  no |    tok 144 |
| generator  |                            Baseline (7B only) |    18.68 |    256 |  13.7 |   1.00x |     -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| generator  | Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.37 |    256 |  34.7 |   2.53x |  82.9% |  34 |    7.53 |      9.41 |  PASS |  no |     tok 16 |
| generator  |      Speculative Jacobi (block=5, max_iter=5) |    17.62 |    256 |  14.5 |   1.06x | 100.0% | 213 |    1.20 |      8.19 | TRUNC |  no |      tok 3 |
| generator  |     Speculative Jacobi (block=10, max_iter=5) |    19.66 |    256 |  13.0 |   0.95x |  54.4% | 235 |    1.09 |      8.19 |  PASS |  no |     tok 15 |
| generator  |     Speculative Jacobi (block=16, max_iter=5) |    21.15 |    256 |  12.1 |   0.88x |  33.1% | 247 |    1.04 |      8.20 |  PASS |  no |      tok 3 |
| decorator  |                            Baseline (7B only) |    18.69 |    256 |  13.7 |   1.00x |     -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| decorator  | Evolutionary Tree (leaves<=8, K=8, split<0.8) |     9.38 |    256 |  27.3 |   1.99x |  63.1% |  43 |    5.95 |      9.42 |  PASS |  no |     tok 48 |
| decorator  |      Speculative Jacobi (block=5, max_iter=5) |    17.85 |    256 |  14.3 |   1.05x | 100.0% | 215 |    1.19 |      8.19 | TRUNC |  no |     tok 42 |
| decorator  |     Speculative Jacobi (block=10, max_iter=5) |    18.72 |    256 |  13.7 |   1.00x |  56.1% | 221 |    1.16 |      8.19 | TRUNC |  no |     tok 42 |
| decorator  |     Speculative Jacobi (block=16, max_iter=5) |    19.37 |    256 |  13.2 |   0.97x |  35.7% | 226 |    1.13 |      8.19 | TRUNC |  no |     tok 42 |
| numeric    |                            Baseline (7B only) |    18.61 |    256 |  13.8 |   1.00x |     -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| numeric    | Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.02 |    256 |  36.5 |   2.65x |  88.2% |  32 |    8.00 |      9.42 |  PASS |  no |  identical |
| numeric    |      Speculative Jacobi (block=5, max_iter=5) |    18.69 |    256 |  13.7 |   1.00x | 100.0% | 226 |    1.13 |      8.19 | TRUNC |  no |     tok 13 |
| numeric    |     Speculative Jacobi (block=10, max_iter=5) |    21.43 |    256 |  11.9 |   0.87x |  50.9% | 254 |    1.01 |      8.19 |  PASS |  no |    tok 215 |
| numeric    |     Speculative Jacobi (block=16, max_iter=5) |    20.56 |    256 |  12.5 |   0.91x |  33.7% | 243 |    1.05 |      8.19 | TRUNC |  no |     tok 12 |
| oop        |                            Baseline (7B only) |    18.59 |    256 |  13.8 |   1.00x |     -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| oop        | Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.25 |    256 |  35.3 |   2.57x |  83.5% |  34 |    7.53 |      9.40 |  PASS |  no |     tok 52 |
| oop        |      Speculative Jacobi (block=5, max_iter=5) |    18.42 |    256 |  13.9 |   1.01x | 100.0% | 223 |    1.15 |      8.19 |  PASS |  no |    tok 208 |
| oop        |     Speculative Jacobi (block=10, max_iter=5) |    19.67 |    256 |  13.0 |   0.95x |  53.7% | 233 |    1.10 |      8.19 |  PASS |  no |     tok 77 |
| oop        |     Speculative Jacobi (block=16, max_iter=5) |    20.10 |    256 |  12.7 |   0.93x |  34.3% | 237 |    1.08 |      8.19 |  PASS |  no |     tok 52 |
| errors     |                            Baseline (7B only) |    18.47 |    256 |  13.9 |   1.00x |     -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| errors     | Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.79 |    256 |  32.9 |   2.37x |  77.8% |  36 |    7.11 |      9.42 |  PASS |  no |     tok 66 |
| errors     |      Speculative Jacobi (block=5, max_iter=5) |    19.45 |    256 |  13.2 |   0.95x | 100.0% | 237 |    1.08 |      8.19 |  PASS |  no |     tok 66 |
| errors     |     Speculative Jacobi (block=10, max_iter=5) |    20.35 |    256 |  12.6 |   0.91x |  52.6% | 241 |    1.06 |      8.19 |  PASS |  no |     tok 66 |
| errors     |     Speculative Jacobi (block=16, max_iter=5) |    20.24 |    256 |  12.6 |   0.91x |  34.1% | 240 |    1.07 |      8.19 |  PASS |  no |     tok 66 |

## Summary

- **Best configuration**: Evolutionary Tree (leaves<=8, K=8, split<0.8) at **2.39x** the baseline
- **Baseline throughput**: 13.8 tok/s
- **Mean acceptance rate**: 66.2%
- **Peak VRAM**: 9.42 GiB of 15.9 GiB
- **Generated code parses**: 42/50 runs (8 truncated by the token budget, not scored)
- **Output identical to baseline**: 9/40 greedy runs

## Why greedy output diverges from the baseline

Speculative decoding is mathematically lossless, so a greedy run should reproduce the baseline token for token. It does here only up to the target's numerical precision, and the 8-bit target used above is well past the point where that holds. This is a property of the quantized model, not of the algorithm.

The evidence is a precision sweep (`exactness_check.py`) holding the draft, the target weights, the engine and the prompts fixed, varying only the target's precision:

| Target precision | Identical to baseline | Mean regime gap | Tightest top-2 margin |
| ---------------- | --------------------: | --------------: | --------------------: |
| float32          |                   5/5 |         1.6e-05 |               8.6e-02 |
| bfloat16         |                   4/5 |         2.0e-01 |               1.3e-01 |
| 8-bit (LLM.int8) |                   2/5 |         7.5e-01 |               9.4e-02 |

*In fp32 the engine is bitwise exact on real GPU models. Exactness degrades monotonically as precision drops, and the reason is visible in the last two columns: the proof assumes the target assigns a position the same probabilities however it is evaluated, but a baseline scores one token on a KV cache while the engine scores a K+1-wide block. Those agree only to the precision's resolution. Once that gap exceeds the gap between the top two logits, a near-tie is decided by rounding, and one flipped token changes everything after it.*

One methodological trap is worth recording, because it produced divergence that looked exactly like an engine bug. Qwen ships `repetition_penalty: 1.1` in `generation_config.json`, and transformers applies it as a LogitsProcessor **even when `do_sample=False`**. An unguarded `model.generate()` baseline is therefore not greedy, and diverged at identical token indices in fp32, bf16 and int8 alike -- which is what gave it away, since numerical noise does not reproduce itself across precisions. The baseline above passes `repetition_penalty=1.0, top_k=0` to match the engine's sampling rule.

Measured on this target directly, with the engine not involved at all:

| Measurement                                     |       Value |
| ----------------------------------------------- | ----------: |
| Max abs logit delta (same position, two shapes) |       2.211 |
| Mean abs logit delta                            |       0.455 |
| Median top-2 logit margin                       |       9.430 |
| Min top-2 logit margin                          |       0.250 |
| Greedy argmax flip rate                         | 0.0% (0/64) |
| Argmax landing in truncated vocab tail          |           0 |
| Positions sampled (generated)                   |          64 |

*Read this carefully rather than as a closed case. On its own greedy continuation the model is usually very confident (median margin 9.43, so the 0.45 mean perturbation is only 5% of it) and no argmax flip appeared in 64 sampled positions. What the sample does establish is that the mechanism is available: the tightest margin seen was 0.25, below the 2.21 worst-case delta, so a close call can be decided by numerical noise, and one flipped token changes everything after it.*

*It does not establish the magnitude. Flips this rare in the sampled regime do not obviously account for divergence starting as early as it does, and the sampled regime is not quite the engine's: this compares one wide forward against per-position recomputation, whereas the engine scores a K+1-wide block on top of a rolled-back KV cache, a third numerical path. The attribution to quantized kernels rests on the ruled-out alternative below plus the fp32 equivalence tests, not on this flip count. See exactness_check.py for the controlled unquantized comparison.*

*The competing explanation is ruled out: the engine truncates both models' logits to a shared vocabulary prefix, but the target's argmax never landed in the truncated tail, so truncation is not causing the divergence.*

- Divergence begins at token **72** on average (earliest 3, latest 224) of 256 generated.
- Both continuations remain valid Python; the Code column is scored independently of whether the wording matched.

The practical consequence: with a reduced-precision target, treat speculative decoding as distribution-preserving up to that precision, not as a bitwise-identical drop-in. The speedups above are unaffected -- they are throughput measurements, and both arms decode the same kind of text at the same budget.
