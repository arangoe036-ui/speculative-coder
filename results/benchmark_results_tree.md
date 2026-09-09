# Speculative Decoding Benchmark

- **Draft model**: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (bfloat16)
- **Target model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (8-bit, LLM.int8)
- **GPU**: NVIDIA GeForce RTX 5080 (15.9 GiB)
- **Sampling**: greedy (temperature 0)
- **Budget**: 256 new tokens (forced: EOS disabled, every run emits exactly this many), 10 prompts
- **Verification vocabulary**: 151936 tokens (shared prefix)
- **Baseline**: `model.generate()` on the 7B target alone

## Aggregate results

| Configuration                                  | Time (s) | Tok/s | Speedup | Alpha | 7B fwd/token | Draft rows/tok | Peak VRAM | Code OK |
| ---------------------------------------------- | -------: | ----: | ------: | ----: | -----------: | -------------: | --------: | ------: |
| Baseline (7B only)                             |    18.59 |  13.8 |   1.00x |    -- |         1.00 |             -- |  8.19 GiB |   10/10 |
| Adaptive Dual-Gate (Max K=8, t=0.65)           |     8.54 |  30.4 |   2.21x | 78.4% |         0.22 |           1.01 |  9.12 GiB |   10/10 |
| Monte Carlo (B=8, K=8, draft T=1.2)            |     8.13 |  32.0 |   2.32x | 74.9% |         0.15 |           9.32 |  9.39 GiB |   10/10 |
| Evolutionary Tree (leaves<=8, K=8, split<0.8)  |     7.92 |  32.9 |   2.39x | 77.1% |         0.14 |           2.98 |  9.42 GiB |    9/10 |
| Evolutionary Tree (leaves<=8, K=8, split<0.95) |     8.57 |  31.7 |   2.30x | 74.9% |         0.15 |           4.45 |  9.42 GiB |   10/10 |

*Alpha is the fraction of drafted tokens accepted. `7B fwd/token` counts every forward pass of a **full-size** model per emitted token, which is the metric that makes the three architectures comparable: the two-model engines get their draft forwards from a 1.5B model (~4.4x cheaper, so excluded), while self-speculation drafts with the 7B itself, so its draft forwards cost full price and are counted. Plain decoding is 1.00 by definition; below 1.00 is a real win, above 1.00 means more full-size compute than simply decoding. `Draft rows/tok` is draft-model batch rows pushed per emitted token -- the draft-side cost. Counting draft forward *launches* would hide the difference, since every breadth engine launches once per depth step and only the widths differ.*

## Speedup by prompt

| Prompt     | Baseline tok/s | Gate t=0.65 | MC B8/K8 | Tree s<0.8 | Tree s<0.95 |                Best |
| ---------- | -------------: | ----------: | -------: | ---------: | ----------: | ------------------: |
| algorithm  |           13.8 |       2.52x |    2.54x |      2.80x |       2.78x |  Tree s<0.8 (2.80x) |
| datastruct |           13.7 |       2.64x |    2.82x |      2.77x |       3.02x | Tree s<0.95 (3.02x) |
| parsing    |           13.8 |       1.75x |    1.79x |      1.93x |       2.00x | Tree s<0.95 (2.00x) |
| regex      |           13.6 |       1.88x |    1.92x |      1.98x |       1.74x |  Tree s<0.8 (1.98x) |
| async      |           13.9 |       2.07x |    2.18x |      2.30x |       2.32x | Tree s<0.95 (2.32x) |
| generator  |           13.7 |       2.33x |    2.40x |      2.53x |       1.29x |  Tree s<0.8 (2.53x) |
| decorator  |           13.7 |       2.17x |    2.32x |      1.99x |       2.17x |    MC B8/K8 (2.32x) |
| numeric    |           13.8 |       2.12x |    2.35x |      2.68x |       2.61x |  Tree s<0.8 (2.68x) |
| oop        |           13.8 |       2.47x |    2.60x |      2.56x |       2.78x | Tree s<0.95 (2.78x) |
| errors     |           13.9 |       2.14x |    2.31x |      2.39x |       2.32x |  Tree s<0.8 (2.39x) |

## Acceptance rate (alpha) by prompt

| Prompt     | Gate t=0.65 | MC B8/K8 | Tree s<0.8 | Tree s<0.95 |
| ---------- | ----------: | -------: | ---------: | ----------: |
| algorithm  |       83.4% |    82.6% |      91.1% |       91.1% |
| datastruct |       85.2% |    91.9% |      87.5% |       97.8% |
| parsing    |       63.5% |    55.4% |      60.2% |       64.8% |
| regex      |       65.6% |    58.9% |      61.6% |       53.1% |
| async      |       79.4% |    69.6% |      75.3% |       77.7% |
| generator  |       83.1% |    77.8% |      82.9% |       36.1% |
| decorator  |       79.8% |    75.8% |      63.1% |       70.6% |
| numeric    |       76.4% |    76.4% |      88.2% |       86.8% |
| oop        |       86.9% |    85.5% |      83.5% |       93.8% |
| errors     |       80.7% |    75.1% |      77.8% |       77.0% |

## Dual-gate telemetry

Draft ceiling 8 tokens. The statistical gate fires when the draft's top-1 probability (read from the unwarped softmax) falls below the threshold; the syntactic gate fires on an unrecoverable bracket state inside fenced code. `Suppressed` counts fatal-looking bracket states seen in prose, where the syntactic gate deliberately stays silent because enumerations like "1)" are not syntax errors.

### Threshold t = 0.65

| Prompt     | Iters | Mean draft len | Statistical | Syntactic | Ungated | Gated % | Suppressed | Alpha | Speedup |
| ---------- | ----: | -------------: | ----------: | --------: | ------: | ------: | ---------: | ----: | ------: |
| algorithm  |    41 |           6.32 |          17 |         0 |      24 |     41% |          0 | 83.4% |   2.52x |
| datastruct |    38 |           6.74 |           9 |         0 |      29 |     24% |          0 | 85.2% |   2.64x |
| parsing    |    76 |           3.75 |          61 |         0 |      15 |     80% |          0 | 63.5% |   1.75x |
| regex      |    68 |           4.24 |          48 |         0 |      20 |     71% |          0 | 65.6% |   1.88x |
| async      |    64 |           3.80 |          51 |         0 |      13 |     80% |          0 | 79.4% |   2.07x |
| generator  |    51 |           4.86 |          34 |         0 |      17 |     67% |          0 | 83.1% |   2.33x |
| decorator  |    56 |           4.50 |          42 |         0 |      14 |     75% |          0 | 79.8% |   2.17x |
| numeric    |    59 |           4.37 |          46 |         0 |      13 |     78% |          0 | 76.4% |   2.12x |
| oop        |    45 |           5.42 |          27 |         0 |      18 |     60% |          0 | 86.9% |   2.47x |
| errors     |    60 |           4.07 |          49 |         0 |      11 |     82% |          0 | 80.7% |   2.14x |
| **total**  |   558 |           4.81 |         384 |         0 |     174 |     69% |          0 | 78.4% |   2.21x |

## Breadth telemetry: Monte Carlo vs Particle Filter

`Best` is the accepted-token count of the winning branch; `Single` is the mean across all branches in the same iteration -- what one branch would have achieved on the same draft samples. `Gain` is the difference, i.e. the extra tokens per iteration that breadth actually buys. `Wins` shows how often each branch index won; a degenerate spread would mean the branches are not diverging. `Distinct` is how many *different* drafts reached verification per iteration, which is the diagnostic for whether resampling has collapsed the population onto one path.

| Config      | Iters | Best | Single |  Gain | Distinct | Tok/fwd | Tok/s | Speedup |                                Win spread |
| ----------- | ----: | ---: | -----: | ----: | -------: | ------: | ----: | ------: | ----------------------------------------: |
| MC B=8, K=8 |   378 | 5.91 |   3.65 | +2.26 |     5.57 |    6.88 |  32.0 |   2.32x | 0:44% 1:17% 2:6% 3:7% 4:6% 5:5% 6:6% 7:5% |

## Evolutionary tree telemetry

`Leaves` is the mean width the tree grew to, against a ceiling of 8. `Draft rows` totals the batch rows pushed through the draft model, with the fixed-width Monte Carlo figure alongside for comparison -- that reduction is the design's central claim. `Best` and `Gain` carry the same meaning as in the breadth table: the accepted count of the winning leaf, and how much taking the max over leaves buys.

| Config               | Iters | Leaves | Splits | Draft rows | Rows/tok | Best |  Gain | Tok/fwd | Tok/s | Speedup |
| -------------------- | ----: | -----: | -----: | ---------: | -------: | ---: | ----: | ------: | ----: | ------: |
| Tree split<0.8       |   367 |   4.72 |   1401 |       7620 |     2.98 | 6.12 | +2.16 |    7.09 |  32.9 |   2.39x |
| Tree split<0.95      |   394 |   6.25 |   2148 |      11386 |     4.45 | 5.91 | +2.78 |    6.88 |  31.7 |   2.30x |
| MC B=8 (fixed width) |   378 |   8.00 |      0 |      23848 |     9.32 | 5.91 | +2.26 |    6.88 |  32.0 |   2.32x |

## Full results

| Prompt     |                                  Configuration | Time (s) | Tokens | Tok/s | Speedup | Alpha | Fwd | Tok/fwd | Peak VRAM |  Code | EOS | Diverges @ |
| ---------- | ---------------------------------------------: | -------: | -----: | ----: | ------: | ----: | --: | ------: | --------: | ----: | --: | ---------: |
| algorithm  |                             Baseline (7B only) |    18.58 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.18 |  PASS |  no |         -- |
| algorithm  |           Adaptive Dual-Gate (Max K=8, t=0.65) |     7.37 |    256 |  34.7 |   2.52x | 83.4% |  41 |    6.24 |      9.11 |  PASS |  no |  identical |
| algorithm  |            Monte Carlo (B=8, K=8, draft T=1.2) |     7.31 |    256 |  35.0 |   2.54x | 82.6% |  34 |    7.53 |      9.38 |  PASS |  no |  identical |
| algorithm  |  Evolutionary Tree (leaves<=8, K=8, split<0.8) |     6.64 |    256 |  38.6 |   2.80x | 91.1% |  31 |    8.26 |      9.39 |  PASS |  no |  identical |
| algorithm  | Evolutionary Tree (leaves<=8, K=8, split<0.95) |     6.68 |    256 |  38.3 |   2.78x | 91.1% |  31 |    8.26 |      9.41 |  PASS |  no |  identical |
| datastruct |                             Baseline (7B only) |    18.74 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| datastruct |           Adaptive Dual-Gate (Max K=8, t=0.65) |     7.10 |    256 |  36.1 |   2.64x | 85.2% |  38 |    6.74 |      9.11 |  PASS |  no |  identical |
| datastruct |            Monte Carlo (B=8, K=8, draft T=1.2) |     6.64 |    256 |  38.6 |   2.82x | 91.9% |  31 |    8.26 |      9.38 |  PASS |  no |  identical |
| datastruct |  Evolutionary Tree (leaves<=8, K=8, split<0.8) |     6.78 |    256 |  37.8 |   2.77x | 87.5% |  32 |    8.00 |      9.30 |  PASS |  no |  identical |
| datastruct | Evolutionary Tree (leaves<=8, K=8, split<0.95) |     6.20 |    256 |  41.3 |   3.02x | 97.8% |  29 |    8.83 |      9.35 |  PASS |  no |  identical |
| parsing    |                             Baseline (7B only) |    18.54 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| parsing    |           Adaptive Dual-Gate (Max K=8, t=0.65) |    10.59 |    256 |  24.2 |   1.75x | 63.5% |  76 |    3.37 |      9.11 |  PASS |  no |     tok 19 |
| parsing    |            Monte Carlo (B=8, K=8, draft T=1.2) |    10.37 |    256 |  24.7 |   1.79x | 55.4% |  48 |    5.33 |      9.39 |  PASS |  no |    tok 158 |
| parsing    |  Evolutionary Tree (leaves<=8, K=8, split<0.8) |     9.61 |    256 |  26.7 |   1.93x | 60.2% |  44 |    5.82 |      9.42 |  PASS |  no |     tok 19 |
| parsing    | Evolutionary Tree (leaves<=8, K=8, split<0.95) |     9.28 |    256 |  27.6 |   2.00x | 64.8% |  42 |    6.10 |      9.42 |  PASS |  no |     tok 19 |
| regex      |                             Baseline (7B only) |    18.76 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| regex      |           Adaptive Dual-Gate (Max K=8, t=0.65) |    10.00 |    256 |  25.6 |   1.88x | 65.6% |  68 |    3.76 |      9.11 |  PASS |  no |     tok 97 |
| regex      |            Monte Carlo (B=8, K=8, draft T=1.2) |     9.76 |    256 |  26.2 |   1.92x | 58.9% |  45 |    5.69 |      9.39 |  PASS |  no |     tok 76 |
| regex      |  Evolutionary Tree (leaves<=8, K=8, split<0.8) |     9.46 |    256 |  27.1 |   1.98x | 61.6% |  44 |    5.82 |      9.41 | TRUNC |  no |     tok 97 |
| regex      | Evolutionary Tree (leaves<=8, K=8, split<0.95) |    10.77 |    256 |  23.8 |   1.74x | 53.1% |  49 |    5.22 |      9.42 |  PASS |  no |     tok 97 |
| async      |                             Baseline (7B only) |    18.43 |    256 |  13.9 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| async      |           Adaptive Dual-Gate (Max K=8, t=0.65) |     8.90 |    256 |  28.8 |   2.07x | 79.4% |  64 |    4.00 |      9.12 |  PASS |  no |    tok 129 |
| async      |            Monte Carlo (B=8, K=8, draft T=1.2) |     8.44 |    256 |  30.3 |   2.18x | 69.6% |  39 |    6.56 |      9.38 |  PASS |  no |    tok 223 |
| async      |  Evolutionary Tree (leaves<=8, K=8, split<0.8) |     8.02 |    256 |  31.9 |   2.30x | 75.3% |  37 |    6.92 |      9.42 |  PASS |  no |  identical |
| async      | Evolutionary Tree (leaves<=8, K=8, split<0.95) |     7.96 |    256 |  32.2 |   2.32x | 77.7% |  37 |    6.92 |      9.42 |  PASS |  no |    tok 129 |
| generator  |                             Baseline (7B only) |    18.65 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| generator  |           Adaptive Dual-Gate (Max K=8, t=0.65) |     8.02 |    256 |  31.9 |   2.33x | 83.1% |  51 |    5.02 |      9.12 |  PASS |  no |    tok 212 |
| generator  |            Monte Carlo (B=8, K=8, draft T=1.2) |     7.76 |    256 |  33.0 |   2.40x | 77.8% |  36 |    7.11 |      9.39 |  PASS |  no |  identical |
| generator  |  Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.36 |    256 |  34.8 |   2.53x | 82.9% |  34 |    7.53 |      9.41 |  PASS |  no |     tok 16 |
| generator  | Evolutionary Tree (leaves<=8, K=8, split<0.95) |    14.51 |    256 |  17.6 |   1.29x | 36.1% |  67 |    3.82 |      9.42 |  PASS |  no |      tok 3 |
| decorator  |                             Baseline (7B only) |    18.62 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| decorator  |           Adaptive Dual-Gate (Max K=8, t=0.65) |     8.58 |    256 |  29.9 |   2.17x | 79.8% |  56 |    4.57 |      9.12 |  PASS |  no |    tok 102 |
| decorator  |            Monte Carlo (B=8, K=8, draft T=1.2) |     8.04 |    256 |  31.8 |   2.32x | 75.8% |  38 |    6.74 |      9.38 |  PASS |  no |    tok 102 |
| decorator  |  Evolutionary Tree (leaves<=8, K=8, split<0.8) |     9.37 |    256 |  27.3 |   1.99x | 63.1% |  43 |    5.95 |      9.42 |  PASS |  no |     tok 48 |
| decorator  | Evolutionary Tree (leaves<=8, K=8, split<0.95) |     8.60 |    256 |  29.8 |   2.17x | 70.6% |  39 |    6.56 |      9.42 |  PASS |  no |    tok 102 |
| numeric    |                             Baseline (7B only) |    18.62 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| numeric    |           Adaptive Dual-Gate (Max K=8, t=0.65) |     8.78 |    256 |  29.2 |   2.12x | 76.4% |  59 |    4.34 |      9.12 |  PASS |  no |     tok 12 |
| numeric    |            Monte Carlo (B=8, K=8, draft T=1.2) |     7.91 |    256 |  32.4 |   2.35x | 76.4% |  37 |    6.92 |      9.38 |  PASS |  no |    tok 215 |
| numeric    |  Evolutionary Tree (leaves<=8, K=8, split<0.8) |     6.95 |    256 |  36.8 |   2.68x | 88.2% |  32 |    8.00 |      9.42 |  PASS |  no |  identical |
| numeric    | Evolutionary Tree (leaves<=8, K=8, split<0.95) |     7.14 |    256 |  35.9 |   2.61x | 86.8% |  33 |    7.76 |      9.40 |  PASS |  no |     tok 12 |
| oop        |                             Baseline (7B only) |    18.51 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| oop        |           Adaptive Dual-Gate (Max K=8, t=0.65) |     7.50 |    256 |  34.1 |   2.47x | 86.9% |  45 |    5.69 |      9.12 |  PASS |  no |     tok 52 |
| oop        |            Monte Carlo (B=8, K=8, draft T=1.2) |     7.12 |    256 |  35.9 |   2.60x | 85.5% |  33 |    7.76 |      9.38 |  PASS |  no |     tok 52 |
| oop        |  Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.24 |    256 |  35.4 |   2.56x | 83.5% |  34 |    7.53 |      9.40 |  PASS |  no |     tok 52 |
| oop        | Evolutionary Tree (leaves<=8, K=8, split<0.95) |     6.65 |    256 |  38.5 |   2.78x | 93.8% |  31 |    8.26 |      9.42 |  PASS |  no |  identical |
| errors     |                             Baseline (7B only) |    18.45 |    256 |  13.9 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| errors     |           Adaptive Dual-Gate (Max K=8, t=0.65) |     8.62 |    256 |  29.7 |   2.14x | 80.7% |  60 |    4.27 |      9.12 |  PASS |  no |     tok 66 |
| errors     |            Monte Carlo (B=8, K=8, draft T=1.2) |     7.98 |    256 |  32.1 |   2.31x | 75.1% |  37 |    6.92 |      9.38 |  PASS |  no |     tok 66 |
| errors     |  Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.73 |    256 |  33.1 |   2.39x | 77.8% |  36 |    7.11 |      9.42 |  PASS |  no |     tok 66 |
| errors     | Evolutionary Tree (leaves<=8, K=8, split<0.95) |     7.94 |    256 |  32.2 |   2.32x | 77.0% |  36 |    7.11 |      9.42 |  PASS |  no |     tok 66 |

## Summary

- **Best configuration**: Evolutionary Tree (leaves<=8, K=8, split<0.8) at **2.39x** the baseline
- **Baseline throughput**: 13.8 tok/s
- **Mean acceptance rate**: 76.3%
- **Peak VRAM**: 9.42 GiB of 15.9 GiB
- **Generated code parses**: 49/50 runs (1 truncated by the token budget, not scored)
- **Output identical to baseline**: 12/40 greedy runs

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

- Divergence begins at token **82** on average (earliest 3, latest 223) of 256 generated.
- Both continuations remain valid Python; the Code column is scored independently of whether the wording matched.

The practical consequence: with a reduced-precision target, treat speculative decoding as distribution-preserving up to that precision, not as a bitwise-identical drop-in. The speedups above are unaffected -- they are throughput measurements, and both arms decode the same kind of text at the same budget.
