# Speculative Decoding Benchmark

- **Draft model**: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (bfloat16)
- **Target model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (8-bit, LLM.int8)
- **GPU**: NVIDIA GeForce RTX 5080 (15.9 GiB)
- **Sampling**: greedy (temperature 0)
- **Budget**: 256 new tokens (forced: EOS disabled, every run emits exactly this many), 10 prompts
- **Verification vocabulary**: 151936 tokens (shared prefix)
- **Baseline**: `model.generate()` on the 7B target alone

## Aggregate results

| Configuration                         | Time (s) | Tok/s | Speedup | Alpha | 7B fwd/token | Weights VRAM | Peak VRAM | Code OK |
| ------------------------------------- | -------: | ----: | ------: | ----: | -----------: | -----------: | --------: | ------: |
| Baseline (7B only)                    |    18.58 |  13.8 |   1.00x |    -- |         1.00 |     8.11 GiB |  8.19 GiB |   10/10 |
| Adaptive Dual-Gate (Max K=8, t=0.65)  |     8.55 |  30.4 |   2.21x | 78.4% |         0.22 |     9.05 GiB |  9.12 GiB |   10/10 |
| Monte Carlo (B=8, K=8, draft T=1.2)   |     8.56 |  31.2 |   2.26x | 72.6% |         0.15 |     9.05 GiB |  9.39 GiB |   10/10 |
| Particle Filter (B=8, K=8, keep=0.25) |     9.54 |  27.5 |   2.00x | 62.7% |         0.17 |     9.05 GiB |  9.37 GiB |   10/10 |
| Particle Filter (B=8, K=8, keep=0.5)  |     9.61 |  27.3 |   1.98x | 62.3% |         0.17 |     9.05 GiB |  9.37 GiB |   10/10 |

*Alpha is the fraction of drafted tokens accepted. `7B fwd/token` counts every forward pass of a **full-size** model per emitted token, which is the metric that makes the three architectures comparable: the two-model engines get their draft forwards from a 1.5B model (~4.4x cheaper, so excluded), while self-speculation drafts with the 7B itself, so its draft forwards cost full price and are counted. Plain decoding is 1.00 by definition; below 1.00 is a real win, above 1.00 means more full-size compute than simply decoding.*

## Speedup by prompt

| Prompt     | Baseline tok/s | Gate t=0.65 | MC B8/K8 | PF keep=0.25 | PF keep=0.5 |                Best |
| ---------- | -------------: | ----------: | -------: | -----------: | ----------: | ------------------: |
| algorithm  |           13.8 |       2.52x |    2.67x |        2.39x |       2.29x |    MC B8/K8 (2.67x) |
| datastruct |           13.7 |       2.65x |    3.01x |        2.42x |       2.49x |    MC B8/K8 (3.01x) |
| parsing    |           13.9 |       1.75x |    1.74x |        1.42x |       1.50x | Gate t=0.65 (1.75x) |
| regex      |           13.6 |       1.87x |    1.47x |        1.71x |       1.60x | Gate t=0.65 (1.87x) |
| async      |           13.9 |       2.07x |    2.07x |        2.00x |       1.68x |    MC B8/K8 (2.07x) |
| generator  |           13.8 |       2.32x |    2.46x |        2.09x |       2.17x |    MC B8/K8 (2.46x) |
| decorator  |           13.8 |       2.17x |    2.00x |        1.90x |       1.86x | Gate t=0.65 (2.17x) |
| numeric    |           13.7 |       2.12x |    2.42x |        1.81x |       2.08x |    MC B8/K8 (2.42x) |
| oop        |           13.8 |       2.47x |    2.48x |        2.32x |       2.27x |    MC B8/K8 (2.48x) |
| errors     |           13.9 |       2.13x |    2.30x |        1.91x |       1.90x |    MC B8/K8 (2.30x) |

## Acceptance rate (alpha) by prompt

| Prompt     | Gate t=0.65 | MC B8/K8 | PF keep=0.25 | PF keep=0.5 |
| ---------- | ----------: | -------: | -----------: | ----------: |
| algorithm  |       83.4% |    87.5% |        77.3% |       73.2% |
| datastruct |       85.2% |    97.8% |        77.0% |       79.6% |
| parsing    |       63.5% |    53.7% |        41.6% |       44.6% |
| regex      |       65.6% |    42.5% |        51.2% |       47.7% |
| async      |       79.4% |    66.5% |        63.1% |       51.6% |
| generator  |       83.1% |    80.1% |        66.2% |       69.3% |
| decorator  |       79.8% |    64.0% |        59.9% |       58.0% |
| numeric    |       76.4% |    77.0% |        54.9% |       65.2% |
| oop        |       86.9% |    81.0% |        74.8% |       73.5% |
| errors     |       80.7% |    75.3% |        60.7% |       60.1% |

## Dual-gate telemetry

Draft ceiling 8 tokens. The statistical gate fires when the draft's top-1 probability (read from the unwarped softmax) falls below the threshold; the syntactic gate fires on an unrecoverable bracket state inside fenced code. `Suppressed` counts fatal-looking bracket states seen in prose, where the syntactic gate deliberately stays silent because enumerations like "1)" are not syntax errors.

### Threshold t = 0.65

| Prompt     | Iters | Mean draft len | Statistical | Syntactic | Ungated | Gated % | Suppressed | Alpha | Speedup |
| ---------- | ----: | -------------: | ----------: | --------: | ------: | ------: | ---------: | ----: | ------: |
| algorithm  |    41 |           6.32 |          17 |         0 |      24 |     41% |          0 | 83.4% |   2.52x |
| datastruct |    38 |           6.74 |           9 |         0 |      29 |     24% |          0 | 85.2% |   2.65x |
| parsing    |    76 |           3.75 |          61 |         0 |      15 |     80% |          0 | 63.5% |   1.75x |
| regex      |    68 |           4.24 |          48 |         0 |      20 |     71% |          0 | 65.6% |   1.87x |
| async      |    64 |           3.80 |          51 |         0 |      13 |     80% |          0 | 79.4% |   2.07x |
| generator  |    51 |           4.86 |          34 |         0 |      17 |     67% |          0 | 83.1% |   2.32x |
| decorator  |    56 |           4.50 |          42 |         0 |      14 |     75% |          0 | 79.8% |   2.17x |
| numeric    |    59 |           4.37 |          46 |         0 |      13 |     78% |          0 | 76.4% |   2.12x |
| oop        |    45 |           5.42 |          27 |         0 |      18 |     60% |          0 | 86.9% |   2.47x |
| errors     |    60 |           4.07 |          49 |         0 |      11 |     82% |          0 | 80.7% |   2.13x |
| **total**  |   558 |           4.81 |         384 |         0 |     174 |     69% |          0 | 78.4% |   2.21x |

## Breadth telemetry: Monte Carlo vs Particle Filter

`Best` is the accepted-token count of the winning branch; `Single` is the mean across all branches in the same iteration -- what one branch would have achieved on the same draft samples. `Gain` is the difference, i.e. the extra tokens per iteration that breadth actually buys. `Wins` shows how often each branch index won; a degenerate spread would mean the branches are not diverging. `Distinct` is how many *different* drafts reached verification per iteration, which is the diagnostic for whether resampling has collapsed the population onto one path.

| Config       | Iters | Best | Single |  Gain | Distinct | Tok/fwd | Tok/s | Speedup |                                 Win spread |
| ------------ | ----: | ---: | -----: | ----: | -------: | ------: | ----: | ------: | -----------------------------------------: |
| MC B=8, K=8  |   395 | 5.76 |   3.49 | +2.27 |     5.64 |    6.74 |  31.2 |   2.26x | 0:44% 1:14% 2:10% 3:7% 4:4% 5:8% 6:5% 7:3% |
| PF keep=0.25 |   443 | 4.94 |   4.84 | +0.10 |     1.90 |    5.92 |  27.5 |   2.00x |                  0:91% 1:4% 2:2% 3:1% 4:0% |
| PF keep=0.5  |   446 | 4.90 |   4.73 | +0.17 |     2.10 |    5.88 |  27.3 |   1.98x |                  0:87% 1:7% 2:3% 3:0% 4:0% |

## Full results

| Prompt     |                         Configuration | Time (s) | Tokens | Tok/s | Speedup | Alpha | Fwd | Tok/fwd | Peak VRAM | Code | EOS | Diverges @ |
| ---------- | ------------------------------------: | -------: | -----: | ----: | ------: | ----: | --: | ------: | --------: | ---: | --: | ---------: |
| algorithm  |                    Baseline (7B only) |    18.56 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.18 | PASS |  no |         -- |
| algorithm  |  Adaptive Dual-Gate (Max K=8, t=0.65) |     7.35 |    256 |  34.8 |   2.52x | 83.4% |  41 |    6.24 |      9.11 | PASS |  no |  identical |
| algorithm  |   Monte Carlo (B=8, K=8, draft T=1.2) |     6.94 |    256 |  36.9 |   2.67x | 87.5% |  32 |    8.00 |      9.38 | PASS |  no |  identical |
| algorithm  |  Particle Filter (B=8, K=8, keep=0.5) |     8.11 |    256 |  31.6 |   2.29x | 73.2% |  38 |    6.74 |      9.25 | PASS |  no |  identical |
| algorithm  | Particle Filter (B=8, K=8, keep=0.25) |     7.75 |    256 |  33.0 |   2.39x | 77.3% |  36 |    7.11 |      9.23 | PASS |  no |  identical |
| datastruct |                    Baseline (7B only) |    18.72 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| datastruct |  Adaptive Dual-Gate (Max K=8, t=0.65) |     7.07 |    256 |  36.2 |   2.65x | 85.2% |  38 |    6.74 |      9.11 | PASS |  no |  identical |
| datastruct |   Monte Carlo (B=8, K=8, draft T=1.2) |     6.22 |    256 |  41.1 |   3.01x | 97.8% |  29 |    8.83 |      9.38 | PASS |  no |  identical |
| datastruct |  Particle Filter (B=8, K=8, keep=0.5) |     7.50 |    256 |  34.1 |   2.49x | 79.6% |  35 |    7.31 |      9.28 | PASS |  no |  identical |
| datastruct | Particle Filter (B=8, K=8, keep=0.25) |     7.75 |    256 |  33.0 |   2.42x | 77.0% |  36 |    7.11 |      9.23 | PASS |  no |  identical |
| parsing    |                    Baseline (7B only) |    18.46 |    256 |  13.9 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| parsing    |  Adaptive Dual-Gate (Max K=8, t=0.65) |    10.55 |    256 |  24.3 |   1.75x | 63.5% |  76 |    3.37 |      9.11 | PASS |  no |     tok 19 |
| parsing    |   Monte Carlo (B=8, K=8, draft T=1.2) |    10.60 |    256 |  24.1 |   1.74x | 53.7% |  49 |    5.22 |      9.38 | PASS |  no |     tok 19 |
| parsing    |  Particle Filter (B=8, K=8, keep=0.5) |    12.31 |    256 |  20.8 |   1.50x | 44.6% |  57 |    4.49 |      9.36 | PASS |  no |     tok 19 |
| parsing    | Particle Filter (B=8, K=8, keep=0.25) |    12.97 |    256 |  19.7 |   1.42x | 41.6% |  60 |    4.27 |      9.31 | PASS |  no |     tok 19 |
| regex      |                    Baseline (7B only) |    18.80 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| regex      |  Adaptive Dual-Gate (Max K=8, t=0.65) |    10.03 |    256 |  25.5 |   1.87x | 65.6% |  68 |    3.76 |      9.11 | PASS |  no |     tok 97 |
| regex      |   Monte Carlo (B=8, K=8, draft T=1.2) |    12.82 |    256 |  20.0 |   1.47x | 42.5% |  59 |    4.34 |      9.38 | PASS |  no |     tok 76 |
| regex      |  Particle Filter (B=8, K=8, keep=0.5) |    11.72 |    256 |  21.8 |   1.60x | 47.7% |  54 |    4.74 |      9.37 | PASS |  no |    tok 133 |
| regex      | Particle Filter (B=8, K=8, keep=0.25) |    11.00 |    256 |  23.3 |   1.71x | 51.2% |  51 |    5.02 |      9.37 | PASS |  no |    tok 133 |
| async      |                    Baseline (7B only) |    18.47 |    256 |  13.9 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| async      |  Adaptive Dual-Gate (Max K=8, t=0.65) |     8.91 |    256 |  28.7 |   2.07x | 79.4% |  64 |    4.00 |      9.12 | PASS |  no |    tok 129 |
| async      |   Monte Carlo (B=8, K=8, draft T=1.2) |     8.90 |    256 |  28.7 |   2.07x | 66.5% |  41 |    6.24 |      9.38 | PASS |  no |    tok 146 |
| async      |  Particle Filter (B=8, K=8, keep=0.5) |    10.97 |    256 |  23.3 |   1.68x | 51.6% |  51 |    5.02 |      9.32 | PASS |  no |    tok 146 |
| async      | Particle Filter (B=8, K=8, keep=0.25) |     9.25 |    256 |  27.7 |   2.00x | 63.1% |  43 |    5.95 |      9.29 | PASS |  no |     tok 30 |
| generator  |                    Baseline (7B only) |    18.62 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| generator  |  Adaptive Dual-Gate (Max K=8, t=0.65) |     8.03 |    256 |  31.9 |   2.32x | 83.1% |  51 |    5.02 |      9.12 | PASS |  no |    tok 212 |
| generator  |   Monte Carlo (B=8, K=8, draft T=1.2) |     7.56 |    256 |  33.9 |   2.46x | 80.1% |  35 |    7.31 |      9.38 | PASS |  no |     tok 16 |
| generator  |  Particle Filter (B=8, K=8, keep=0.5) |     8.59 |    256 |  29.8 |   2.17x | 69.3% |  40 |    6.40 |      9.29 | PASS |  no |    tok 212 |
| generator  | Particle Filter (B=8, K=8, keep=0.25) |     8.91 |    256 |  28.7 |   2.09x | 66.2% |  42 |    6.10 |      9.29 | PASS |  no |  identical |
| decorator  |                    Baseline (7B only) |    18.58 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| decorator  |  Adaptive Dual-Gate (Max K=8, t=0.65) |     8.58 |    256 |  29.8 |   2.17x | 79.8% |  56 |    4.57 |      9.12 | PASS |  no |    tok 102 |
| decorator  |   Monte Carlo (B=8, K=8, draft T=1.2) |     9.31 |    256 |  27.5 |   2.00x | 64.0% |  42 |    6.10 |      9.39 | PASS |  no |    tok 102 |
| decorator  |  Particle Filter (B=8, K=8, keep=0.5) |    10.01 |    256 |  25.6 |   1.86x | 58.0% |  46 |    5.57 |      9.31 | PASS |  no |     tok 42 |
| decorator  | Particle Filter (B=8, K=8, keep=0.25) |     9.79 |    256 |  26.2 |   1.90x | 59.9% |  45 |    5.69 |      9.33 | PASS |  no |      tok 0 |
| numeric    |                    Baseline (7B only) |    18.75 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| numeric    |  Adaptive Dual-Gate (Max K=8, t=0.65) |     8.86 |    256 |  28.9 |   2.12x | 76.4% |  59 |    4.34 |      9.12 | PASS |  no |     tok 12 |
| numeric    |   Monte Carlo (B=8, K=8, draft T=1.2) |     7.76 |    256 |  33.0 |   2.42x | 77.0% |  36 |    7.11 |      9.38 | PASS |  no |  identical |
| numeric    |  Particle Filter (B=8, K=8, keep=0.5) |     9.03 |    256 |  28.3 |   2.08x | 65.2% |  42 |    6.10 |      9.29 | PASS |  no |     tok 12 |
| numeric    | Particle Filter (B=8, K=8, keep=0.25) |    10.37 |    256 |  24.7 |   1.81x | 54.9% |  48 |    5.33 |      9.28 | PASS |  no |     tok 12 |
| oop        |                    Baseline (7B only) |    18.51 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| oop        |  Adaptive Dual-Gate (Max K=8, t=0.65) |     7.49 |    256 |  34.2 |   2.47x | 86.9% |  45 |    5.69 |      9.12 | PASS |  no |     tok 52 |
| oop        |   Monte Carlo (B=8, K=8, draft T=1.2) |     7.48 |    256 |  34.2 |   2.48x | 81.0% |  35 |    7.31 |      9.38 | PASS |  no |    tok 208 |
| oop        |  Particle Filter (B=8, K=8, keep=0.5) |     8.14 |    256 |  31.5 |   2.27x | 73.5% |  38 |    6.74 |      9.30 | PASS |  no |     tok 52 |
| oop        | Particle Filter (B=8, K=8, keep=0.25) |     7.97 |    256 |  32.1 |   2.32x | 74.8% |  37 |    6.92 |      9.29 | PASS |  no |  identical |
| errors     |                    Baseline (7B only) |    18.37 |    256 |  13.9 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| errors     |  Adaptive Dual-Gate (Max K=8, t=0.65) |     8.64 |    256 |  29.6 |   2.13x | 80.7% |  60 |    4.27 |      9.12 | PASS |  no |     tok 66 |
| errors     |   Monte Carlo (B=8, K=8, draft T=1.2) |     7.97 |    256 |  32.1 |   2.30x | 75.3% |  37 |    6.92 |      9.38 | PASS |  no |     tok 66 |
| errors     |  Particle Filter (B=8, K=8, keep=0.5) |     9.67 |    256 |  26.5 |   1.90x | 60.1% |  45 |    5.69 |      9.35 | PASS |  no |     tok 66 |
| errors     | Particle Filter (B=8, K=8, keep=0.25) |     9.61 |    256 |  26.6 |   1.91x | 60.7% |  45 |    5.69 |      9.34 | PASS |  no |     tok 66 |

## Summary

- **Best configuration**: Monte Carlo (B=8, K=8, draft T=1.2) at **2.26x** the baseline
- **Baseline throughput**: 13.8 tok/s
- **Mean acceptance rate**: 69.0%
- **Peak VRAM**: 9.39 GiB of 15.9 GiB
- **Generated code parses**: 50/50 runs
- **Output identical to baseline**: 11/40 greedy runs

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

- Divergence begins at token **78** on average (earliest 0, latest 212) of 256 generated.
- Both continuations remain valid Python; the Code column is scored independently of whether the wording matched.

The practical consequence: with a reduced-precision target, treat speculative decoding as distribution-preserving up to that precision, not as a bitwise-identical drop-in. The speedups above are unaffected -- they are throughput measurements, and both arms decode the same kind of text at the same budget.
