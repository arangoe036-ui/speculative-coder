# Speculative Decoding Benchmark

- **Draft model**: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (bfloat16)
- **Target model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (8-bit, LLM.int8)
- **GPU**: NVIDIA GeForce RTX 5080 (15.9 GiB)
- **Sampling**: greedy (temperature 0)
- **Budget**: 256 new tokens (forced: EOS disabled, every run emits exactly this many), 10 prompts
- **Verification vocabulary**: 151936 tokens (shared prefix)
- **Baseline**: `model.generate()` on the 7B target alone

## Aggregate results

| Configuration                        | Time (s) | Tok/s | Speedup | Alpha | 7B fwd/token | Weights VRAM | Peak VRAM | Code OK |
| ------------------------------------ | -------: | ----: | ------: | ----: | -----------: | -----------: | --------: | ------: |
| Baseline (7B only)                   |    18.54 |  13.8 |   1.00x |    -- |         1.00 |     8.11 GiB |  8.19 GiB |   10/10 |
| Adaptive Dual-Gate (Max K=8, t=0.65) |     8.51 |  30.5 |   2.21x | 78.4% |         0.22 |     9.05 GiB |  9.12 GiB |   10/10 |
| Monte Carlo (B=4, K=5, draft T=1.2)  |     9.17 |  28.3 |   2.05x | 73.8% |         0.22 |     9.05 GiB |  9.20 GiB |   10/10 |
| Monte Carlo (B=4, K=8, draft T=1.2)  |     9.55 |  28.2 |   2.05x | 62.8% |         0.18 |     9.05 GiB |  9.23 GiB |   10/10 |
| Monte Carlo (B=8, K=5, draft T=1.2)  |     8.38 |  30.7 |   2.23x | 83.5% |         0.20 |     9.05 GiB |  9.33 GiB |    9/10 |
| Monte Carlo (B=8, K=8, draft T=1.2)  |     8.41 |  31.1 |   2.25x | 71.8% |         0.15 |     9.05 GiB |  9.39 GiB |    9/10 |

*Alpha is the fraction of drafted tokens accepted. `7B fwd/token` counts every forward pass of a **full-size** model per emitted token, which is the metric that makes the three architectures comparable: the two-model engines get their draft forwards from a 1.5B model (~4.4x cheaper, so excluded), while self-speculation drafts with the 7B itself, so its draft forwards cost full price and are counted. Plain decoding is 1.00 by definition; below 1.00 is a real win, above 1.00 means more full-size compute than simply decoding.*

## Speedup by prompt

| Prompt     | Baseline tok/s | Gate t=0.65 | MC B4/K5 | MC B4/K8 | MC B8/K5 | MC B8/K8 |                Best |
| ---------- | -------------: | ----------: | -------: | -------: | -------: | -------: | ------------------: |
| algorithm  |           13.8 |       2.52x |    2.37x |    2.45x |    2.38x |    2.59x |    MC B8/K8 (2.59x) |
| datastruct |           13.7 |       2.62x |    2.48x |    2.90x |    2.53x |    2.78x |    MC B4/K8 (2.90x) |
| parsing    |           13.9 |       1.75x |    1.67x |    1.55x |    2.05x |    1.79x |    MC B8/K5 (2.05x) |
| regex      |           13.7 |       1.88x |    1.76x |    1.78x |    1.88x |    1.77x | Gate t=0.65 (1.88x) |
| async      |           13.9 |       2.07x |    1.84x |    1.79x |    2.15x |    2.21x |    MC B8/K8 (2.21x) |
| generator  |           13.8 |       2.33x |    2.09x |    1.28x |    2.27x |    2.34x |    MC B8/K8 (2.34x) |
| decorator  |           13.8 |       2.18x |    1.94x |    1.94x |    2.12x |    2.17x | Gate t=0.65 (2.18x) |
| numeric    |           13.8 |       2.10x |    1.98x |    2.05x |    2.33x |    2.29x |    MC B8/K5 (2.33x) |
| oop        |           13.8 |       2.49x |    2.32x |    2.51x |    2.42x |    2.57x |    MC B8/K8 (2.57x) |
| errors     |           13.9 |       2.15x |    2.07x |    2.19x |    2.14x |    1.98x |    MC B4/K8 (2.19x) |

## Acceptance rate (alpha) by prompt

| Prompt     | Gate t=0.65 | MC B4/K5 | MC B4/K8 | MC B8/K5 | MC B8/K8 |
| ---------- | ----------: | -------: | -------: | -------: | -------: |
| algorithm  |       83.4% |    88.2% |    77.3% |    90.1% |    83.5% |
| datastruct |       85.2% |    92.1% |    93.8% |    95.9% |    90.4% |
| parsing    |       63.5% |    56.6% |    45.3% |    77.3% |    55.1% |
| regex      |       65.6% |    59.6% |    52.2% |    66.8% |    54.0% |
| async      |       79.4% |    64.7% |    54.5% |    80.8% |    70.9% |
| generator  |       83.1% |    74.8% |    34.9% |    84.9% |    75.3% |
| decorator  |       79.8% |    69.3% |    59.9% |    78.5% |    69.3% |
| numeric    |       76.4% |    70.7% |    62.8% |    87.8% |    73.0% |
| oop        |       86.9% |    86.3% |    79.3% |    92.1% |    83.8% |
| errors     |       80.7% |    75.5% |    68.7% |    80.4% |    62.1% |

## Dual-gate telemetry

Draft ceiling 8 tokens. The statistical gate fires when the draft's top-1 probability (read from the unwarped softmax) falls below the threshold; the syntactic gate fires on an unrecoverable bracket state inside fenced code. `Suppressed` counts fatal-looking bracket states seen in prose, where the syntactic gate deliberately stays silent because enumerations like "1)" are not syntax errors.

### Threshold t = 0.65

| Prompt     | Iters | Mean draft len | Statistical | Syntactic | Ungated | Gated % | Suppressed | Alpha | Speedup |
| ---------- | ----: | -------------: | ----------: | --------: | ------: | ------: | ---------: | ----: | ------: |
| algorithm  |    41 |           6.32 |          17 |         0 |      24 |     41% |          0 | 83.4% |   2.52x |
| datastruct |    38 |           6.74 |           9 |         0 |      29 |     24% |          0 | 85.2% |   2.62x |
| parsing    |    76 |           3.75 |          61 |         0 |      15 |     80% |          0 | 63.5% |   1.75x |
| regex      |    68 |           4.24 |          48 |         0 |      20 |     71% |          0 | 65.6% |   1.88x |
| async      |    64 |           3.80 |          51 |         0 |      13 |     80% |          0 | 79.4% |   2.07x |
| generator  |    51 |           4.86 |          34 |         0 |      17 |     67% |          0 | 83.1% |   2.33x |
| decorator  |    56 |           4.50 |          42 |         0 |      14 |     75% |          0 | 79.8% |   2.18x |
| numeric    |    59 |           4.37 |          46 |         0 |      13 |     78% |          0 | 76.4% |   2.10x |
| oop        |    45 |           5.42 |          27 |         0 |      18 |     60% |          0 | 86.9% |   2.49x |
| errors     |    60 |           4.07 |          49 |         0 |      11 |     82% |          0 | 80.7% |   2.15x |
| **total**  |   558 |           4.81 |         384 |         0 |     174 |     69% |          0 | 78.4% |   2.21x |

## Monte Carlo branch telemetry

`Best` is the accepted-token count of the winning branch; `Single` is the mean across all branches in the same iteration -- what one branch would have achieved on the same draft samples. `Gain` is the difference, i.e. the extra tokens per iteration that breadth actually buys. `Wins` shows how often each branch index won; a degenerate spread would mean the branches are not diverging.

| Config   | Iters | Best | Single |  Gain | Tok/fwd | Tok/s | Speedup |                                 Win spread |
| -------- | ----: | ---: | -----: | ----: | ------: | ----: | ------: | -----------------------------------------: |
| B=4, K=5 |   559 | 3.66 |   2.71 | +0.95 |    4.65 |  28.3 |   2.05x |                      0:68% 1:14% 2:8% 3:8% |
| B=4, K=8 |   454 | 4.95 |   3.42 | +1.53 |    5.94 |  28.2 |   2.05x |                     0:62% 1:17% 2:12% 3:7% |
| B=8, K=5 |   503 | 4.14 |   2.76 | +1.38 |    5.12 |  30.7 |   2.23x |  0:51% 1:19% 2:7% 3:7% 4:5% 5:3% 6:1% 7:2% |
| B=8, K=8 |   395 | 5.63 |   3.59 | +2.04 |    6.61 |  31.1 |   2.25x | 0:43% 1:17% 2:11% 3:6% 4:5% 5:5% 6:4% 7:5% |

## Full results

| Prompt     |                        Configuration | Time (s) | Tokens | Tok/s | Speedup | Alpha | Fwd | Tok/fwd | Peak VRAM |  Code | EOS | Diverges @ |
| ---------- | -----------------------------------: | -------: | -----: | ----: | ------: | ----: | --: | ------: | --------: | ----: | --: | ---------: |
| algorithm  |                   Baseline (7B only) |    18.55 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.18 |  PASS |  no |         -- |
| algorithm  | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.35 |    256 |  34.8 |   2.52x | 83.4% |  41 |    6.24 |      9.11 |  PASS |  no |  identical |
| algorithm  |  Monte Carlo (B=4, K=5, draft T=1.2) |     7.81 |    256 |  32.8 |   2.37x | 88.2% |  48 |    5.33 |      9.19 |  PASS |  no |  identical |
| algorithm  |  Monte Carlo (B=8, K=5, draft T=1.2) |     7.80 |    256 |  32.8 |   2.38x | 90.1% |  47 |    5.45 |      9.32 |  PASS |  no |  identical |
| algorithm  |  Monte Carlo (B=4, K=8, draft T=1.2) |     7.56 |    256 |  33.9 |   2.45x | 77.3% |  36 |    7.11 |      9.23 |  PASS |  no |  identical |
| algorithm  |  Monte Carlo (B=8, K=8, draft T=1.2) |     7.15 |    256 |  35.8 |   2.59x | 83.5% |  34 |    7.53 |      9.38 |  PASS |  no |  identical |
| datastruct |                   Baseline (7B only) |    18.64 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| datastruct | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.11 |    256 |  36.0 |   2.62x | 85.2% |  38 |    6.74 |      9.11 |  PASS |  no |  identical |
| datastruct |  Monte Carlo (B=4, K=5, draft T=1.2) |     7.50 |    256 |  34.1 |   2.48x | 92.1% |  46 |    5.57 |      9.19 |  PASS |  no |  identical |
| datastruct |  Monte Carlo (B=8, K=5, draft T=1.2) |     7.38 |    256 |  34.7 |   2.53x | 95.9% |  45 |    5.69 |      9.32 |  PASS |  no |  identical |
| datastruct |  Monte Carlo (B=4, K=8, draft T=1.2) |     6.42 |    256 |  39.9 |   2.90x | 93.8% |  31 |    8.26 |      9.23 |  PASS |  no |  identical |
| datastruct |  Monte Carlo (B=8, K=8, draft T=1.2) |     6.71 |    256 |  38.2 |   2.78x | 90.4% |  32 |    8.00 |      9.38 |  PASS |  no |  identical |
| parsing    |                   Baseline (7B only) |    18.40 |    256 |  13.9 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| parsing    | Adaptive Dual-Gate (Max K=8, t=0.65) |    10.52 |    256 |  24.3 |   1.75x | 63.5% |  76 |    3.37 |      9.11 |  PASS |  no |     tok 19 |
| parsing    |  Monte Carlo (B=4, K=5, draft T=1.2) |    11.03 |    256 |  23.2 |   1.67x | 56.6% |  67 |    3.82 |      9.19 |  PASS |  no |     tok 19 |
| parsing    |  Monte Carlo (B=8, K=5, draft T=1.2) |     8.96 |    256 |  28.6 |   2.05x | 77.3% |  53 |    4.83 |      9.32 |  PASS |  no |    tok 158 |
| parsing    |  Monte Carlo (B=4, K=8, draft T=1.2) |    11.85 |    256 |  21.6 |   1.55x | 45.3% |  56 |    4.57 |      9.23 |  PASS |  no |     tok 19 |
| parsing    |  Monte Carlo (B=8, K=8, draft T=1.2) |    10.25 |    256 |  25.0 |   1.79x | 55.1% |  48 |    5.33 |      9.38 |  PASS |  no |    tok 158 |
| regex      |                   Baseline (7B only) |    18.74 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| regex      | Adaptive Dual-Gate (Max K=8, t=0.65) |     9.96 |    256 |  25.7 |   1.88x | 65.6% |  68 |    3.76 |      9.11 |  PASS |  no |     tok 97 |
| regex      |  Monte Carlo (B=4, K=5, draft T=1.2) |    10.68 |    256 |  24.0 |   1.76x | 59.6% |  65 |    3.94 |      9.19 |  PASS |  no |     tok 76 |
| regex      |  Monte Carlo (B=8, K=5, draft T=1.2) |     9.98 |    256 |  25.7 |   1.88x | 66.8% |  60 |    4.27 |      9.32 | TRUNC |  no |     tok 97 |
| regex      |  Monte Carlo (B=4, K=8, draft T=1.2) |    10.53 |    256 |  24.3 |   1.78x | 52.2% |  50 |    5.12 |      9.23 |  PASS |  no |     tok 97 |
| regex      |  Monte Carlo (B=8, K=8, draft T=1.2) |    10.57 |    256 |  24.2 |   1.77x | 54.0% |  49 |    5.22 |      9.38 | TRUNC |  no |    tok 155 |
| async      |                   Baseline (7B only) |    18.37 |    256 |  13.9 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| async      | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.85 |    256 |  28.9 |   2.07x | 79.4% |  64 |    4.00 |      9.12 |  PASS |  no |    tok 129 |
| async      |  Monte Carlo (B=4, K=5, draft T=1.2) |     9.99 |    256 |  25.6 |   1.84x | 64.7% |  61 |    4.20 |      9.19 |  PASS |  no |     tok 30 |
| async      |  Monte Carlo (B=8, K=5, draft T=1.2) |     8.53 |    256 |  30.0 |   2.15x | 80.8% |  51 |    5.02 |      9.32 |  PASS |  no |    tok 208 |
| async      |  Monte Carlo (B=4, K=8, draft T=1.2) |    10.24 |    256 |  25.0 |   1.79x | 54.5% |  49 |    5.22 |      9.23 |  PASS |  no |    tok 129 |
| async      |  Monte Carlo (B=8, K=8, draft T=1.2) |     8.31 |    256 |  30.8 |   2.21x | 70.9% |  39 |    6.56 |      9.38 |  PASS |  no |    tok 129 |
| generator  |                   Baseline (7B only) |    18.59 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| generator  | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.97 |    256 |  32.1 |   2.33x | 83.1% |  51 |    5.02 |      9.12 |  PASS |  no |    tok 212 |
| generator  |  Monte Carlo (B=4, K=5, draft T=1.2) |     8.90 |    256 |  28.8 |   2.09x | 74.8% |  54 |    4.74 |      9.20 |  PASS |  no |  identical |
| generator  |  Monte Carlo (B=8, K=5, draft T=1.2) |     8.20 |    256 |  31.2 |   2.27x | 84.9% |  49 |    5.22 |      9.33 |  PASS |  no |     tok 16 |
| generator  |  Monte Carlo (B=4, K=8, draft T=1.2) |    14.51 |    256 |  17.6 |   1.28x | 34.9% |  69 |    3.71 |      9.23 |  PASS |  no |      tok 3 |
| generator  |  Monte Carlo (B=8, K=8, draft T=1.2) |     7.95 |    256 |  32.2 |   2.34x | 75.3% |  37 |    6.92 |      9.39 |  PASS |  no |  identical |
| decorator  |                   Baseline (7B only) |    18.55 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| decorator  | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.52 |    256 |  30.0 |   2.18x | 79.8% |  56 |    4.57 |      9.12 |  PASS |  no |    tok 102 |
| decorator  |  Monte Carlo (B=4, K=5, draft T=1.2) |     9.57 |    256 |  26.8 |   1.94x | 69.3% |  58 |    4.41 |      9.19 |  PASS |  no |     tok 42 |
| decorator  |  Monte Carlo (B=8, K=5, draft T=1.2) |     8.76 |    256 |  29.2 |   2.12x | 78.5% |  52 |    4.92 |      9.33 |  PASS |  no |    tok 102 |
| decorator  |  Monte Carlo (B=4, K=8, draft T=1.2) |     9.55 |    256 |  26.8 |   1.94x | 59.9% |  45 |    5.69 |      9.23 |  PASS |  no |     tok 42 |
| decorator  |  Monte Carlo (B=8, K=8, draft T=1.2) |     8.54 |    256 |  30.0 |   2.17x | 69.3% |  40 |    6.40 |      9.39 |  PASS |  no |     tok 42 |
| numeric    |                   Baseline (7B only) |    18.54 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| numeric    | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.81 |    256 |  29.0 |   2.10x | 76.4% |  59 |    4.34 |      9.12 |  PASS |  no |     tok 12 |
| numeric    |  Monte Carlo (B=4, K=5, draft T=1.2) |     9.36 |    256 |  27.4 |   1.98x | 70.7% |  57 |    4.49 |      9.19 |  PASS |  no |     tok 12 |
| numeric    |  Monte Carlo (B=8, K=5, draft T=1.2) |     7.96 |    256 |  32.1 |   2.33x | 87.8% |  48 |    5.33 |      9.32 |  PASS |  no |     tok 12 |
| numeric    |  Monte Carlo (B=4, K=8, draft T=1.2) |     9.04 |    256 |  28.3 |   2.05x | 62.8% |  43 |    5.95 |      9.23 |  PASS |  no |     tok 12 |
| numeric    |  Monte Carlo (B=8, K=8, draft T=1.2) |     8.11 |    256 |  31.5 |   2.29x | 73.0% |  38 |    6.74 |      9.39 |  PASS |  no |     tok 12 |
| oop        |                   Baseline (7B only) |    18.54 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| oop        | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.43 |    256 |  34.5 |   2.49x | 86.9% |  45 |    5.69 |      9.12 |  PASS |  no |     tok 52 |
| oop        |  Monte Carlo (B=4, K=5, draft T=1.2) |     7.97 |    256 |  32.1 |   2.32x | 86.3% |  49 |    5.22 |      9.19 |  PASS |  no |    tok 208 |
| oop        |  Monte Carlo (B=8, K=5, draft T=1.2) |     7.65 |    256 |  33.5 |   2.42x | 92.1% |  46 |    5.57 |      9.32 |  PASS |  no |  identical |
| oop        |  Monte Carlo (B=4, K=8, draft T=1.2) |     7.38 |    256 |  34.7 |   2.51x | 79.3% |  35 |    7.31 |      9.23 |  PASS |  no |     tok 52 |
| oop        |  Monte Carlo (B=8, K=8, draft T=1.2) |     7.21 |    256 |  35.5 |   2.57x | 83.8% |  34 |    7.53 |      9.38 |  PASS |  no |     tok 52 |
| errors     |                   Baseline (7B only) |    18.46 |    256 |  13.9 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| errors     | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.59 |    256 |  29.8 |   2.15x | 80.7% |  60 |    4.27 |      9.12 |  PASS |  no |     tok 66 |
| errors     |  Monte Carlo (B=4, K=5, draft T=1.2) |     8.90 |    256 |  28.8 |   2.07x | 75.5% |  54 |    4.74 |      9.19 |  PASS |  no |  identical |
| errors     |  Monte Carlo (B=8, K=5, draft T=1.2) |     8.64 |    256 |  29.6 |   2.14x | 80.4% |  52 |    4.92 |      9.32 |  PASS |  no |     tok 66 |
| errors     |  Monte Carlo (B=4, K=8, draft T=1.2) |     8.43 |    256 |  30.4 |   2.19x | 68.7% |  40 |    6.40 |      9.23 |  PASS |  no |     tok 66 |
| errors     |  Monte Carlo (B=8, K=8, draft T=1.2) |     9.30 |    256 |  27.5 |   1.98x | 62.1% |  44 |    5.82 |      9.39 |  PASS |  no |    tok 189 |

## Summary

- **Best configuration**: Monte Carlo (B=8, K=8, draft T=1.2) at **2.25x** the baseline
- **Baseline throughput**: 13.8 tok/s
- **Mean acceptance rate**: 74.0%
- **Peak VRAM**: 9.39 GiB of 15.9 GiB
- **Generated code parses**: 58/60 runs (2 truncated by the token budget, not scored)
- **Output identical to baseline**: 14/50 greedy runs

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

- Divergence begins at token **80** on average (earliest 3, latest 212) of 256 generated.
- Both continuations remain valid Python; the Code column is scored independently of whether the wording matched.

The practical consequence: with a reduced-precision target, treat speculative decoding as distribution-preserving up to that precision, not as a bitwise-identical drop-in. The speedups above are unaffected -- they are throughput measurements, and both arms decode the same kind of text at the same budget.
