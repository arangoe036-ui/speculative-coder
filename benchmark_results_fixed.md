# Speculative Decoding Benchmark

- **Draft model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (bfloat16)
- **Target model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (8-bit, LLM.int8)
- **GPU**: NVIDIA GeForce RTX 5080 (15.9 GiB)
- **Sampling**: greedy (temperature 0)
- **Budget**: 256 new tokens (forced: EOS disabled, every run emits exactly this many), 10 prompts
- **Verification vocabulary**: 151936 tokens (shared prefix)
- **Baseline**: `model.generate()` on the 7B target alone

## Aggregate results

| Configuration                        | Time (s) | Tok/s | Speedup | Alpha | Draft len | Fwd passes | Tok/fwd | Peak VRAM | Code OK |
| ------------------------------------ | -------: | ----: | ------: | ----: | --------: | ---------: | ------: | --------: | ------: |
| Baseline (7B only)                   |    18.72 |  13.7 |   1.00x |    -- |        -- |      256.0 |    1.00 | 11.07 GiB |   10/10 |
| Speculative K=1                      |    12.82 |  20.0 |   1.46x | 92.3% |      1.00 |      133.3 |    1.92 | 11.07 GiB |   10/10 |
| Speculative K=3                      |    10.01 |  25.8 |   1.88x | 81.7% |      2.99 |       75.1 |    3.43 | 11.08 GiB |   10/10 |
| Speculative K=5                      |     9.07 |  28.4 |   2.08x | 77.2% |      4.97 |       53.5 |    4.82 | 11.08 GiB |   10/10 |
| Speculative K=7                      |     9.03 |  28.8 |   2.10x | 70.8% |      6.93 |       44.1 |    5.89 | 11.08 GiB |    9/10 |
| Adaptive Dual-Gate (Max K=8, t=0.35) |     8.99 |  28.9 |   2.12x | 70.6% |      7.53 |       41.5 |    6.34 | 11.08 GiB |   10/10 |
| Adaptive Dual-Gate (Max K=8, t=0.65) |     8.64 |  30.1 |   2.20x | 82.8% |      5.05 |       51.2 |    5.22 | 11.08 GiB |   10/10 |

*Alpha is the fraction of drafted tokens accepted. Tok/fwd is tokens emitted per target forward pass, the hardware-independent measure of the win: the baseline is exactly 1.00 by definition.*

## Speedup by prompt

| Prompt     | Baseline tok/s |   K=1 |   K=3 |   K=5 |   K=7 | Gate t=0.35 | Gate t=0.65 |                Best |
| ---------- | -------------: | ----: | ----: | ----: | ----: | ----------: | ----------: | ------------------: |
| algorithm  |           13.7 | 1.49x | 2.00x | 2.20x | 2.29x |       2.25x |       2.47x | Gate t=0.65 (2.47x) |
| datastruct |           13.6 | 1.48x | 2.08x | 2.41x | 2.50x |       2.53x |       2.65x | Gate t=0.65 (2.65x) |
| parsing    |           13.8 | 1.44x | 1.74x | 1.89x | 1.86x |       1.80x |       2.00x | Gate t=0.65 (2.00x) |
| regex      |           13.6 | 1.43x | 1.77x | 1.79x | 1.63x |       1.65x |       1.80x | Gate t=0.65 (1.80x) |
| async      |           13.8 | 1.45x | 1.82x | 1.88x | 2.04x |       1.90x |       2.25x | Gate t=0.65 (2.25x) |
| generator  |           13.6 | 1.47x | 1.55x | 2.06x | 2.02x |       2.05x |       1.76x |         K=5 (2.06x) |
| decorator  |           13.7 | 1.44x | 1.95x | 2.15x | 2.11x |       2.27x |       2.12x | Gate t=0.35 (2.27x) |
| numeric    |           13.7 | 1.42x | 1.92x | 2.01x | 2.13x |       2.20x |       2.17x | Gate t=0.35 (2.20x) |
| oop        |           13.7 | 1.51x | 2.03x | 2.27x | 2.46x |       2.42x |       2.44x |         K=7 (2.46x) |
| errors     |           13.7 | 1.47x | 1.97x | 2.13x | 1.99x |       2.09x |       2.36x | Gate t=0.65 (2.36x) |

## Acceptance rate (alpha) by prompt

| Prompt     |   K=1 |   K=3 |   K=5 |   K=7 | Gate t=0.35 | Gate t=0.65 |
| ---------- | ----: | ----: | ----: | ----: | ----------: | ----------: |
| algorithm  | 95.4% | 88.6% | 82.4% | 77.1% |       74.8% |       88.6% |
| datastruct | 96.2% | 92.6% | 92.5% | 86.3% |       84.5% |       94.4% |
| parsing    | 89.6% | 73.8% | 69.3% | 61.9% |       59.3% |       81.3% |
| regex      | 86.9% | 73.3% | 63.5% | 51.4% |       54.2% |       68.7% |
| async      | 92.5% | 78.6% | 68.3% | 69.3% |       62.5% |       87.3% |
| generator  | 92.5% | 61.3% | 75.2% | 67.4% |       67.9% |       63.3% |
| decorator  | 89.6% | 86.4% | 80.4% | 71.3% |       77.4% |       83.5% |
| numeric    | 89.0% | 84.0% | 74.0% | 71.3% |       73.7% |       80.0% |
| oop        | 96.9% | 91.3% | 87.1% | 84.9% |       82.5% |       91.1% |
| errors     | 94.7% | 87.3% | 79.2% | 67.0% |       69.6% |       90.2% |

## Dual-gate telemetry

Draft ceiling 8 tokens. The statistical gate fires when the draft's top-1 probability (read from the unwarped softmax) falls below the threshold; the syntactic gate fires on an unrecoverable bracket state inside fenced code. `Suppressed` counts fatal-looking bracket states seen in prose, where the syntactic gate deliberately stays silent because enumerations like "1)" are not syntax errors.

### Threshold t = 0.35

| Prompt     | Iters | Mean draft len | Statistical | Syntactic | Ungated | Gated % | Suppressed | Alpha | Speedup |
| ---------- | ----: | -------------: | ----------: | --------: | ------: | ------: | ---------: | ----: | ------: |
| algorithm  |    37 |           7.95 |           0 |         0 |      37 |      0% |          0 | 74.8% |   2.25x |
| datastruct |    33 |           8.00 |           1 |         0 |      32 |      3% |          0 | 84.5% |   2.53x |
| parsing    |    49 |           7.16 |           5 |         0 |      44 |     10% |          0 | 59.3% |   1.80x |
| regex      |    59 |           6.19 |          22 |         0 |      37 |     37% |          0 | 54.2% |   1.65x |
| async      |    45 |           7.53 |           5 |         0 |      40 |     11% |          0 | 62.5% |   1.90x |
| generator  |    41 |           7.76 |           2 |         0 |      39 |      5% |          0 | 67.9% |   2.05x |
| decorator  |    38 |           7.45 |           6 |         0 |      32 |     16% |          0 | 77.4% |   2.27x |
| numeric    |    38 |           7.82 |           1 |         0 |      37 |      3% |          0 | 73.7% |   2.20x |
| oop        |    35 |           7.69 |           2 |         0 |      33 |      6% |          0 | 82.5% |   2.42x |
| errors     |    40 |           7.80 |           1 |         0 |      39 |      2% |          0 | 69.6% |   2.09x |
| **total**  |   415 |           7.53 |          45 |         0 |     370 |     11% |          0 | 70.6% |   2.12x |

### Threshold t = 0.65

| Prompt     | Iters | Mean draft len | Statistical | Syntactic | Ungated | Gated % | Suppressed | Alpha | Speedup |
| ---------- | ----: | -------------: | ----------: | --------: | ------: | ------: | ---------: | ----: | ------: |
| algorithm  |    39 |           6.31 |          16 |         0 |      23 |     41% |          0 | 88.6% |   2.47x |
| datastruct |    36 |           6.47 |          12 |         0 |      24 |     33% |          0 | 94.4% |   2.65x |
| parsing    |    61 |           3.95 |          49 |         0 |      12 |     80% |          0 | 81.3% |   2.00x |
| regex      |    68 |           4.04 |          47 |         0 |      21 |     69% |          0 | 68.7% |   1.80x |
| async      |    50 |           4.72 |          34 |         0 |      16 |     68% |          0 | 87.3% |   2.25x |
| generator  |    66 |           4.55 |          40 |         0 |      26 |     61% |          0 | 63.3% |   1.76x |
| decorator  |    55 |           4.40 |          39 |         0 |      16 |     71% |          0 | 83.5% |   2.12x |
| numeric    |    49 |           5.31 |          30 |         0 |      19 |     61% |          0 | 80.0% |   2.17x |
| oop        |    42 |           5.62 |          23 |         0 |      19 |     55% |          0 | 91.1% |   2.44x |
| errors     |    46 |           5.09 |          29 |         0 |      17 |     63% |          0 | 90.2% |   2.36x |
| **total**  |   512 |           5.05 |         319 |         0 |     193 |     62% |          0 | 82.8% |   2.20x |

## Full results

| Prompt     |                        Configuration | Time (s) | Tokens | Tok/s | Speedup | Alpha | Fwd | Tok/fwd | Peak VRAM |  Code | EOS | Diverges @ |
| ---------- | -----------------------------------: | -------: | -----: | ----: | ------: | ----: | --: | ------: | --------: | ----: | --: | ---------: |
| algorithm  |                   Baseline (7B only) |    18.70 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |     11.06 |  PASS |  no |         -- |
| algorithm  |                      Speculative K=1 |    12.57 |    256 |  20.4 |   1.49x | 95.4% | 131 |    1.95 |     11.06 |  PASS |  no |  identical |
| algorithm  |                      Speculative K=3 |     9.34 |    256 |  27.4 |   2.00x | 88.6% |  70 |    3.66 |     11.06 |  PASS |  no |  identical |
| algorithm  |                      Speculative K=5 |     8.50 |    256 |  30.1 |   2.20x | 82.4% |  50 |    5.12 |     11.07 |  PASS |  no |  identical |
| algorithm  |                      Speculative K=7 |     8.17 |    256 |  31.3 |   2.29x | 77.1% |  40 |    6.40 |     11.07 |  PASS |  no |  identical |
| algorithm  | Adaptive Dual-Gate (Max K=8, t=0.35) |     8.31 |    256 |  30.8 |   2.25x | 74.8% |  37 |    6.92 |     11.07 |  PASS |  no |  identical |
| algorithm  | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.58 |    256 |  33.8 |   2.47x | 88.6% |  39 |    6.56 |     11.06 |  PASS |  no |    tok 202 |
| datastruct |                   Baseline (7B only) |    18.82 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |     11.06 |  PASS |  no |         -- |
| datastruct |                      Speculative K=1 |    12.68 |    256 |  20.2 |   1.48x | 96.2% | 131 |    1.95 |     11.07 |  PASS |  no |    tok 102 |
| datastruct |                      Speculative K=3 |     9.03 |    256 |  28.4 |   2.08x | 92.6% |  68 |    3.76 |     11.07 |  PASS |  no |  identical |
| datastruct |                      Speculative K=5 |     7.82 |    256 |  32.7 |   2.41x | 92.5% |  46 |    5.57 |     11.07 |  PASS |  no |  identical |
| datastruct |                      Speculative K=7 |     7.53 |    256 |  34.0 |   2.50x | 86.3% |  37 |    6.92 |     11.07 |  PASS |  no |  identical |
| datastruct | Adaptive Dual-Gate (Max K=8, t=0.35) |     7.44 |    256 |  34.4 |   2.53x | 84.5% |  33 |    7.76 |     11.07 |  PASS |  no |  identical |
| datastruct | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.11 |    256 |  36.0 |   2.65x | 94.4% |  36 |    7.11 |     11.07 |  PASS |  no |  identical |
| parsing    |                   Baseline (7B only) |    18.60 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |     11.06 |  PASS |  no |         -- |
| parsing    |                      Speculative K=1 |    12.96 |    256 |  19.8 |   1.44x | 89.6% | 135 |    1.90 |     11.07 |  PASS |  no |    tok 158 |
| parsing    |                      Speculative K=3 |    10.71 |    256 |  23.9 |   1.74x | 73.8% |  80 |    3.20 |     11.07 |  PASS |  no |     tok 19 |
| parsing    |                      Speculative K=5 |     9.84 |    256 |  26.0 |   1.89x | 69.3% |  58 |    4.41 |     11.07 |  PASS |  no |    tok 158 |
| parsing    |                      Speculative K=7 |    10.02 |    256 |  25.5 |   1.86x | 61.9% |  49 |    5.22 |     11.07 |  PASS |  no |     tok 19 |
| parsing    | Adaptive Dual-Gate (Max K=8, t=0.35) |    10.31 |    256 |  24.8 |   1.80x | 59.3% |  49 |    5.22 |     11.07 |  PASS |  no |    tok 158 |
| parsing    | Adaptive Dual-Gate (Max K=8, t=0.65) |     9.31 |    256 |  27.5 |   2.00x | 81.3% |  61 |    4.20 |     11.07 |  PASS |  no |     tok 19 |
| regex      |                   Baseline (7B only) |    18.83 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |     11.06 |  PASS |  no |         -- |
| regex      |                      Speculative K=1 |    13.19 |    256 |  19.4 |   1.43x | 86.9% | 137 |    1.87 |     11.07 |  PASS |  no |     tok 97 |
| regex      |                      Speculative K=3 |    10.66 |    256 |  24.0 |   1.77x | 73.3% |  80 |    3.20 |     11.07 |  PASS |  no |     tok 97 |
| regex      |                      Speculative K=5 |    10.53 |    256 |  24.3 |   1.79x | 63.5% |  62 |    4.13 |     11.07 |  PASS |  no |     tok 97 |
| regex      |                      Speculative K=7 |    11.53 |    256 |  22.2 |   1.63x | 51.4% |  56 |    4.57 |     11.07 | TRUNC |  no |     tok 76 |
| regex      | Adaptive Dual-Gate (Max K=8, t=0.35) |    11.43 |    256 |  22.4 |   1.65x | 54.2% |  59 |    4.34 |     11.08 |  PASS |  no |     tok 76 |
| regex      | Adaptive Dual-Gate (Max K=8, t=0.65) |    10.44 |    256 |  24.5 |   1.80x | 68.7% |  68 |    3.76 |     11.07 |  PASS |  no |     tok 97 |
| async      |                   Baseline (7B only) |    18.56 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |     11.07 |  PASS |  no |         -- |
| async      |                      Speculative K=1 |    12.78 |    256 |  20.0 |   1.45x | 92.5% | 133 |    1.92 |     11.07 |  PASS |  no |  identical |
| async      |                      Speculative K=3 |    10.18 |    256 |  25.2 |   1.82x | 78.6% |  77 |    3.32 |     11.07 |  PASS |  no |    tok 129 |
| async      |                      Speculative K=5 |     9.88 |    256 |  25.9 |   1.88x | 68.3% |  59 |    4.34 |     11.07 |  PASS |  no |    tok 129 |
| async      |                      Speculative K=7 |     9.11 |    256 |  28.1 |   2.04x | 69.3% |  45 |    5.69 |     11.07 |  PASS |  no |    tok 144 |
| async      | Adaptive Dual-Gate (Max K=8, t=0.35) |     9.79 |    256 |  26.1 |   1.90x | 62.5% |  45 |    5.69 |     11.08 |  PASS |  no |    tok 129 |
| async      | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.24 |    256 |  31.1 |   2.25x | 87.3% |  50 |    5.12 |     11.07 |  PASS |  no |    tok 144 |
| generator  |                   Baseline (7B only) |    18.84 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |     11.07 |  PASS |  no |         -- |
| generator  |                      Speculative K=1 |    12.81 |    256 |  20.0 |   1.47x | 92.5% | 133 |    1.92 |     11.07 |  PASS |  no |  identical |
| generator  |                      Speculative K=3 |    12.16 |    256 |  21.0 |   1.55x | 61.3% |  91 |    2.81 |     11.08 |  PASS |  no |      tok 3 |
| generator  |                      Speculative K=5 |     9.16 |    256 |  27.9 |   2.06x | 75.2% |  54 |    4.74 |     11.08 |  PASS |  no |     tok 16 |
| generator  |                      Speculative K=7 |     9.32 |    256 |  27.5 |   2.02x | 67.4% |  46 |    5.57 |     11.08 |  PASS |  no |    tok 212 |
| generator  | Adaptive Dual-Gate (Max K=8, t=0.35) |     9.17 |    256 |  27.9 |   2.05x | 67.9% |  41 |    6.24 |     11.08 |  PASS |  no |     tok 16 |
| generator  | Adaptive Dual-Gate (Max K=8, t=0.65) |    10.72 |    256 |  23.9 |   1.76x | 63.3% |  66 |    3.88 |     11.08 |  PASS |  no |     tok 15 |
| decorator  |                   Baseline (7B only) |    18.73 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |     11.07 |  PASS |  no |         -- |
| decorator  |                      Speculative K=1 |    13.04 |    256 |  19.6 |   1.44x | 89.6% | 135 |    1.90 |     11.07 |  PASS |  no |  identical |
| decorator  |                      Speculative K=3 |     9.61 |    256 |  26.6 |   1.95x | 86.4% |  72 |    3.56 |     11.07 |  PASS |  no |     tok 42 |
| decorator  |                      Speculative K=5 |     8.71 |    256 |  29.4 |   2.15x | 80.4% |  51 |    5.02 |     11.07 |  PASS |  no |    tok 112 |
| decorator  |                      Speculative K=7 |     8.89 |    256 |  28.8 |   2.11x | 71.3% |  43 |    5.95 |     11.08 |  PASS |  no |    tok 112 |
| decorator  | Adaptive Dual-Gate (Max K=8, t=0.35) |     8.24 |    256 |  31.1 |   2.27x | 77.4% |  38 |    6.74 |     11.08 |  PASS |  no |     tok 48 |
| decorator  | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.84 |    256 |  28.9 |   2.12x | 83.5% |  55 |    4.65 |     11.07 |  PASS |  no |    tok 102 |
| numeric    |                   Baseline (7B only) |    18.74 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |     11.06 |  PASS |  no |         -- |
| numeric    |                      Speculative K=1 |    13.16 |    256 |  19.4 |   1.42x | 89.0% | 136 |    1.88 |     11.07 |  PASS |  no |     tok 12 |
| numeric    |                      Speculative K=3 |     9.76 |    256 |  26.2 |   1.92x | 84.0% |  73 |    3.51 |     11.07 |  PASS |  no |  identical |
| numeric    |                      Speculative K=5 |     9.33 |    256 |  27.5 |   2.01x | 74.0% |  55 |    4.65 |     11.07 |  PASS |  no |     tok 12 |
| numeric    |                      Speculative K=7 |     8.81 |    256 |  29.0 |   2.13x | 71.3% |  43 |    5.95 |     11.07 |  PASS |  no |    tok 215 |
| numeric    | Adaptive Dual-Gate (Max K=8, t=0.35) |     8.53 |    256 |  30.0 |   2.20x | 73.7% |  38 |    6.74 |     11.08 |  PASS |  no |     tok 12 |
| numeric    | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.64 |    256 |  29.6 |   2.17x | 80.0% |  49 |    5.22 |     11.07 |  PASS |  no |     tok 12 |
| oop        |                   Baseline (7B only) |    18.70 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |     11.06 |  PASS |  no |         -- |
| oop        |                      Speculative K=1 |    12.41 |    256 |  20.6 |   1.51x | 96.9% | 130 |    1.97 |     11.07 |  PASS |  no |     tok 52 |
| oop        |                      Speculative K=3 |     9.19 |    256 |  27.8 |   2.03x | 91.3% |  69 |    3.71 |     11.07 |  PASS |  no |  identical |
| oop        |                      Speculative K=5 |     8.25 |    256 |  31.0 |   2.27x | 87.1% |  48 |    5.33 |     11.07 |  PASS |  no |    tok 208 |
| oop        |                      Speculative K=7 |     7.58 |    256 |  33.8 |   2.46x | 84.9% |  37 |    6.92 |     11.07 |  PASS |  no |    tok 208 |
| oop        | Adaptive Dual-Gate (Max K=8, t=0.35) |     7.72 |    256 |  33.1 |   2.42x | 82.5% |  35 |    7.31 |     11.07 |  PASS |  no |    tok 208 |
| oop        | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.66 |    256 |  33.4 |   2.44x | 91.1% |  42 |    6.10 |     11.07 |  PASS |  no |     tok 52 |
| errors     |                   Baseline (7B only) |    18.63 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |     11.07 |  PASS |  no |         -- |
| errors     |                      Speculative K=1 |    12.64 |    256 |  20.3 |   1.47x | 94.7% | 132 |    1.94 |     11.07 |  PASS |  no |     tok 66 |
| errors     |                      Speculative K=3 |     9.47 |    256 |  27.0 |   1.97x | 87.3% |  71 |    3.61 |     11.07 |  PASS |  no |     tok 66 |
| errors     |                      Speculative K=5 |     8.74 |    256 |  29.3 |   2.13x | 79.2% |  52 |    4.92 |     11.07 |  PASS |  no |     tok 66 |
| errors     |                      Speculative K=7 |     9.34 |    256 |  27.4 |   1.99x | 67.0% |  45 |    5.69 |     11.07 |  PASS |  no |     tok 66 |
| errors     | Adaptive Dual-Gate (Max K=8, t=0.35) |     8.93 |    256 |  28.7 |   2.09x | 69.6% |  40 |    6.40 |     11.08 |  PASS |  no |     tok 66 |
| errors     | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.90 |    256 |  32.4 |   2.36x | 90.2% |  46 |    5.57 |     11.08 |  PASS |  no |    tok 223 |

## Summary

- **Best configuration**: Adaptive Dual-Gate (Max K=8, t=0.65) at **2.20x** the baseline
- **Baseline throughput**: 13.7 tok/s
- **Mean acceptance rate**: 79.3%
- **Peak VRAM**: 11.08 GiB of 15.9 GiB
- **Generated code parses**: 69/70 runs (1 truncated by the token budget, not scored)
- **Output identical to baseline**: 15/60 greedy runs

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

- Divergence begins at token **94** on average (earliest 3, latest 223) of 256 generated.
- Both continuations remain valid Python; the Code column is scored independently of whether the wording matched.

The practical consequence: with a reduced-precision target, treat speculative decoding as distribution-preserving up to that precision, not as a bitwise-identical drop-in. The speedups above are unaffected -- they are throughput measurements, and both arms decode the same kind of text at the same budget.
