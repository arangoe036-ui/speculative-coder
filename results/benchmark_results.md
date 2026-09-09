# Speculative Decoding Benchmark

- **Draft model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (bfloat16)
- **Target model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (8-bit, LLM.int8)
- **GPU**: NVIDIA GeForce RTX 5080 (15.9 GiB)
- **Sampling**: greedy (temperature 0)
- **Budget**: 384 new tokens, 10 prompts
- **Verification vocabulary**: 151936 tokens (shared prefix)
- **Baseline**: `model.generate()` on the 7B target alone

## Aggregate results

| Configuration                        | Time (s) | Tok/s | Speedup | Alpha | Draft len | Fwd passes | Tok/fwd | Peak VRAM | Code OK |
| ------------------------------------ | -------: | ----: | ------: | ----: | --------: | ---------: | ------: | --------: | ------: |
| Baseline (7B only)                   |    21.66 |  13.8 |   1.00x |    -- |        -- |      299.0 |    1.00 | 11.07 GiB |    9/10 |
| Speculative K=1                      |    15.06 |  20.0 |   1.44x | 91.7% |      1.00 |      157.4 |    1.91 | 11.07 GiB |    9/10 |
| Speculative K=3                      |    10.72 |  26.0 |   1.88x | 82.6% |      3.00 |       80.9 |    3.45 | 11.08 GiB |    9/10 |
| Speculative K=5                      |    10.36 |  28.4 |   2.06x | 76.7% |      4.99 |       61.7 |    4.78 | 11.08 GiB |    9/10 |
| Speculative K=7                      |    10.40 |  28.2 |   2.04x | 69.1% |      6.97 |       51.1 |    5.74 | 11.08 GiB |    9/10 |
| Adaptive Dual-Gate (Max K=8, t=0.35) |    10.64 |  28.4 |   2.06x | 69.3% |      7.48 |       49.6 |    6.11 | 11.08 GiB |    9/10 |
| Adaptive Dual-Gate (Max K=8, t=0.65) |     8.96 |  30.2 |   2.19x | 83.5% |      5.04 |       52.8 |    5.17 | 11.08 GiB |    9/10 |

*Alpha is the fraction of drafted tokens accepted. Tok/fwd is tokens emitted per target forward pass, the hardware-independent measure of the win: the baseline is exactly 1.00 by definition.*

## Speedup by prompt

| Prompt     | Baseline tok/s |   K=1 |   K=3 |   K=5 |   K=7 | Gate t=0.35 | Gate t=0.65 |                Best |
| ---------- | -------------: | ----: | ----: | ----: | ----: | ----------: | ----------: | ------------------: |
| algorithm  |           13.7 | 1.45x | 1.88x | 2.08x | 2.01x |       2.05x |       2.18x | Gate t=0.65 (2.18x) |
| datastruct |           13.7 | 1.48x | 2.06x | 2.37x | 2.48x |       2.46x |       2.47x |         K=7 (2.48x) |
| parsing    |           13.9 | 1.42x | 1.72x | 1.81x | 1.83x |       1.75x |       1.98x | Gate t=0.65 (1.98x) |
| regex      |           13.9 | 1.42x | 1.69x | 1.91x | 1.66x |       1.77x |       1.77x |         K=5 (1.91x) |
| async      |           14.0 | 1.43x | 1.82x | 1.87x | 2.00x |       1.86x |       2.21x | Gate t=0.65 (2.21x) |
| generator  |           13.7 | 1.45x | 1.87x | 1.96x | 1.91x |       2.00x |       2.43x | Gate t=0.65 (2.43x) |
| decorator  |           13.8 | 1.43x | 1.94x | 2.16x | 2.10x |       2.26x |       2.10x | Gate t=0.35 (2.26x) |
| numeric    |           13.7 | 1.43x | 1.91x | 2.03x | 2.02x |       2.15x |       2.17x | Gate t=0.65 (2.17x) |
| oop        |           13.8 | 1.47x | 2.02x | 2.29x | 2.38x |       2.39x |       2.42x | Gate t=0.65 (2.42x) |
| errors     |           13.9 | 1.46x | 1.92x | 2.11x | 2.01x |       1.91x |       2.12x | Gate t=0.65 (2.12x) |

## Acceptance rate (alpha) by prompt

| Prompt     |   K=1 |   K=3 |   K=5 |   K=7 | Gate t=0.35 | Gate t=0.65 |
| ---------- | ----: | ----: | ----: | ----: | ----------: | ----------: |
| algorithm  | 91.0% | 81.5% | 76.4% | 67.1% |       68.5% |       81.6% |
| datastruct | 95.4% | 91.6% | 89.5% | 84.4% |       82.4% |       88.8% |
| parsing    | 89.8% | 73.8% | 67.5% | 61.8% |       58.8% |       80.8% |
| regex      | 90.5% | 71.2% | 70.2% | 53.4% |       59.1% |       67.4% |
| async      | 90.8% | 79.1% | 68.3% | 68.3% |       62.1% |       87.0% |
| generator  | 90.1% | 83.3% | 71.8% | 62.5% |       66.5% |       91.7% |
| decorator  | 89.8% | 85.8% | 80.8% | 71.8% |       77.3% |       83.8% |
| numeric    | 89.1% | 83.5% | 74.2% | 69.0% |       71.8% |       80.2% |
| oop        | 97.2% | 92.0% | 88.1% | 83.6% |       82.3% |       91.4% |
| errors     | 93.5% | 84.3% | 80.0% | 69.0% |       64.3% |       82.5% |

## Dual-gate telemetry

Draft ceiling 8 tokens. The statistical gate fires when the draft's top-1 probability (read from the unwarped softmax) falls below the threshold; the syntactic gate fires on an unrecoverable bracket state inside fenced code. `Suppressed` counts fatal-looking bracket states seen in prose, where the syntactic gate deliberately stays silent because enumerations like "1)" are not syntax errors.

### Threshold t = 0.35

| Prompt     | Iters | Mean draft len | Statistical | Syntactic | Ungated | Gated % | Suppressed | Alpha | Speedup |
| ---------- | ----: | -------------: | ----------: | --------: | ------: | ------: | ---------: | ----: | ------: |
| algorithm  |    65 |           7.18 |           8 |         0 |      57 |     12% |          0 | 68.5% |   2.05x |
| datastruct |    52 |           7.77 |           3 |         0 |      49 |      6% |          0 | 82.4% |   2.46x |
| parsing    |    49 |           7.43 |           5 |         0 |      44 |     10% |          0 | 58.8% |   1.75x |
| regex      |    35 |           6.63 |           9 |         0 |      26 |     26% |          0 | 59.1% |   1.77x |
| async      |    46 |           7.70 |           5 |         0 |      41 |     11% |          0 | 62.1% |   1.86x |
| generator  |    67 |           7.13 |           8 |         0 |      59 |     12% |          0 | 66.5% |   2.00x |
| decorator  |    38 |           7.53 |           7 |         0 |      31 |     18% |          0 | 77.3% |   2.26x |
| numeric    |    56 |           7.91 |           2 |         0 |      54 |      4% |          0 | 71.8% |   2.15x |
| oop        |    38 |           7.74 |           4 |         0 |      34 |     11% |          0 | 82.3% |   2.39x |
| errors     |    50 |           7.74 |           5 |         0 |      45 |     10% |          0 | 64.3% |   1.91x |
| **total**  |   496 |           7.48 |          56 |         0 |     440 |     11% |          0 | 69.3% |   2.06x |

### Threshold t = 0.65

| Prompt     | Iters | Mean draft len | Statistical | Syntactic | Ungated | Gated % | Suppressed | Alpha | Speedup |
| ---------- | ----: | -------------: | ----------: | --------: | ------: | ------: | ---------: | ----: | ------: |
| algorithm  |    68 |           5.21 |          41 |         0 |      27 |     60% |          0 | 81.6% |   2.18x |
| datastruct |    60 |           6.10 |          24 |         0 |      36 |     40% |          0 | 88.8% |   2.47x |
| parsing    |    56 |           3.91 |          45 |         0 |      11 |     80% |          0 | 80.8% |   1.98x |
| regex      |    54 |           4.20 |          37 |         0 |      17 |     69% |          0 | 67.4% |   1.77x |
| async      |    48 |           4.79 |          32 |         0 |      16 |     67% |          0 | 87.0% |   2.21x |
| generator  |    30 |           5.60 |          17 |         0 |      13 |     57% |          0 | 91.7% |   2.43x |
| decorator  |    59 |           4.49 |          43 |         0 |      16 |     73% |          0 | 83.8% |   2.10x |
| numeric    |    49 |           5.37 |          30 |         0 |      19 |     61% |          0 | 80.2% |   2.17x |
| oop        |    45 |           5.67 |          25 |         0 |      20 |     56% |          0 | 91.4% |   2.42x |
| errors     |    59 |           5.03 |          38 |         0 |      21 |     64% |          0 | 82.5% |   2.12x |
| **total**  |   528 |           5.04 |         332 |         0 |     196 |     63% |          0 | 83.5% |   2.19x |

## Full results

| Prompt     |                        Configuration | Time (s) | Tokens | Tok/s | Speedup | Alpha | Fwd | Tok/fwd | Peak VRAM |  Code | EOS | Diverges @ |
| ---------- | -----------------------------------: | -------: | -----: | ----: | ------: | ----: | --: | ------: | --------: | ----: | --: | ---------: |
| algorithm  |                   Baseline (7B only) |    28.05 |    384 |  13.7 |   1.00x |    -- | 384 |    1.00 |     11.06 |  PASS |  no |         -- |
| algorithm  |                      Speculative K=1 |    19.25 |    383 |  19.9 |   1.45x | 91.0% | 201 |    1.91 |     11.06 |  PASS | yes |    tok 381 |
| algorithm  |                      Speculative K=3 |    14.84 |    383 |  25.8 |   1.88x | 81.5% | 112 |    3.42 |     11.06 |  PASS | yes |    tok 381 |
| algorithm  |                      Speculative K=5 |    13.10 |    374 |  28.5 |   2.08x | 76.4% |  78 |    4.79 |     11.07 |  PASS | yes |    tok 359 |
| algorithm  |                      Speculative K=7 |    13.90 |    383 |  27.5 |   2.01x | 67.1% |  69 |    5.55 |     11.07 |  PASS | yes |    tok 381 |
| algorithm  | Adaptive Dual-Gate (Max K=8, t=0.35) |    13.61 |    383 |  28.1 |   2.05x | 68.5% |  65 |    5.89 |     11.07 |  PASS | yes |    tok 381 |
| algorithm  | Adaptive Dual-Gate (Max K=8, t=0.65) |    11.80 |    353 |  29.9 |   2.18x | 81.6% |  68 |    5.19 |     11.06 |  PASS | yes |    tok 202 |
| datastruct |                   Baseline (7B only) |    28.00 |    384 |  13.7 |   1.00x |    -- | 384 |    1.00 |     11.06 | TRUNC |  no |         -- |
| datastruct |                      Speculative K=1 |    18.91 |    384 |  20.3 |   1.48x | 95.4% | 197 |    1.95 |     11.07 | TRUNC |  no |    tok 102 |
| datastruct |                      Speculative K=3 |    13.60 |    384 |  28.2 |   2.06x | 91.6% | 103 |    3.73 |     11.07 | TRUNC |  no |  identical |
| datastruct |                      Speculative K=5 |    11.81 |    384 |  32.5 |   2.37x | 89.5% |  71 |    5.41 |     11.07 | TRUNC |  no |    tok 341 |
| datastruct |                      Speculative K=7 |    11.28 |    384 |  34.0 |   2.48x | 84.4% |  56 |    6.86 |     11.07 | TRUNC |  no |    tok 341 |
| datastruct | Adaptive Dual-Gate (Max K=8, t=0.35) |    11.40 |    384 |  33.7 |   2.46x | 82.4% |  52 |    7.38 |     11.07 | TRUNC |  no |    tok 341 |
| datastruct | Adaptive Dual-Gate (Max K=8, t=0.65) |    11.32 |    384 |  33.9 |   2.47x | 88.8% |  60 |    6.40 |     11.07 | TRUNC |  no |    tok 326 |
| parsing    |                   Baseline (7B only) |    20.23 |    282 |  13.9 |   1.00x |    -- | 282 |    1.00 |     11.06 |  PASS | yes |         -- |
| parsing    |                      Speculative K=1 |    13.98 |    277 |  19.8 |   1.42x | 89.8% | 147 |    1.88 |     11.07 |  PASS | yes |    tok 158 |
| parsing    |                      Speculative K=3 |    10.59 |    254 |  24.0 |   1.72x | 73.8% |  80 |    3.17 |     11.07 |  PASS | yes |     tok 19 |
| parsing    |                      Speculative K=5 |    10.37 |    262 |  25.3 |   1.81x | 67.5% |  61 |    4.30 |     11.07 |  PASS | yes |    tok 158 |
| parsing    |                      Speculative K=7 |     9.97 |    254 |  25.5 |   1.83x | 61.8% |  49 |    5.18 |     11.07 |  PASS | yes |     tok 19 |
| parsing    | Adaptive Dual-Gate (Max K=8, t=0.35) |    10.45 |    255 |  24.4 |   1.75x | 58.8% |  49 |    5.20 |     11.07 |  PASS | yes |    tok 158 |
| parsing    | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.42 |    232 |  27.6 |   1.98x | 80.8% |  56 |    4.14 |     11.07 |  PASS | yes |     tok 19 |
| regex      |                   Baseline (7B only) |    11.23 |    156 |  13.9 |   1.00x |    -- | 156 |    1.00 |     11.06 |  PASS | yes |         -- |
| regex      |                      Speculative K=1 |    11.16 |    220 |  19.7 |   1.42x | 90.5% | 116 |    1.90 |     11.07 |  PASS | yes |     tok 97 |
| regex      |                      Speculative K=3 |     8.78 |    206 |  23.5 |   1.69x | 71.2% |  66 |    3.12 |     11.07 |  PASS | yes |     tok 97 |
| regex      |                      Speculative K=5 |     7.86 |    209 |  26.6 |   1.91x | 70.2% |  47 |    4.45 |     11.07 |  PASS | yes |     tok 97 |
| regex      |                      Speculative K=7 |     6.95 |    160 |  23.0 |   1.66x | 53.4% |  34 |    4.71 |     11.07 |  PASS | yes |     tok 76 |
| regex      | Adaptive Dual-Gate (Max K=8, t=0.35) |     6.97 |    171 |  24.6 |   1.77x | 59.1% |  35 |    4.89 |     11.08 |  PASS | yes |     tok 76 |
| regex      | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.39 |    206 |  24.5 |   1.77x | 67.4% |  54 |    3.81 |     11.07 |  PASS | yes |     tok 97 |
| async      |                   Baseline (7B only) |    19.23 |    269 |  14.0 |   1.00x |    -- | 269 |    1.00 |     11.07 |  PASS | yes |         -- |
| async      |                      Speculative K=1 |    13.43 |    268 |  20.0 |   1.43x | 90.8% | 141 |    1.90 |     11.07 |  PASS | yes |    tok 268 |
| async      |                      Speculative K=3 |    10.28 |    262 |  25.5 |   1.82x | 79.1% |  78 |    3.36 |     11.07 |  PASS | yes |    tok 129 |
| async      |                      Speculative K=5 |    10.04 |    262 |  26.1 |   1.87x | 68.3% |  60 |    4.37 |     11.07 |  PASS | yes |    tok 129 |
| async      |                      Speculative K=7 |     9.39 |    262 |  27.9 |   2.00x | 68.3% |  46 |    5.70 |     11.07 |  PASS | yes |    tok 144 |
| async      | Adaptive Dual-Gate (Max K=8, t=0.35) |    10.01 |    260 |  26.0 |   1.86x | 62.1% |  46 |    5.65 |     11.08 |  PASS | yes |    tok 129 |
| async      | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.95 |    246 |  30.9 |   2.21x | 87.0% |  48 |    5.12 |     11.07 |  PASS | yes |    tok 144 |
| generator  |                   Baseline (7B only) |    27.96 |    384 |  13.7 |   1.00x |    -- | 384 |    1.00 |     11.07 |  PASS |  no |         -- |
| generator  |                      Speculative K=1 |    19.33 |    384 |  19.9 |   1.45x | 90.1% | 202 |    1.90 |     11.07 |  PASS |  no |    tok 289 |
| generator  |                      Speculative K=3 |     6.65 |    171 |  25.7 |   1.87x | 83.3% |  50 |    3.42 |     11.08 |  PASS | yes |      tok 3 |
| generator  |                      Speculative K=5 |    14.26 |    384 |  26.9 |   1.96x | 71.8% |  85 |    4.52 |     11.08 |  PASS |  no |     tok 16 |
| generator  |                      Speculative K=7 |    14.63 |    384 |  26.2 |   1.91x | 62.5% |  72 |    5.33 |     11.08 |  PASS |  no |    tok 212 |
| generator  | Adaptive Dual-Gate (Max K=8, t=0.35) |    14.00 |    384 |  27.4 |   2.00x | 66.5% |  67 |    5.73 |     11.08 |  PASS |  no |     tok 16 |
| generator  | Adaptive Dual-Gate (Max K=8, t=0.65) |     5.40 |    180 |  33.3 |   2.43x | 91.7% |  30 |    6.00 |     11.08 |  PASS | yes |     tok 15 |
| decorator  |                   Baseline (7B only) |    18.84 |    260 |  13.8 |   1.00x |    -- | 260 |    1.00 |     11.07 |  PASS | yes |         -- |
| decorator  |                      Speculative K=1 |    13.12 |    259 |  19.7 |   1.43x | 89.8% | 137 |    1.89 |     11.07 |  PASS | yes |    tok 259 |
| decorator  |                      Speculative K=3 |     9.98 |    267 |  26.8 |   1.94x | 85.8% |  75 |    3.56 |     11.07 |  PASS | yes |     tok 42 |
| decorator  |                      Speculative K=5 |     8.75 |    261 |  29.8 |   2.16x | 80.8% |  52 |    5.02 |     11.07 |  PASS | yes |    tok 112 |
| decorator  |                      Speculative K=7 |     9.00 |    261 |  29.0 |   2.10x | 71.8% |  44 |    5.93 |     11.08 |  PASS | yes |    tok 112 |
| decorator  | Adaptive Dual-Gate (Max K=8, t=0.35) |     8.19 |    255 |  31.1 |   2.26x | 77.3% |  38 |    6.71 |     11.08 |  PASS | yes |     tok 48 |
| decorator  | Adaptive Dual-Gate (Max K=8, t=0.65) |     9.52 |    276 |  29.0 |   2.10x | 83.8% |  59 |    4.68 |     11.07 |  PASS | yes |    tok 102 |
| numeric    |                   Baseline (7B only) |    21.54 |    296 |  13.7 |   1.00x |    -- | 296 |    1.00 |     11.06 |  PASS | yes |         -- |
| numeric    |                      Speculative K=1 |    13.13 |    258 |  19.6 |   1.43x | 89.1% | 137 |    1.88 |     11.07 |  PASS | yes |     tok 12 |
| numeric    |                      Speculative K=3 |    11.25 |    295 |  26.2 |   1.91x | 83.5% |  85 |    3.47 |     11.07 |  PASS | yes |    tok 295 |
| numeric    |                      Speculative K=5 |     9.26 |    258 |  27.9 |   2.03x | 74.2% |  55 |    4.69 |     11.07 |  PASS | yes |     tok 12 |
| numeric    |                      Speculative K=7 |    10.70 |    297 |  27.8 |   2.02x | 69.0% |  52 |    5.71 |     11.07 |  PASS | yes |    tok 215 |
| numeric    | Adaptive Dual-Gate (Max K=8, t=0.35) |    12.42 |    367 |  29.5 |   2.15x | 71.8% |  56 |    6.55 |     11.08 |  PASS | yes |     tok 12 |
| numeric    | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.63 |    258 |  29.9 |   2.17x | 80.2% |  49 |    5.27 |     11.07 |  PASS | yes |     tok 12 |
| oop        |                   Baseline (7B only) |    20.17 |    279 |  13.8 |   1.00x |    -- | 279 |    1.00 |     11.06 |  PASS | yes |         -- |
| oop        |                      Speculative K=1 |    13.54 |    276 |  20.4 |   1.47x | 97.2% | 141 |    1.96 |     11.07 |  PASS | yes |     tok 52 |
| oop        |                      Speculative K=3 |     9.93 |    278 |  28.0 |   2.02x | 92.0% |  75 |    3.71 |     11.07 |  PASS | yes |    tok 278 |
| oop        |                      Speculative K=5 |     8.70 |    275 |  31.6 |   2.29x | 88.1% |  52 |    5.29 |     11.07 |  PASS | yes |    tok 208 |
| oop        |                      Speculative K=7 |     8.36 |    275 |  32.9 |   2.38x | 83.6% |  41 |    6.71 |     11.07 |  PASS | yes |    tok 208 |
| oop        | Adaptive Dual-Gate (Max K=8, t=0.35) |     8.31 |    275 |  33.1 |   2.39x | 82.3% |  38 |    7.24 |     11.07 |  PASS | yes |    tok 208 |
| oop        | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.16 |    273 |  33.5 |   2.42x | 91.4% |  45 |    6.07 |     11.07 |  PASS | yes |     tok 52 |
| errors     |                   Baseline (7B only) |    21.34 |    296 |  13.9 |   1.00x |    -- | 296 |    1.00 |     11.07 |  PASS | yes |         -- |
| errors     |                      Speculative K=1 |    14.76 |    299 |  20.3 |   1.46x | 93.5% | 155 |    1.93 |     11.07 |  PASS | yes |     tok 66 |
| errors     |                      Speculative K=3 |    11.25 |    299 |  26.6 |   1.92x | 84.3% |  85 |    3.52 |     11.07 |  PASS | yes |     tok 66 |
| errors     |                      Speculative K=5 |     9.45 |    276 |  29.2 |   2.11x | 80.0% |  56 |    4.93 |     11.07 |  PASS | yes |     tok 66 |
| errors     |                      Speculative K=7 |     9.82 |    274 |  27.9 |   2.01x | 69.0% |  48 |    5.71 |     11.07 |  PASS | yes |     tok 66 |
| errors     | Adaptive Dual-Gate (Max K=8, t=0.35) |    11.00 |    292 |  26.5 |   1.91x | 64.3% |  50 |    5.84 |     11.08 |  PASS | yes |     tok 66 |
| errors     | Adaptive Dual-Gate (Max K=8, t=0.65) |    10.05 |    296 |  29.4 |   2.12x | 82.5% |  59 |    5.02 |     11.08 |  PASS | yes |    tok 223 |

## Summary

- **Best configuration**: Adaptive Dual-Gate (Max K=8, t=0.65) at **2.19x** the baseline
- **Baseline throughput**: 13.8 tok/s
- **Mean acceptance rate**: 78.8%
- **Peak VRAM**: 11.08 GiB of 15.9 GiB
- **Generated code parses**: 63/70 runs (7 truncated by the token budget, not scored)
- **Output identical to baseline**: 1/60 greedy runs

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

- Divergence begins at token **150** on average (earliest 3, latest 381) of 384 generated.
- Both continuations remain valid Python; the Code column is scored independently of whether the wording matched.

The practical consequence: with a reduced-precision target, treat speculative decoding as distribution-preserving up to that precision, not as a bitwise-identical drop-in. The speedups above are unaffected -- they are throughput measurements, and both arms decode the same kind of text at the same budget.
