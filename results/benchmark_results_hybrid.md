# Speculative Decoding Benchmark

- **Draft model**: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (bfloat16)
- **Target model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (8-bit, LLM.int8)
- **GPU**: NVIDIA GeForce RTX 5080 (15.9 GiB)
- **Sampling**: greedy (temperature 0)
- **Budget**: 256 new tokens (forced: EOS disabled, every run emits exactly this many), 10 prompts
- **Verification vocabulary**: 151936 tokens (shared prefix)
- **Baseline**: `model.generate()` on the 7B target alone

## Aggregate results

| Configuration                             | Time (s) | Tok/s | Speedup | Alpha | 7B fwd/token | Weights VRAM | Peak VRAM | Code OK |
| ----------------------------------------- | -------: | ----: | ------: | ----: | -----------: | -----------: | --------: | ------: |
| Baseline (7B only)                        |    18.81 |  13.6 |   1.00x |    -- |         1.00 |     8.11 GiB |  8.19 GiB |   10/10 |
| Adaptive Dual-Gate (Max K=8, t=0.65)      |     8.63 |  30.1 |   2.21x | 78.4% |         0.22 |     9.05 GiB |  9.12 GiB |   10/10 |
| Hybrid Cascade (n-gram m=2 + 0.5B + gate) |     8.73 |  29.8 |   2.19x | 69.7% |         0.24 |     9.05 GiB |  9.12 GiB |   10/10 |
| Hybrid Cascade (n-gram m=3 + 0.5B + gate) |     8.61 |  30.2 |   2.22x | 74.0% |         0.23 |     9.05 GiB |  9.12 GiB |   10/10 |
| Hybrid Cascade (n-gram m=4 + 0.5B + gate) |     8.57 |  30.4 |   2.23x | 76.6% |         0.22 |     9.05 GiB |  9.12 GiB |   10/10 |
| Hybrid Cascade (n-gram m=6 + 0.5B + gate) |     8.73 |  30.0 |   2.21x | 76.4% |         0.23 |     9.05 GiB |  9.12 GiB |    9/10 |

*Alpha is the fraction of drafted tokens accepted. `7B fwd/token` counts every forward pass of a **full-size** model per emitted token, which is the metric that makes the three architectures comparable: the two-model engines get their draft forwards from a 1.5B model (~4.4x cheaper, so excluded), while self-speculation drafts with the 7B itself, so its draft forwards cost full price and are counted. Plain decoding is 1.00 by definition; below 1.00 is a real win, above 1.00 means more full-size compute than simply decoding.*

## Speedup by prompt

| Prompt     | Baseline tok/s | Gate t=0.65 | Hybrid m=2 | Hybrid m=3 | Hybrid m=4 | Hybrid m=6 |                Best |
| ---------- | -------------: | ----------: | ---------: | ---------: | ---------: | ---------: | ------------------: |
| algorithm  |           13.5 |       2.54x |      2.35x |      2.49x |      2.58x |      2.54x |  Hybrid m=4 (2.58x) |
| datastruct |           13.5 |       2.64x |      2.61x |      2.70x |      2.67x |      2.72x |  Hybrid m=6 (2.72x) |
| parsing    |           13.7 |       1.75x |      1.73x |      1.71x |      1.76x |      1.75x |  Hybrid m=4 (1.76x) |
| regex      |           13.5 |       1.87x |      1.76x |      1.87x |      1.83x |      1.62x |  Hybrid m=3 (1.87x) |
| async      |           13.7 |       2.08x |      2.09x |      2.14x |      2.15x |      2.07x |  Hybrid m=4 (2.15x) |
| generator  |           13.6 |       2.33x |      2.24x |      2.32x |      2.30x |      2.31x | Gate t=0.65 (2.33x) |
| decorator  |           13.6 |       2.17x |      2.21x |      2.22x |      2.30x |      2.30x |  Hybrid m=4 (2.30x) |
| numeric    |           13.6 |       2.11x |      2.13x |      2.12x |      2.07x |      2.10x |  Hybrid m=2 (2.13x) |
| oop        |           13.7 |       2.49x |      2.62x |      2.57x |      2.48x |      2.50x |  Hybrid m=2 (2.62x) |
| errors     |           13.7 |       2.15x |      2.17x |      2.09x |      2.18x |      2.17x |  Hybrid m=4 (2.18x) |

## Acceptance rate (alpha) by prompt

| Prompt     | Gate t=0.65 | Hybrid m=2 | Hybrid m=3 | Hybrid m=4 | Hybrid m=6 |
| ---------- | ----------: | ---------: | ---------: | ---------: | ---------: |
| algorithm  |       83.4% |      69.6% |      79.3% |      83.9% |      82.4% |
| datastruct |       85.2% |      67.1% |      77.0% |      80.5% |      84.8% |
| parsing    |       63.5% |      55.4% |      59.3% |      62.5% |      63.6% |
| regex      |       65.6% |      54.6% |      60.4% |      61.8% |      54.2% |
| async      |       79.4% |      74.3% |      79.4% |      81.5% |      79.4% |
| generator  |       83.1% |      70.0% |      81.3% |      81.3% |      81.3% |
| decorator  |       79.8% |      74.4% |      75.9% |      78.8% |      78.8% |
| numeric    |       76.4% |      70.5% |      70.5% |      71.5% |      74.5% |
| oop        |       86.9% |      84.7% |      83.3% |      85.0% |      85.0% |
| errors     |       80.7% |      76.9% |      73.8% |      78.7% |      80.3% |

## Dual-gate telemetry

Draft ceiling 8 tokens. The statistical gate fires when the draft's top-1 probability (read from the unwarped softmax) falls below the threshold; the syntactic gate fires on an unrecoverable bracket state inside fenced code. `Suppressed` counts fatal-looking bracket states seen in prose, where the syntactic gate deliberately stays silent because enumerations like "1)" are not syntax errors.

### Threshold t = 0.65

| Prompt     | Iters | Mean draft len | Statistical | Syntactic | Ungated | Gated % | Suppressed | Alpha | Speedup |
| ---------- | ----: | -------------: | ----------: | --------: | ------: | ------: | ---------: | ----: | ------: |
| algorithm  |    41 |           6.32 |          17 |         0 |      24 |     41% |          0 | 83.4% |   2.54x |
| datastruct |    38 |           6.74 |           9 |         0 |      29 |     24% |          0 | 85.2% |   2.64x |
| parsing    |    76 |           3.75 |          61 |         0 |      15 |     80% |          0 | 63.5% |   1.75x |
| regex      |    68 |           4.24 |          48 |         0 |      20 |     71% |          0 | 65.6% |   1.87x |
| async      |    64 |           3.80 |          51 |         0 |      13 |     80% |          0 | 79.4% |   2.08x |
| generator  |    51 |           4.86 |          34 |         0 |      17 |     67% |          0 | 83.1% |   2.33x |
| decorator  |    56 |           4.50 |          42 |         0 |      14 |     75% |          0 | 79.8% |   2.17x |
| numeric    |    59 |           4.37 |          46 |         0 |      13 |     78% |          0 | 76.4% |   2.11x |
| oop        |    45 |           5.42 |          27 |         0 |      18 |     60% |          0 | 86.9% |   2.49x |
| errors     |    60 |           4.07 |          49 |         0 |      11 |     82% |          0 | 80.7% |   2.15x |
| **total**  |   558 |           4.81 |         384 |         0 |     174 |     69% |          0 | 78.4% |   2.21x |

## Hybrid cascade routing

`Fast %` is how often the free CPU n-gram drafter served a block; the two alpha columns are how often each drafter's proposals survived verification. They answer different questions, and a router can fire constantly while proposing badly. `Draft fwd` is GPU draft forwards actually spent, against `if all model` -- what pure model drafting would have cost. That counterfactual is estimated as iterations x the mean *model* block length observed in the same run, not the draft ceiling: the dual gate already shortens blocks well below the ceiling, so measuring against it would credit the n-gram router with the gate's savings.

### Pattern length m = 2

| Prompt     | Iters | n-gram blocks | Fast % | n-gram alpha | model alpha | Draft fwd | if all model | Saved | Speedup |
| ---------- | ----: | ------------: | -----: | -----------: | ----------: | --------: | -----------: | ----: | ------: |
| algorithm  |    51 |            10 |    20% |        12.0% |       81.3% |       246 |          306 |   20% |   2.35x |
| datastruct |    51 |            23 |    45% |        30.4% |       89.1% |       192 |          350 |   45% |   2.61x |
| parsing    |    82 |            12 |    15% |        21.7% |       63.3% |       256 |          300 |   15% |   1.73x |
| regex      |    80 |            13 |    16% |        42.6% |       57.4% |       263 |          314 |   16% |   1.76x |
| async      |    63 |             5 |     8% |        21.7% |       79.4% |       238 |          259 |    8% |   2.09x |
| generator  |    59 |            12 |    20% |        24.1% |       81.8% |       225 |          282 |   20% |   2.24x |
| decorator  |    58 |             7 |    12% |        51.4% |       77.9% |       231 |          263 |   12% |   2.21x |
| numeric    |    58 |             5 |     9% |         8.0% |       76.6% |       256 |          280 |    9% |   2.13x |
| oop        |    46 |             8 |    17% |        58.3% |       89.2% |       213 |          258 |   17% |   2.62x |
| errors     |    60 |             4 |     7% |        60.0% |       78.3% |       235 |          252 |    7% |   2.17x |
| **total**  |   608 |            99 |    16% |        31.5% |       77.4% |      2355 |         2864 |   18% |   2.19x |

### Pattern length m = 3

| Prompt     | Iters | n-gram blocks | Fast % | n-gram alpha | model alpha | Draft fwd | if all model | Saved | Speedup |
| ---------- | ----: | ------------: | -----: | -----------: | ----------: | --------: | -----------: | ----: | ------: |
| algorithm  |    46 |             5 |    11% |        32.0% |       84.2% |       241 |          270 |   11% |   2.49x |
| datastruct |    43 |            11 |    26% |        32.7% |       87.9% |       223 |          300 |   26% |   2.70x |
| parsing    |    79 |             4 |     5% |        35.0% |       61.1% |       280 |          295 |    5% |   1.71x |
| regex      |    74 |             9 |    12% |        68.2% |       59.1% |       259 |          295 |   12% |   1.87x |
| async      |    63 |             2 |     3% |         0.0% |       82.8% |       233 |          241 |    3% |   2.14x |
| generator  |    53 |             2 |     4% |        30.0% |       83.4% |       241 |          250 |    4% |   2.32x |
| decorator  |    58 |             6 |    10% |        60.0% |       77.9% |       231 |          258 |   10% |   2.22x |
| numeric    |    58 |             5 |     9% |         8.0% |       76.6% |       256 |          280 |    9% |   2.12x |
| oop        |    48 |             7 |    15% |        45.7% |       89.4% |       216 |          253 |   15% |   2.57x |
| errors     |    62 |             3 |     5% |        66.7% |       74.2% |       248 |          261 |    5% |   2.09x |
| **total**  |   584 |            54 |     9% |        41.6% |       77.7% |      2428 |         2703 |   10% |   2.22x |

### Pattern length m = 4

| Prompt     | Iters | n-gram blocks | Fast % | n-gram alpha | model alpha | Draft fwd | if all model | Saved | Speedup |
| ---------- | ----: | ------------: | -----: | -----------: | ----------: | --------: | -----------: | ----: | ------: |
| algorithm  |    43 |             2 |     5% |        60.0% |       84.9% |       245 |          257 |    5% |   2.58x |
| datastruct |    42 |             7 |    17% |        25.7% |       88.8% |       232 |          278 |   17% |   2.67x |
| parsing    |    77 |             2 |     3% |        60.0% |       62.6% |       278 |          285 |    2% |   1.76x |
| regex      |    73 |             4 |     5% |        55.0% |       62.3% |       276 |          292 |    5% |   1.83x |
| async      |    62 |             1 |     2% |         0.0% |       83.3% |       233 |          237 |    2% |   2.15x |
| generator  |    53 |             2 |     4% |        30.0% |       83.4% |       241 |          250 |    4% |   2.30x |
| decorator  |    55 |             5 |     9% |        76.0% |       79.1% |       230 |          253 |    9% |   2.30x |
| numeric    |    61 |             3 |     5% |        13.3% |       74.9% |       259 |          272 |    5% |   2.07x |
| oop        |    47 |             3 |     6% |        40.0% |       87.9% |       232 |          248 |    6% |   2.48x |
| errors     |    58 |             2 |     3% |       100.0% |       77.8% |       243 |          252 |    4% |   2.18x |
| **total**  |   571 |            31 |     5% |        46.5% |       78.5% |      2469 |         2624 |    6% |   2.23x |

### Pattern length m = 6

| Prompt     | Iters | n-gram blocks | Fast % | n-gram alpha | model alpha | Draft fwd | if all model | Saved | Speedup |
| ---------- | ----: | ------------: | -----: | -----------: | ----------: | --------: | -----------: | ----: | ------: |
| algorithm  |    42 |             1 |     2% |        40.0% |       83.2% |       256 |          262 |    2% |   2.54x |
| datastruct |    39 |             3 |     8% |        20.0% |       88.8% |       241 |          261 |    8% |   2.72x |
| parsing    |    77 |             1 |     1% |        60.0% |       63.7% |       278 |          282 |    1% |   1.75x |
| regex      |    89 |             6 |     7% |        72.0% |       52.6% |       285 |          306 |    7% |   1.62x |
| async      |    64 |             0 |     0% |         0.0% |       79.4% |       243 |          243 |    0% |   2.07x |
| generator  |    53 |             2 |     4% |        30.0% |       83.4% |       241 |          250 |    4% |   2.31x |
| decorator  |    55 |             5 |     9% |        76.0% |       79.1% |       230 |          253 |    9% |   2.30x |
| numeric    |    60 |             1 |     2% |        20.0% |       75.6% |       258 |          262 |    2% |   2.10x |
| oop        |    47 |             3 |     6% |        40.0% |       87.9% |       232 |          248 |    6% |   2.50x |
| errors     |    60 |             1 |     2% |        80.0% |       80.3% |       239 |          243 |    2% |   2.17x |
| **total**  |   586 |            23 |     4% |        53.6% |       77.4% |      2503 |         2610 |    4% |   2.21x |

## Full results

| Prompt     |                             Configuration | Time (s) | Tokens | Tok/s | Speedup | Alpha | Fwd | Tok/fwd | Peak VRAM |  Code | EOS | Diverges @ |
| ---------- | ----------------------------------------: | -------: | -----: | ----: | ------: | ----: | --: | ------: | --------: | ----: | --: | ---------: |
| algorithm  |                        Baseline (7B only) |    18.95 |    256 |  13.5 |   1.00x |    -- | 256 |    1.00 |      8.18 |  PASS |  no |         -- |
| algorithm  |      Adaptive Dual-Gate (Max K=8, t=0.65) |     7.45 |    256 |  34.4 |   2.54x | 83.4% |  41 |    6.24 |      9.11 |  PASS |  no |  identical |
| algorithm  | Hybrid Cascade (n-gram m=2 + 0.5B + gate) |     8.07 |    256 |  31.7 |   2.35x | 69.6% |  51 |    5.02 |      9.11 |  PASS |  no |  identical |
| algorithm  | Hybrid Cascade (n-gram m=3 + 0.5B + gate) |     7.60 |    256 |  33.7 |   2.49x | 79.3% |  46 |    5.57 |      9.11 |  PASS |  no |  identical |
| algorithm  | Hybrid Cascade (n-gram m=4 + 0.5B + gate) |     7.36 |    256 |  34.8 |   2.58x | 83.9% |  43 |    5.95 |      9.11 |  PASS |  no |  identical |
| algorithm  | Hybrid Cascade (n-gram m=6 + 0.5B + gate) |     7.47 |    256 |  34.3 |   2.54x | 82.4% |  42 |    6.10 |      9.11 |  PASS |  no |  identical |
| datastruct |                        Baseline (7B only) |    18.99 |    256 |  13.5 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| datastruct |      Adaptive Dual-Gate (Max K=8, t=0.65) |     7.19 |    256 |  35.6 |   2.64x | 85.2% |  38 |    6.74 |      9.11 |  PASS |  no |  identical |
| datastruct | Hybrid Cascade (n-gram m=2 + 0.5B + gate) |     7.26 |    256 |  35.3 |   2.61x | 67.1% |  51 |    5.02 |      9.11 |  PASS |  no |  identical |
| datastruct | Hybrid Cascade (n-gram m=3 + 0.5B + gate) |     7.02 |    256 |  36.5 |   2.70x | 77.0% |  43 |    5.95 |      9.11 |  PASS |  no |  identical |
| datastruct | Hybrid Cascade (n-gram m=4 + 0.5B + gate) |     7.11 |    256 |  36.0 |   2.67x | 80.5% |  42 |    6.10 |      9.11 |  PASS |  no |  identical |
| datastruct | Hybrid Cascade (n-gram m=6 + 0.5B + gate) |     6.99 |    256 |  36.6 |   2.72x | 84.8% |  39 |    6.56 |      9.11 |  PASS |  no |  identical |
| parsing    |                        Baseline (7B only) |    18.72 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| parsing    |      Adaptive Dual-Gate (Max K=8, t=0.65) |    10.72 |    256 |  23.9 |   1.75x | 63.5% |  76 |    3.37 |      9.11 |  PASS |  no |     tok 19 |
| parsing    | Hybrid Cascade (n-gram m=2 + 0.5B + gate) |    10.81 |    256 |  23.7 |   1.73x | 55.4% |  82 |    3.12 |      9.11 |  PASS |  no |     tok 19 |
| parsing    | Hybrid Cascade (n-gram m=3 + 0.5B + gate) |    10.95 |    256 |  23.4 |   1.71x | 59.3% |  79 |    3.24 |      9.11 |  PASS |  no |     tok 19 |
| parsing    | Hybrid Cascade (n-gram m=4 + 0.5B + gate) |    10.64 |    256 |  24.1 |   1.76x | 62.5% |  77 |    3.32 |      9.11 |  PASS |  no |     tok 19 |
| parsing    | Hybrid Cascade (n-gram m=6 + 0.5B + gate) |    10.68 |    256 |  24.0 |   1.75x | 63.6% |  77 |    3.32 |      9.11 |  PASS |  no |     tok 19 |
| regex      |                        Baseline (7B only) |    18.94 |    256 |  13.5 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| regex      |      Adaptive Dual-Gate (Max K=8, t=0.65) |    10.13 |    256 |  25.3 |   1.87x | 65.6% |  68 |    3.76 |      9.11 |  PASS |  no |     tok 97 |
| regex      | Hybrid Cascade (n-gram m=2 + 0.5B + gate) |    10.73 |    256 |  23.9 |   1.76x | 54.6% |  80 |    3.20 |      9.11 |  PASS |  no |     tok 97 |
| regex      | Hybrid Cascade (n-gram m=3 + 0.5B + gate) |    10.12 |    256 |  25.3 |   1.87x | 60.4% |  74 |    3.46 |      9.11 |  PASS |  no |     tok 97 |
| regex      | Hybrid Cascade (n-gram m=4 + 0.5B + gate) |    10.36 |    256 |  24.7 |   1.83x | 61.8% |  73 |    3.51 |      9.11 |  PASS |  no |     tok 97 |
| regex      | Hybrid Cascade (n-gram m=6 + 0.5B + gate) |    11.68 |    256 |  21.9 |   1.62x | 54.2% |  89 |    2.88 |      9.11 | TRUNC |  no |     tok 76 |
| async      |                        Baseline (7B only) |    18.68 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| async      |      Adaptive Dual-Gate (Max K=8, t=0.65) |     8.98 |    256 |  28.5 |   2.08x | 79.4% |  64 |    4.00 |      9.12 |  PASS |  no |    tok 129 |
| async      | Hybrid Cascade (n-gram m=2 + 0.5B + gate) |     8.94 |    256 |  28.6 |   2.09x | 74.3% |  63 |    4.06 |      9.12 |  PASS |  no |    tok 223 |
| async      | Hybrid Cascade (n-gram m=3 + 0.5B + gate) |     8.73 |    256 |  29.3 |   2.14x | 79.4% |  63 |    4.06 |      9.12 |  PASS |  no |    tok 146 |
| async      | Hybrid Cascade (n-gram m=4 + 0.5B + gate) |     8.70 |    256 |  29.4 |   2.15x | 81.5% |  62 |    4.13 |      9.12 |  PASS |  no |    tok 146 |
| async      | Hybrid Cascade (n-gram m=6 + 0.5B + gate) |     9.04 |    256 |  28.3 |   2.07x | 79.4% |  64 |    4.00 |      9.12 |  PASS |  no |    tok 129 |
| generator  |                        Baseline (7B only) |    18.83 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| generator  |      Adaptive Dual-Gate (Max K=8, t=0.65) |     8.10 |    256 |  31.6 |   2.33x | 83.1% |  51 |    5.02 |      9.12 |  PASS |  no |    tok 212 |
| generator  | Hybrid Cascade (n-gram m=2 + 0.5B + gate) |     8.41 |    256 |  30.4 |   2.24x | 70.0% |  59 |    4.34 |      9.12 |  PASS |  no |  identical |
| generator  | Hybrid Cascade (n-gram m=3 + 0.5B + gate) |     8.12 |    256 |  31.5 |   2.32x | 81.3% |  53 |    4.83 |      9.12 |  PASS |  no |  identical |
| generator  | Hybrid Cascade (n-gram m=4 + 0.5B + gate) |     8.17 |    256 |  31.3 |   2.30x | 81.3% |  53 |    4.83 |      9.12 |  PASS |  no |  identical |
| generator  | Hybrid Cascade (n-gram m=6 + 0.5B + gate) |     8.14 |    256 |  31.4 |   2.31x | 81.3% |  53 |    4.83 |      9.12 |  PASS |  no |  identical |
| decorator  |                        Baseline (7B only) |    18.80 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| decorator  |      Adaptive Dual-Gate (Max K=8, t=0.65) |     8.65 |    256 |  29.6 |   2.17x | 79.8% |  56 |    4.57 |      9.12 |  PASS |  no |    tok 102 |
| decorator  | Hybrid Cascade (n-gram m=2 + 0.5B + gate) |     8.50 |    256 |  30.1 |   2.21x | 74.4% |  58 |    4.41 |      9.12 |  PASS |  no |     tok 48 |
| decorator  | Hybrid Cascade (n-gram m=3 + 0.5B + gate) |     8.46 |    256 |  30.3 |   2.22x | 75.9% |  58 |    4.41 |      9.12 |  PASS |  no |     tok 48 |
| decorator  | Hybrid Cascade (n-gram m=4 + 0.5B + gate) |     8.17 |    256 |  31.3 |   2.30x | 78.8% |  55 |    4.65 |      9.12 |  PASS |  no |     tok 48 |
| decorator  | Hybrid Cascade (n-gram m=6 + 0.5B + gate) |     8.18 |    256 |  31.3 |   2.30x | 78.8% |  55 |    4.65 |      9.12 |  PASS |  no |     tok 48 |
| numeric    |                        Baseline (7B only) |    18.80 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| numeric    |      Adaptive Dual-Gate (Max K=8, t=0.65) |     8.89 |    256 |  28.8 |   2.11x | 76.4% |  59 |    4.34 |      9.12 |  PASS |  no |     tok 12 |
| numeric    | Hybrid Cascade (n-gram m=2 + 0.5B + gate) |     8.84 |    256 |  29.0 |   2.13x | 70.5% |  58 |    4.41 |      9.12 |  PASS |  no |     tok 12 |
| numeric    | Hybrid Cascade (n-gram m=3 + 0.5B + gate) |     8.87 |    256 |  28.9 |   2.12x | 70.5% |  58 |    4.41 |      9.12 |  PASS |  no |     tok 12 |
| numeric    | Hybrid Cascade (n-gram m=4 + 0.5B + gate) |     9.08 |    256 |  28.2 |   2.07x | 71.5% |  61 |    4.20 |      9.12 |  PASS |  no |     tok 12 |
| numeric    | Hybrid Cascade (n-gram m=6 + 0.5B + gate) |     8.96 |    256 |  28.6 |   2.10x | 74.5% |  60 |    4.27 |      9.12 |  PASS |  no |     tok 12 |
| oop        |                        Baseline (7B only) |    18.73 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| oop        |      Adaptive Dual-Gate (Max K=8, t=0.65) |     7.51 |    256 |  34.1 |   2.49x | 86.9% |  45 |    5.69 |      9.12 |  PASS |  no |     tok 52 |
| oop        | Hybrid Cascade (n-gram m=2 + 0.5B + gate) |     7.14 |    256 |  35.9 |   2.62x | 84.7% |  46 |    5.57 |      9.12 |  PASS |  no |    tok 208 |
| oop        | Hybrid Cascade (n-gram m=3 + 0.5B + gate) |     7.30 |    256 |  35.1 |   2.57x | 83.3% |  48 |    5.33 |      9.12 |  PASS |  no |  identical |
| oop        | Hybrid Cascade (n-gram m=4 + 0.5B + gate) |     7.54 |    256 |  34.0 |   2.48x | 85.0% |  47 |    5.45 |      9.12 |  PASS |  no |     tok 52 |
| oop        | Hybrid Cascade (n-gram m=6 + 0.5B + gate) |     7.49 |    256 |  34.2 |   2.50x | 85.0% |  47 |    5.45 |      9.12 |  PASS |  no |     tok 52 |
| errors     |                        Baseline (7B only) |    18.70 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| errors     |      Adaptive Dual-Gate (Max K=8, t=0.65) |     8.71 |    256 |  29.4 |   2.15x | 80.7% |  60 |    4.27 |      9.12 |  PASS |  no |     tok 66 |
| errors     | Hybrid Cascade (n-gram m=2 + 0.5B + gate) |     8.61 |    256 |  29.8 |   2.17x | 76.9% |  60 |    4.27 |      9.12 |  PASS |  no |     tok 66 |
| errors     | Hybrid Cascade (n-gram m=3 + 0.5B + gate) |     8.97 |    256 |  28.5 |   2.09x | 73.8% |  62 |    4.13 |      9.12 |  PASS |  no |     tok 66 |
| errors     | Hybrid Cascade (n-gram m=4 + 0.5B + gate) |     8.56 |    256 |  29.9 |   2.18x | 78.7% |  58 |    4.41 |      9.12 |  PASS |  no |     tok 66 |
| errors     | Hybrid Cascade (n-gram m=6 + 0.5B + gate) |     8.63 |    256 |  29.7 |   2.17x | 80.3% |  60 |    4.27 |      9.12 |  PASS |  no |     tok 66 |

## Summary

- **Best configuration**: Hybrid Cascade (n-gram m=4 + 0.5B + gate) at **2.23x** the baseline
- **Baseline throughput**: 13.6 tok/s
- **Mean acceptance rate**: 75.0%
- **Peak VRAM**: 9.12 GiB of 15.9 GiB
- **Generated code parses**: 59/60 runs (1 truncated by the token budget, not scored)
- **Output identical to baseline**: 15/50 greedy runs

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

- Divergence begins at token **74** on average (earliest 12, latest 223) of 256 generated.
- Both continuations remain valid Python; the Code column is scored independently of whether the wording matched.

The practical consequence: with a reduced-precision target, treat speculative decoding as distribution-preserving up to that precision, not as a bitwise-identical drop-in. The speedups above are unaffected -- they are throughput measurements, and both arms decode the same kind of text at the same budget.
