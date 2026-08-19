# Speculative Decoding Benchmark

- **Draft model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (bfloat16)
- **Target model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (8-bit, LLM.int8)
- **GPU**: NVIDIA GeForce RTX 5080 (15.9 GiB)
- **Sampling**: greedy (temperature 0)
- **Budget**: 256 new tokens (forced: EOS disabled, every run emits exactly this many), 10 prompts
- **Verification vocabulary**: 151936 tokens (shared prefix)
- **Baseline**: `model.generate()` on the 7B target alone

## Aggregate results

| Configuration                         | Time (s) | Tok/s | Speedup | Alpha | 7B fwd/token | Weights VRAM | Peak VRAM | Code OK |
| ------------------------------------- | -------: | ----: | ------: | ----: | -----------: | -----------: | --------: | ------: |
| Baseline (7B only)                    |    18.61 |  13.8 |   1.00x |    -- |         1.00 |     8.11 GiB |  8.19 GiB |   10/10 |
| Speculative K=5                       |     9.01 |  28.6 |   2.08x | 77.2% |         0.21 |    11.00 GiB | 11.08 GiB |   10/10 |
| Adaptive Dual-Gate (Max K=8, t=0.65)  |     8.61 |  30.2 |   2.20x | 82.8% |         0.20 |    11.00 GiB | 11.08 GiB |   10/10 |
| Self-Spec Twin-Cache (window=64, K=5) |    28.95 |   9.0 |   0.65x | 62.3% |         1.48 |     8.11 GiB |  8.20 GiB |   10/10 |

*Alpha is the fraction of drafted tokens accepted. `7B fwd/token` counts every forward pass of a **full-size** model per emitted token, which is the metric that makes the three architectures comparable: the two-model engines get their draft forwards from a 1.5B model (~4.4x cheaper, so excluded), while self-speculation drafts with the 7B itself, so its draft forwards cost full price and are counted. Plain decoding is 1.00 by definition; below 1.00 is a real win, above 1.00 means more full-size compute than simply decoding.*

## Speedup by prompt

| Prompt     | Baseline tok/s |   K=5 | Gate t=0.65 | Self-Spec |                Best |
| ---------- | -------------: | ----: | ----------: | --------: | ------------------: |
| algorithm  |           13.7 | 2.21x |       2.45x |     0.58x | Gate t=0.65 (2.45x) |
| datastruct |           13.7 | 2.44x |       2.62x |     0.83x | Gate t=0.65 (2.62x) |
| parsing    |           13.9 | 1.90x |       1.99x |     0.59x | Gate t=0.65 (1.99x) |
| regex      |           13.7 | 1.80x |       1.80x |     0.63x |         K=5 (1.80x) |
| async      |           13.9 | 1.86x |       2.25x |     0.58x | Gate t=0.65 (2.25x) |
| generator  |           13.7 | 2.02x |       1.74x |     0.62x |         K=5 (2.02x) |
| decorator  |           13.7 | 2.17x |       2.11x |     0.59x |         K=5 (2.17x) |
| numeric    |           13.7 | 2.00x |       2.18x |     0.63x | Gate t=0.65 (2.18x) |
| oop        |           13.7 | 2.30x |       2.46x |     0.80x | Gate t=0.65 (2.46x) |
| errors     |           13.8 | 2.11x |       2.36x |     0.67x | Gate t=0.65 (2.36x) |

## Acceptance rate (alpha) by prompt

| Prompt     |   K=5 | Gate t=0.65 | Self-Spec |
| ---------- | ----: | ----------: | --------: |
| algorithm  | 82.4% |       88.6% |     53.6% |
| datastruct | 92.5% |       94.4% |     85.6% |
| parsing    | 69.3% |       81.3% |     54.2% |
| regex      | 63.5% |       68.7% |     57.9% |
| async      | 68.3% |       87.3% |     53.4% |
| generator  | 75.2% |       63.3% |     57.6% |
| decorator  | 80.4% |       83.5% |     55.3% |
| numeric    | 74.0% |       80.0% |     59.8% |
| oop        | 87.1% |       91.1% |     81.1% |
| errors     | 79.2% |       90.2% |     64.5% |

## Dual-gate telemetry

Draft ceiling 8 tokens. The statistical gate fires when the draft's top-1 probability (read from the unwarped softmax) falls below the threshold; the syntactic gate fires on an unrecoverable bracket state inside fenced code. `Suppressed` counts fatal-looking bracket states seen in prose, where the syntactic gate deliberately stays silent because enumerations like "1)" are not syntax errors.

### Threshold t = 0.65

| Prompt     | Iters | Mean draft len | Statistical | Syntactic | Ungated | Gated % | Suppressed | Alpha | Speedup |
| ---------- | ----: | -------------: | ----------: | --------: | ------: | ------: | ---------: | ----: | ------: |
| algorithm  |    39 |           6.31 |          16 |         0 |      23 |     41% |          0 | 88.6% |   2.45x |
| datastruct |    36 |           6.47 |          12 |         0 |      24 |     33% |          0 | 94.4% |   2.62x |
| parsing    |    61 |           3.95 |          49 |         0 |      12 |     80% |          0 | 81.3% |   1.99x |
| regex      |    68 |           4.04 |          47 |         0 |      21 |     69% |          0 | 68.7% |   1.80x |
| async      |    50 |           4.72 |          34 |         0 |      16 |     68% |          0 | 87.3% |   2.25x |
| generator  |    66 |           4.55 |          40 |         0 |      26 |     61% |          0 | 63.3% |   1.74x |
| decorator  |    55 |           4.40 |          39 |         0 |      16 |     71% |          0 | 83.5% |   2.11x |
| numeric    |    49 |           5.31 |          30 |         0 |      19 |     61% |          0 | 80.0% |   2.18x |
| oop        |    42 |           5.62 |          23 |         0 |      19 |     55% |          0 | 91.1% |   2.46x |
| errors     |    46 |           5.09 |          29 |         0 |      17 |     63% |          0 | 90.2% |   2.36x |
| **total**  |   512 |           5.05 |         319 |         0 |     193 |     62% |          0 | 82.8% |   2.20x |

## Full results

| Prompt     |                         Configuration | Time (s) | Tokens | Tok/s | Speedup | Alpha | Fwd | Tok/fwd | Peak VRAM | Code | EOS | Diverges @ |
| ---------- | ------------------------------------: | -------: | -----: | ----: | ------: | ----: | --: | ------: | --------: | ---: | --: | ---------: |
| algorithm  |                    Baseline (7B only) |    18.63 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.18 | PASS |  no |         -- |
| algorithm  |                       Speculative K=5 |     8.43 |    256 |  30.4 |   2.21x | 82.4% |  50 |    5.12 |     11.07 | PASS |  no |  identical |
| algorithm  |  Adaptive Dual-Gate (Max K=8, t=0.65) |     7.61 |    256 |  33.7 |   2.45x | 88.6% |  39 |    6.56 |     11.06 | PASS |  no |    tok 202 |
| algorithm  | Self-Spec Twin-Cache (window=64, K=5) |    31.96 |    256 |   8.0 |   0.58x | 53.6% |  70 |    3.66 |      8.19 | PASS |  no |    tok 222 |
| datastruct |                    Baseline (7B only) |    18.65 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| datastruct |                       Speculative K=5 |     7.65 |    256 |  33.5 |   2.44x | 92.5% |  46 |    5.57 |     11.07 | PASS |  no |  identical |
| datastruct |  Adaptive Dual-Gate (Max K=8, t=0.65) |     7.11 |    256 |  36.0 |   2.62x | 94.4% |  36 |    7.11 |     11.07 | PASS |  no |  identical |
| datastruct | Self-Spec Twin-Cache (window=64, K=5) |    22.41 |    256 |  11.4 |   0.83x | 85.6% |  49 |    5.22 |      8.19 | PASS |  no |  identical |
| parsing    |                    Baseline (7B only) |    18.46 |    256 |  13.9 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| parsing    |                       Speculative K=5 |     9.73 |    256 |  26.3 |   1.90x | 69.3% |  58 |    4.41 |     11.07 | PASS |  no |    tok 158 |
| parsing    |  Adaptive Dual-Gate (Max K=8, t=0.65) |     9.26 |    256 |  27.6 |   1.99x | 81.3% |  61 |    4.20 |     11.07 | PASS |  no |     tok 19 |
| parsing    | Self-Spec Twin-Cache (window=64, K=5) |    31.28 |    256 |   8.2 |   0.59x | 54.2% |  69 |    3.71 |      8.19 | PASS |  no |     tok 19 |
| regex      |                    Baseline (7B only) |    18.73 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| regex      |                       Speculative K=5 |    10.38 |    256 |  24.7 |   1.80x | 63.5% |  62 |    4.13 |     11.07 | PASS |  no |     tok 97 |
| regex      |  Adaptive Dual-Gate (Max K=8, t=0.65) |    10.39 |    256 |  24.6 |   1.80x | 68.7% |  68 |    3.76 |     11.07 | PASS |  no |     tok 97 |
| regex      | Self-Spec Twin-Cache (window=64, K=5) |    29.85 |    256 |   8.6 |   0.63x | 57.9% |  66 |    3.88 |      8.19 | PASS |  no |     tok 97 |
| async      |                    Baseline (7B only) |    18.41 |    256 |  13.9 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| async      |                       Speculative K=5 |     9.88 |    256 |  25.9 |   1.86x | 68.3% |  59 |    4.34 |     11.07 | PASS |  no |    tok 129 |
| async      |  Adaptive Dual-Gate (Max K=8, t=0.65) |     8.19 |    256 |  31.3 |   2.25x | 87.3% |  50 |    5.12 |     11.07 | PASS |  no |    tok 144 |
| async      | Self-Spec Twin-Cache (window=64, K=5) |    31.76 |    256 |   8.1 |   0.58x | 53.4% |  70 |    3.66 |      8.20 | PASS |  no |     tok 30 |
| generator  |                    Baseline (7B only) |    18.67 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| generator  |                       Speculative K=5 |     9.23 |    256 |  27.7 |   2.02x | 75.2% |  54 |    4.74 |     11.08 | PASS |  no |     tok 16 |
| generator  |  Adaptive Dual-Gate (Max K=8, t=0.65) |    10.72 |    256 |  23.9 |   1.74x | 63.3% |  66 |    3.88 |     11.08 | PASS |  no |     tok 15 |
| generator  | Self-Spec Twin-Cache (window=64, K=5) |    30.29 |    256 |   8.5 |   0.62x | 57.6% |  67 |    3.82 |      8.20 | PASS |  no |    tok 212 |
| decorator  |                    Baseline (7B only) |    18.67 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| decorator  |                       Speculative K=5 |     8.61 |    256 |  29.7 |   2.17x | 80.4% |  51 |    5.02 |     11.07 | PASS |  no |    tok 112 |
| decorator  |  Adaptive Dual-Gate (Max K=8, t=0.65) |     8.83 |    256 |  29.0 |   2.11x | 83.5% |  55 |    4.65 |     11.07 | PASS |  no |    tok 102 |
| decorator  | Self-Spec Twin-Cache (window=64, K=5) |    31.51 |    256 |   8.1 |   0.59x | 55.3% |  69 |    3.71 |      8.20 | PASS |  no |    tok 102 |
| numeric    |                    Baseline (7B only) |    18.69 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| numeric    |                       Speculative K=5 |     9.33 |    256 |  27.4 |   2.00x | 74.0% |  55 |    4.65 |     11.07 | PASS |  no |     tok 12 |
| numeric    |  Adaptive Dual-Gate (Max K=8, t=0.65) |     8.58 |    256 |  29.8 |   2.18x | 80.0% |  49 |    5.22 |     11.08 | PASS |  no |     tok 12 |
| numeric    | Self-Spec Twin-Cache (window=64, K=5) |    29.47 |    256 |   8.7 |   0.63x | 59.8% |  65 |    3.94 |      8.20 | PASS |  no |     tok 12 |
| oop        |                    Baseline (7B only) |    18.65 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| oop        |                       Speculative K=5 |     8.11 |    256 |  31.6 |   2.30x | 87.1% |  48 |    5.33 |     11.07 | PASS |  no |    tok 208 |
| oop        |  Adaptive Dual-Gate (Max K=8, t=0.65) |     7.58 |    256 |  33.8 |   2.46x | 91.1% |  42 |    6.10 |     11.07 | PASS |  no |     tok 52 |
| oop        | Self-Spec Twin-Cache (window=64, K=5) |    23.31 |    256 |  11.0 |   0.80x | 81.1% |  51 |    5.02 |      8.19 | PASS |  no |    tok 208 |
| errors     |                    Baseline (7B only) |    18.51 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 | PASS |  no |         -- |
| errors     |                       Speculative K=5 |     8.76 |    256 |  29.2 |   2.11x | 79.2% |  52 |    4.92 |     11.07 | PASS |  no |     tok 66 |
| errors     |  Adaptive Dual-Gate (Max K=8, t=0.65) |     7.83 |    256 |  32.7 |   2.36x | 90.2% |  46 |    5.57 |     11.08 | PASS |  no |    tok 223 |
| errors     | Self-Spec Twin-Cache (window=64, K=5) |    27.68 |    256 |   9.2 |   0.67x | 64.5% |  61 |    4.20 |      8.20 | PASS |  no |    tok 159 |

## Summary

- **Best configuration**: Adaptive Dual-Gate (Max K=8, t=0.65) at **2.20x** the baseline
- **Baseline throughput**: 13.8 tok/s
- **Mean acceptance rate**: 74.1%
- **Peak VRAM**: 11.08 GiB of 15.9 GiB
- **Generated code parses**: 40/40 runs
- **Output identical to baseline**: 4/30 greedy runs

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

- Divergence begins at token **104** on average (earliest 12, latest 223) of 256 generated.
- Both continuations remain valid Python; the Code column is scored independently of whether the wording matched.

The practical consequence: with a reduced-precision target, treat speculative decoding as distribution-preserving up to that precision, not as a bitwise-identical drop-in. The speedups above are unaffected -- they are throughput measurements, and both arms decode the same kind of text at the same budget.
