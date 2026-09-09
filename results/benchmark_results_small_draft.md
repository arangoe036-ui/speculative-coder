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
| Baseline (7B only)                   |    18.76 |  13.6 |   1.00x |    -- |         1.00 |     8.11 GiB |  8.19 GiB |   10/10 |
| Speculative K=5                      |     9.18 |  28.3 |   2.07x | 71.2% |         0.22 |     9.05 GiB |  9.12 GiB |    9/10 |
| Adaptive Dual-Gate (Max K=8, t=0.65) |     8.65 |  30.1 |   2.20x | 78.4% |         0.22 |     9.05 GiB |  9.12 GiB |   10/10 |

*Alpha is the fraction of drafted tokens accepted. `7B fwd/token` counts every forward pass of a **full-size** model per emitted token, which is the metric that makes the three architectures comparable: the two-model engines get their draft forwards from a 1.5B model (~4.4x cheaper, so excluded), while self-speculation drafts with the 7B itself, so its draft forwards cost full price and are counted. Plain decoding is 1.00 by definition; below 1.00 is a real win, above 1.00 means more full-size compute than simply decoding.*

## Speedup by prompt

| Prompt     | Baseline tok/s |   K=5 | Gate t=0.65 |                Best |
| ---------- | -------------: | ----: | ----------: | ------------------: |
| algorithm  |           13.7 | 2.30x |       2.51x | Gate t=0.65 (2.51x) |
| datastruct |           13.5 | 2.51x |       2.65x | Gate t=0.65 (2.65x) |
| parsing    |           13.8 | 1.69x |       1.74x | Gate t=0.65 (1.74x) |
| regex      |           13.6 | 1.73x |       1.85x | Gate t=0.65 (1.85x) |
| async      |           13.7 | 1.88x |       2.08x | Gate t=0.65 (2.08x) |
| generator  |           13.7 | 2.10x |       2.31x | Gate t=0.65 (2.31x) |
| decorator  |           13.7 | 2.11x |       2.16x | Gate t=0.65 (2.16x) |
| numeric    |           13.5 | 2.11x |       2.12x | Gate t=0.65 (2.12x) |
| oop        |           13.6 | 2.23x |       2.49x | Gate t=0.65 (2.49x) |
| errors     |           13.8 | 2.07x |       2.13x | Gate t=0.65 (2.13x) |

## Acceptance rate (alpha) by prompt

| Prompt     |   K=5 | Gate t=0.65 |
| ---------- | ----: | ----------: |
| algorithm  | 80.4% |       83.4% |
| datastruct | 89.7% |       85.2% |
| parsing    | 55.0% |       63.5% |
| regex      | 55.3% |       65.6% |
| async      | 63.7% |       79.4% |
| generator  | 73.1% |       83.1% |
| decorator  | 74.0% |       79.8% |
| numeric    | 72.6% |       76.4% |
| oop        | 76.6% |       86.9% |
| errors     | 71.8% |       80.7% |

## Dual-gate telemetry

Draft ceiling 8 tokens. The statistical gate fires when the draft's top-1 probability (read from the unwarped softmax) falls below the threshold; the syntactic gate fires on an unrecoverable bracket state inside fenced code. `Suppressed` counts fatal-looking bracket states seen in prose, where the syntactic gate deliberately stays silent because enumerations like "1)" are not syntax errors.

### Threshold t = 0.65

| Prompt     | Iters | Mean draft len | Statistical | Syntactic | Ungated | Gated % | Suppressed | Alpha | Speedup |
| ---------- | ----: | -------------: | ----------: | --------: | ------: | ------: | ---------: | ----: | ------: |
| algorithm  |    41 |           6.32 |          17 |         0 |      24 |     41% |          0 | 83.4% |   2.51x |
| datastruct |    38 |           6.74 |           9 |         0 |      29 |     24% |          0 | 85.2% |   2.65x |
| parsing    |    76 |           3.75 |          61 |         0 |      15 |     80% |          0 | 63.5% |   1.74x |
| regex      |    68 |           4.24 |          48 |         0 |      20 |     71% |          0 | 65.6% |   1.85x |
| async      |    64 |           3.80 |          51 |         0 |      13 |     80% |          0 | 79.4% |   2.08x |
| generator  |    51 |           4.86 |          34 |         0 |      17 |     67% |          0 | 83.1% |   2.31x |
| decorator  |    56 |           4.50 |          42 |         0 |      14 |     75% |          0 | 79.8% |   2.16x |
| numeric    |    59 |           4.37 |          46 |         0 |      13 |     78% |          0 | 76.4% |   2.12x |
| oop        |    45 |           5.42 |          27 |         0 |      18 |     60% |          0 | 86.9% |   2.49x |
| errors     |    60 |           4.07 |          49 |         0 |      11 |     82% |          0 | 80.7% |   2.13x |
| **total**  |   558 |           4.81 |         384 |         0 |     174 |     69% |          0 | 78.4% |   2.20x |

## Full results

| Prompt     |                        Configuration | Time (s) | Tokens | Tok/s | Speedup | Alpha | Fwd | Tok/fwd | Peak VRAM |  Code | EOS | Diverges @ |
| ---------- | -----------------------------------: | -------: | -----: | ----: | ------: | ----: | --: | ------: | --------: | ----: | --: | ---------: |
| algorithm  |                   Baseline (7B only) |    18.75 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.18 |  PASS |  no |         -- |
| algorithm  |                      Speculative K=5 |     8.15 |    256 |  31.4 |   2.30x | 80.4% |  51 |    5.02 |      9.11 |  PASS |  no |  identical |
| algorithm  | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.48 |    256 |  34.2 |   2.51x | 83.4% |  41 |    6.24 |      9.11 |  PASS |  no |  identical |
| datastruct |                   Baseline (7B only) |    18.93 |    256 |  13.5 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| datastruct |                      Speculative K=5 |     7.53 |    256 |  34.0 |   2.51x | 89.7% |  47 |    5.45 |      9.12 |  PASS |  no |  identical |
| datastruct | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.14 |    256 |  35.8 |   2.65x | 85.2% |  38 |    6.74 |      9.11 |  PASS |  no |  identical |
| parsing    |                   Baseline (7B only) |    18.62 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| parsing    |                      Speculative K=5 |    11.02 |    256 |  23.2 |   1.69x | 55.0% |  69 |    3.71 |      9.12 |  PASS |  no |    tok 158 |
| parsing    | Adaptive Dual-Gate (Max K=8, t=0.65) |    10.69 |    256 |  24.0 |   1.74x | 63.5% |  76 |    3.37 |      9.11 |  PASS |  no |     tok 19 |
| regex      |                   Baseline (7B only) |    18.85 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| regex      |                      Speculative K=5 |    10.93 |    256 |  23.4 |   1.73x | 55.3% |  68 |    3.76 |      9.12 | TRUNC |  no |    tok 116 |
| regex      | Adaptive Dual-Gate (Max K=8, t=0.65) |    10.21 |    256 |  25.1 |   1.85x | 65.6% |  68 |    3.76 |      9.11 |  PASS |  no |     tok 97 |
| async      |                   Baseline (7B only) |    18.65 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| async      |                      Speculative K=5 |     9.93 |    256 |  25.8 |   1.88x | 63.7% |  62 |    4.13 |      9.12 |  PASS |  no |    tok 222 |
| async      | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.95 |    256 |  28.6 |   2.08x | 79.4% |  64 |    4.00 |      9.12 |  PASS |  no |    tok 129 |
| generator  |                   Baseline (7B only) |    18.74 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| generator  |                      Speculative K=5 |     8.94 |    256 |  28.6 |   2.10x | 73.1% |  56 |    4.57 |      9.12 |  PASS |  no |  identical |
| generator  | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.13 |    256 |  31.5 |   2.31x | 83.1% |  51 |    5.02 |      9.12 |  PASS |  no |    tok 212 |
| decorator  |                   Baseline (7B only) |    18.70 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| decorator  |                      Speculative K=5 |     8.86 |    256 |  28.9 |   2.11x | 74.0% |  55 |    4.65 |      9.12 |  PASS |  no |    tok 102 |
| decorator  | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.64 |    256 |  29.6 |   2.16x | 79.8% |  56 |    4.57 |      9.12 |  PASS |  no |    tok 102 |
| numeric    |                   Baseline (7B only) |    18.97 |    256 |  13.5 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| numeric    |                      Speculative K=5 |     8.99 |    256 |  28.5 |   2.11x | 72.6% |  56 |    4.57 |      9.12 |  PASS |  no |     tok 12 |
| numeric    | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.94 |    256 |  28.6 |   2.12x | 76.4% |  59 |    4.34 |      9.12 |  PASS |  no |     tok 12 |
| oop        |                   Baseline (7B only) |    18.82 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| oop        |                      Speculative K=5 |     8.46 |    256 |  30.3 |   2.23x | 76.6% |  53 |    4.83 |      9.12 |  PASS |  no |     tok 52 |
| oop        | Adaptive Dual-Gate (Max K=8, t=0.65) |     7.57 |    256 |  33.8 |   2.49x | 86.9% |  45 |    5.69 |      9.12 |  PASS |  no |     tok 52 |
| errors     |                   Baseline (7B only) |    18.60 |    256 |  13.8 |   1.00x |    -- | 256 |    1.00 |      8.19 |  PASS |  no |         -- |
| errors     |                      Speculative K=5 |     8.97 |    256 |  28.5 |   2.07x | 71.8% |  56 |    4.57 |      9.12 |  PASS |  no |     tok 66 |
| errors     | Adaptive Dual-Gate (Max K=8, t=0.65) |     8.74 |    256 |  29.3 |   2.13x | 80.7% |  60 |    4.27 |      9.12 |  PASS |  no |     tok 66 |

## Summary

- **Best configuration**: Adaptive Dual-Gate (Max K=8, t=0.65) at **2.20x** the baseline
- **Baseline throughput**: 13.6 tok/s
- **Mean acceptance rate**: 74.8%
- **Peak VRAM**: 9.12 GiB of 15.9 GiB
- **Generated code parses**: 29/30 runs (1 truncated by the token budget, not scored)
- **Output identical to baseline**: 5/20 greedy runs

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

- Divergence begins at token **94** on average (earliest 12, latest 222) of 256 generated.
- Both continuations remain valid Python; the Code column is scored independently of whether the wording matched.

The practical consequence: with a reduced-precision target, treat speculative decoding as distribution-preserving up to that precision, not as a bitwise-identical drop-in. The speedups above are unaffected -- they are throughput measurements, and both arms decode the same kind of text at the same budget.
