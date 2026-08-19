# Speculative Decoding Benchmark

- **Draft model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (bfloat16)
- **Target model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (8-bit, LLM.int8)
- **GPU**: NVIDIA GeForce RTX 5080 (15.9 GiB)
- **Sampling**: greedy (temperature 0)
- **Budget**: 384 new tokens, 10 prompts
- **Verification vocabulary**: 151936 tokens (shared prefix)
- **Baseline**: `model.generate()` on the 7B target alone

## Aggregate results

| Configuration      | Time (s) | Tok/s | Speedup | Alpha | Fwd passes | Tok/fwd | Peak VRAM | Code OK |
| ------------------ | -------: | ----: | ------: | ----: | ---------: | ------: | --------: | ------: |
| Baseline (7B only) |    21.68 |  13.8 |   1.00x |    -- |      299.0 |    1.00 | 11.07 GiB |    9/10 |
| Speculative K=1    |    15.04 |  20.0 |   1.45x | 91.7% |      157.4 |    1.91 | 11.07 GiB |    9/10 |
| Speculative K=3    |    10.71 |  26.0 |   1.89x | 82.6% |       80.9 |    3.45 | 11.08 GiB |    9/10 |
| Speculative K=5    |    10.37 |  28.4 |   2.06x | 76.7% |       61.7 |    4.78 | 11.08 GiB |    9/10 |
| Speculative K=7    |    10.44 |  28.1 |   2.03x | 69.1% |       51.1 |    5.74 | 11.08 GiB |    9/10 |

*Alpha is the fraction of drafted tokens accepted. Tok/fwd is tokens emitted per target forward pass, the hardware-independent measure of the win: the baseline is exactly 1.00 by definition.*

## Speedup by prompt

| Prompt     | Baseline tok/s |   K=1 |   K=3 |   K=5 |   K=7 |        Best |
| ---------- | -------------: | ----: | ----: | ----: | ----: | ----------: |
| algorithm  |           13.7 | 1.45x | 1.88x | 2.07x | 2.01x | K=5 (2.07x) |
| datastruct |           13.7 | 1.48x | 2.06x | 2.36x | 2.46x | K=7 (2.46x) |
| parsing    |           13.9 | 1.42x | 1.72x | 1.84x | 1.82x | K=5 (1.84x) |
| regex      |           13.8 | 1.42x | 1.71x | 1.90x | 1.67x | K=5 (1.90x) |
| async      |           13.9 | 1.44x | 1.83x | 1.87x | 2.01x | K=7 (2.01x) |
| generator  |           13.7 | 1.45x | 1.88x | 1.97x | 1.91x | K=5 (1.97x) |
| decorator  |           13.8 | 1.43x | 1.94x | 2.16x | 2.10x | K=5 (2.16x) |
| numeric    |           13.7 | 1.44x | 1.91x | 2.04x | 2.02x | K=5 (2.04x) |
| oop        |           13.8 | 1.49x | 2.03x | 2.27x | 2.36x | K=7 (2.36x) |
| errors     |           13.9 | 1.46x | 1.92x | 2.11x | 2.00x | K=5 (2.11x) |

## Acceptance rate (alpha) by prompt

| Prompt     |   K=1 |   K=3 |   K=5 |   K=7 |
| ---------- | ----: | ----: | ----: | ----: |
| algorithm  | 91.0% | 81.5% | 76.4% | 67.1% |
| datastruct | 95.4% | 91.6% | 89.5% | 84.4% |
| parsing    | 89.8% | 73.8% | 67.5% | 61.8% |
| regex      | 90.5% | 71.2% | 70.2% | 53.4% |
| async      | 90.8% | 79.1% | 68.3% | 68.3% |
| generator  | 90.1% | 83.3% | 71.8% | 62.5% |
| decorator  | 89.8% | 85.8% | 80.8% | 71.8% |
| numeric    | 89.1% | 83.5% | 74.2% | 69.0% |
| oop        | 97.2% | 92.0% | 88.1% | 83.6% |
| errors     | 93.5% | 84.3% | 80.0% | 69.0% |

## Full results

| Prompt     |      Configuration | Time (s) | Tokens | Tok/s | Speedup | Alpha | Fwd | Tok/fwd | Peak VRAM |  Code | EOS | Diverges @ |
| ---------- | -----------------: | -------: | -----: | ----: | ------: | ----: | --: | ------: | --------: | ----: | --: | ---------: |
| algorithm  | Baseline (7B only) |    28.04 |    384 |  13.7 |   1.00x |    -- | 384 |    1.00 |     11.06 |  PASS |  no |         -- |
| algorithm  |    Speculative K=1 |    19.27 |    383 |  19.9 |   1.45x | 91.0% | 201 |    1.91 |     11.06 |  PASS | yes |    tok 381 |
| algorithm  |    Speculative K=3 |    14.89 |    383 |  25.7 |   1.88x | 81.5% | 112 |    3.42 |     11.06 |  PASS | yes |    tok 381 |
| algorithm  |    Speculative K=5 |    13.17 |    374 |  28.4 |   2.07x | 76.4% |  78 |    4.79 |     11.07 |  PASS | yes |    tok 359 |
| algorithm  |    Speculative K=7 |    13.92 |    383 |  27.5 |   2.01x | 67.1% |  69 |    5.55 |     11.07 |  PASS | yes |    tok 381 |
| datastruct | Baseline (7B only) |    27.96 |    384 |  13.7 |   1.00x |    -- | 384 |    1.00 |     11.06 | TRUNC |  no |         -- |
| datastruct |    Speculative K=1 |    18.84 |    384 |  20.4 |   1.48x | 95.4% | 197 |    1.95 |     11.07 | TRUNC |  no |    tok 102 |
| datastruct |    Speculative K=3 |    13.56 |    384 |  28.3 |   2.06x | 91.6% | 103 |    3.73 |     11.07 | TRUNC |  no |  identical |
| datastruct |    Speculative K=5 |    11.85 |    384 |  32.4 |   2.36x | 89.5% |  71 |    5.41 |     11.07 | TRUNC |  no |    tok 341 |
| datastruct |    Speculative K=7 |    11.37 |    384 |  33.8 |   2.46x | 84.4% |  56 |    6.86 |     11.07 | TRUNC |  no |    tok 341 |
| parsing    | Baseline (7B only) |    20.28 |    282 |  13.9 |   1.00x |    -- | 282 |    1.00 |     11.06 |  PASS | yes |         -- |
| parsing    |    Speculative K=1 |    14.03 |    277 |  19.7 |   1.42x | 89.8% | 147 |    1.88 |     11.07 |  PASS | yes |    tok 158 |
| parsing    |    Speculative K=3 |    10.62 |    254 |  23.9 |   1.72x | 73.8% |  80 |    3.17 |     11.07 |  PASS | yes |     tok 19 |
| parsing    |    Speculative K=5 |    10.26 |    262 |  25.5 |   1.84x | 67.5% |  61 |    4.30 |     11.07 |  PASS | yes |    tok 158 |
| parsing    |    Speculative K=7 |    10.03 |    254 |  25.3 |   1.82x | 61.8% |  49 |    5.18 |     11.07 |  PASS | yes |     tok 19 |
| regex      | Baseline (7B only) |    11.28 |    156 |  13.8 |   1.00x |    -- | 156 |    1.00 |     11.06 |  PASS | yes |         -- |
| regex      |    Speculative K=1 |    11.21 |    220 |  19.6 |   1.42x | 90.5% | 116 |    1.90 |     11.07 |  PASS | yes |     tok 97 |
| regex      |    Speculative K=3 |     8.72 |    206 |  23.6 |   1.71x | 71.2% |  66 |    3.12 |     11.07 |  PASS | yes |     tok 97 |
| regex      |    Speculative K=5 |     7.94 |    209 |  26.3 |   1.90x | 70.2% |  47 |    4.45 |     11.07 |  PASS | yes |     tok 97 |
| regex      |    Speculative K=7 |     6.95 |    160 |  23.0 |   1.67x | 53.4% |  34 |    4.71 |     11.07 |  PASS | yes |     tok 76 |
| async      | Baseline (7B only) |    19.31 |    269 |  13.9 |   1.00x |    -- | 269 |    1.00 |     11.07 |  PASS | yes |         -- |
| async      |    Speculative K=1 |    13.39 |    268 |  20.0 |   1.44x | 90.8% | 141 |    1.90 |     11.07 |  PASS | yes |    tok 268 |
| async      |    Speculative K=3 |    10.29 |    262 |  25.5 |   1.83x | 79.1% |  78 |    3.36 |     11.07 |  PASS | yes |    tok 129 |
| async      |    Speculative K=5 |    10.08 |    262 |  26.0 |   1.87x | 68.3% |  60 |    4.37 |     11.07 |  PASS | yes |    tok 129 |
| async      |    Speculative K=7 |     9.38 |    262 |  27.9 |   2.01x | 68.3% |  46 |    5.70 |     11.07 |  PASS | yes |    tok 144 |
| generator  | Baseline (7B only) |    27.98 |    384 |  13.7 |   1.00x |    -- | 384 |    1.00 |     11.07 |  PASS |  no |         -- |
| generator  |    Speculative K=1 |    19.33 |    384 |  19.9 |   1.45x | 90.1% | 202 |    1.90 |     11.07 |  PASS |  no |    tok 289 |
| generator  |    Speculative K=3 |     6.64 |    171 |  25.8 |   1.88x | 83.3% |  50 |    3.42 |     11.08 |  PASS | yes |      tok 3 |
| generator  |    Speculative K=5 |    14.23 |    384 |  27.0 |   1.97x | 71.8% |  85 |    4.52 |     11.08 |  PASS |  no |     tok 16 |
| generator  |    Speculative K=7 |    14.69 |    384 |  26.1 |   1.91x | 62.5% |  72 |    5.33 |     11.08 |  PASS |  no |    tok 212 |
| decorator  | Baseline (7B only) |    18.85 |    260 |  13.8 |   1.00x |    -- | 260 |    1.00 |     11.07 |  PASS | yes |         -- |
| decorator  |    Speculative K=1 |    13.11 |    259 |  19.8 |   1.43x | 89.8% | 137 |    1.89 |     11.07 |  PASS | yes |    tok 259 |
| decorator  |    Speculative K=3 |    10.00 |    267 |  26.7 |   1.94x | 85.8% |  75 |    3.56 |     11.07 |  PASS | yes |     tok 42 |
| decorator  |    Speculative K=5 |     8.76 |    261 |  29.8 |   2.16x | 80.8% |  52 |    5.02 |     11.07 |  PASS | yes |    tok 112 |
| decorator  |    Speculative K=7 |     9.02 |    261 |  28.9 |   2.10x | 71.8% |  44 |    5.93 |     11.08 |  PASS | yes |    tok 112 |
| numeric    | Baseline (7B only) |    21.60 |    296 |  13.7 |   1.00x |    -- | 296 |    1.00 |     11.06 |  PASS | yes |         -- |
| numeric    |    Speculative K=1 |    13.09 |    258 |  19.7 |   1.44x | 89.1% | 137 |    1.88 |     11.07 |  PASS | yes |     tok 12 |
| numeric    |    Speculative K=3 |    11.28 |    295 |  26.1 |   1.91x | 83.5% |  85 |    3.47 |     11.07 |  PASS | yes |    tok 295 |
| numeric    |    Speculative K=5 |     9.24 |    258 |  27.9 |   2.04x | 74.2% |  55 |    4.69 |     11.07 |  PASS | yes |     tok 12 |
| numeric    |    Speculative K=7 |    10.71 |    297 |  27.7 |   2.02x | 69.0% |  52 |    5.71 |     11.07 |  PASS | yes |    tok 215 |
| oop        | Baseline (7B only) |    20.18 |    279 |  13.8 |   1.00x |    -- | 279 |    1.00 |     11.06 |  PASS | yes |         -- |
| oop        |    Speculative K=1 |    13.42 |    276 |  20.6 |   1.49x | 97.2% | 141 |    1.96 |     11.07 |  PASS | yes |     tok 52 |
| oop        |    Speculative K=3 |     9.92 |    278 |  28.0 |   2.03x | 92.0% |  75 |    3.71 |     11.07 |  PASS | yes |    tok 278 |
| oop        |    Speculative K=5 |     8.75 |    275 |  31.4 |   2.27x | 88.1% |  52 |    5.29 |     11.07 |  PASS | yes |    tok 208 |
| oop        |    Speculative K=7 |     8.44 |    275 |  32.6 |   2.36x | 83.6% |  41 |    6.71 |     11.07 |  PASS | yes |    tok 208 |
| errors     | Baseline (7B only) |    21.30 |    296 |  13.9 |   1.00x |    -- | 296 |    1.00 |     11.07 |  PASS | yes |         -- |
| errors     |    Speculative K=1 |    14.76 |    299 |  20.3 |   1.46x | 93.5% | 155 |    1.93 |     11.07 |  PASS | yes |     tok 66 |
| errors     |    Speculative K=3 |    11.20 |    299 |  26.7 |   1.92x | 84.3% |  85 |    3.52 |     11.07 |  PASS | yes |     tok 66 |
| errors     |    Speculative K=5 |     9.40 |    276 |  29.3 |   2.11x | 80.0% |  56 |    4.93 |     11.07 |  PASS | yes |     tok 66 |
| errors     |    Speculative K=7 |     9.84 |    274 |  27.8 |   2.00x | 69.0% |  48 |    5.71 |     11.07 |  PASS | yes |     tok 66 |

## Summary

- **Best configuration**: K=5 at **2.06x** the baseline
- **Baseline throughput**: 13.8 tok/s
- **Mean acceptance rate**: 80.0%
- **Peak VRAM**: 11.08 GiB of 15.9 GiB
- **Generated code parses**: 45/50 runs (5 truncated by the token budget, not scored)
- **Output identical to baseline**: 1/40 greedy runs

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

- Divergence begins at token **160** on average (earliest 3, latest 381) of 384 generated.
- Both continuations remain valid Python; the Code column is scored independently of whether the wording matched.

The practical consequence: with a reduced-precision target, treat speculative decoding as distribution-preserving up to that precision, not as a bitwise-identical drop-in. The speedups above are unaffected -- they are throughput measurements, and both arms decode the same kind of text at the same budget.
