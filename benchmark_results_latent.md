# Speculative Decoding Benchmark

- **Draft model**: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (bfloat16)
- **Target model**: `Qwen/Qwen2.5-Coder-7B-Instruct` (8-bit, LLM.int8)
- **GPU**: NVIDIA GeForce RTX 5080 (15.9 GiB)
- **Sampling**: greedy (temperature 0)
- **Budget**: 256 new tokens (forced: EOS disabled, every run emits exactly this many), 10 prompts
- **Verification vocabulary**: 151936 tokens (shared prefix)
- **Baseline**: `model.generate()` on the 7B target alone

## Aggregate results

| Configuration                                              | Time (s) | Tok/s | Speedup | Alpha | 7B fwd/token | Draft rows/tok | Peak VRAM | Code OK |
| ---------------------------------------------------------- | -------: | ----: | ------: | ----: | -----------: | -------------: | --------: | ------: |
| Baseline (7B only)                                         |    18.76 |  13.6 |   1.00x |    -- |         1.00 |             -- |  8.63 GiB |   10/10 |
| Evolutionary Tree (leaves<=8, K=8, split<0.8)              |     7.98 |  32.7 |   2.40x | 77.1% |         0.14 |           2.98 |  9.86 GiB |    9/10 |
| Latent Mitosis Tree (EAGLE-2, leaves<=16, K=8, split<0.85) |    20.87 |  12.4 |   0.91x |  6.1% |         0.69 |          85.80 |  9.23 GiB |   10/10 |

*Alpha is the fraction of drafted tokens accepted. `7B fwd/token` counts every forward pass of a **full-size** model per emitted token, which is the metric that makes the three architectures comparable: the two-model engines get their draft forwards from a 1.5B model (~4.4x cheaper, so excluded), while self-speculation drafts with the 7B itself, so its draft forwards cost full price and are counted. Plain decoding is 1.00 by definition; below 1.00 is a real win, above 1.00 means more full-size compute than simply decoding. `Draft rows/tok` is draft-model batch rows pushed per emitted token -- the draft-side cost. Counting draft forward *launches* would hide the difference, since every breadth engine launches once per depth step and only the widths differ.*

## Speedup by prompt

| Prompt     | Baseline tok/s | Tree s<0.8 | Latent L=16 |               Best |
| ---------- | -------------: | ---------: | ----------: | -----------------: |
| algorithm  |           13.6 |      2.79x |       0.91x | Tree s<0.8 (2.79x) |
| datastruct |           13.6 |      2.77x |       1.04x | Tree s<0.8 (2.77x) |
| parsing    |           13.7 |      1.91x |       0.85x | Tree s<0.8 (1.91x) |
| regex      |           13.5 |      2.00x |       0.79x | Tree s<0.8 (2.00x) |
| async      |           13.7 |      2.32x |       0.82x | Tree s<0.8 (2.32x) |
| generator  |           13.6 |      2.54x |       0.85x | Tree s<0.8 (2.54x) |
| decorator  |           13.7 |      1.98x |       0.91x | Tree s<0.8 (1.98x) |
| numeric    |           13.6 |      2.68x |       1.03x | Tree s<0.8 (2.68x) |
| oop        |           13.7 |      2.57x |       0.94x | Tree s<0.8 (2.57x) |
| errors     |           13.7 |      2.39x |       0.91x | Tree s<0.8 (2.39x) |

## Acceptance rate (alpha) by prompt

| Prompt     | Tree s<0.8 | Latent L=16 |
| ---------- | ---------: | ----------: |
| algorithm  |      91.1% |        6.0% |
| datastruct |      87.5% |        8.7% |
| parsing    |      60.2% |        5.0% |
| regex      |      61.6% |        3.5% |
| async      |      75.3% |        4.5% |
| generator  |      82.9% |        4.8% |
| decorator  |      63.1% |        6.3% |
| numeric    |      88.2% |        8.6% |
| oop        |      83.5% |        6.8% |
| errors     |      77.8% |        6.5% |

## Tree telemetry: token-space vs latent-space mitosis

`Leaves` is the mean width the tree grew to, against a ceiling of 8. `Draft rows` totals the batch rows pushed through the draft model, with the fixed-width Monte Carlo figure alongside for comparison -- that reduction is the design's central claim. `Best` and `Gain` carry the same meaning as in the breadth table: the accepted count of the winning leaf, and how much taking the max over leaves buys.

| Config                      | Iters | Leaves | Splits | Draft rows | Rows/tok | Best |  Gain | Tok/fwd | Tok/s | Speedup |
| --------------------------- | ----: | -----: | -----: | ---------: | -------: | ---: | ----: | ------: | ----: | ------: |
| Tree split<0.8              |   367 |   4.72 |   1401 |       7620 |     2.98 | 6.12 | +2.16 |    7.09 |  32.7 |   2.40x |
| Latent Mitosis (leaves<=16) |  1746 |  15.85 |  25932 |     219648 |    85.80 | 0.48 |    -- |    1.47 |  12.4 |   0.91x |

## Full results

| Prompt     |                                              Configuration | Time (s) | Tokens | Tok/s | Speedup | Alpha | Fwd | Tok/fwd | Peak VRAM |  Code | EOS | Diverges @ |
| ---------- | ---------------------------------------------------------: | -------: | -----: | ----: | ------: | ----: | --: | ------: | --------: | ----: | --: | ---------: |
| algorithm  |                                         Baseline (7B only) |    18.79 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |      8.62 |  PASS |  no |         -- |
| algorithm  |              Evolutionary Tree (leaves<=8, K=8, split<0.8) |     6.73 |    256 |  38.0 |   2.79x | 91.1% |  31 |    8.26 |      9.83 |  PASS |  no |  identical |
| algorithm  | Latent Mitosis Tree (EAGLE-2, leaves<=16, K=8, split<0.85) |    20.74 |    256 |  12.3 |   0.91x |  6.0% | 174 |    1.47 |      9.21 |  PASS |  no |  identical |
| datastruct |                                         Baseline (7B only) |    18.89 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |      8.62 |  PASS |  no |         -- |
| datastruct |              Evolutionary Tree (leaves<=8, K=8, split<0.8) |     6.81 |    256 |  37.6 |   2.77x | 87.5% |  32 |    8.00 |      9.73 |  PASS |  no |  identical |
| datastruct | Latent Mitosis Tree (EAGLE-2, leaves<=16, K=8, split<0.85) |    18.22 |    256 |  14.1 |   1.04x |  8.7% | 153 |    1.67 |      9.23 |  PASS |  no |  identical |
| parsing    |                                         Baseline (7B only) |    18.66 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.62 |  PASS |  no |         -- |
| parsing    |              Evolutionary Tree (leaves<=8, K=8, split<0.8) |     9.75 |    256 |  26.3 |   1.91x | 60.2% |  44 |    5.82 |      9.86 |  PASS |  no |     tok 19 |
| parsing    | Latent Mitosis Tree (EAGLE-2, leaves<=16, K=8, split<0.85) |    21.89 |    256 |  11.7 |   0.85x |  5.0% | 185 |    1.38 |      9.23 |  PASS |  no |    tok 158 |
| regex      |                                         Baseline (7B only) |    18.98 |    256 |  13.5 |   1.00x |    -- | 256 |    1.00 |      8.62 |  PASS |  no |         -- |
| regex      |              Evolutionary Tree (leaves<=8, K=8, split<0.8) |     9.49 |    256 |  27.0 |   2.00x | 61.6% |  44 |    5.82 |      9.84 | TRUNC |  no |     tok 97 |
| regex      | Latent Mitosis Tree (EAGLE-2, leaves<=16, K=8, split<0.85) |    23.99 |    256 |  10.7 |   0.79x |  3.5% | 202 |    1.27 |      9.23 |  PASS |  no |     tok 97 |
| async      |                                         Baseline (7B only) |    18.64 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.62 |  PASS |  no |         -- |
| async      |              Evolutionary Tree (leaves<=8, K=8, split<0.8) |     8.03 |    256 |  31.9 |   2.32x | 75.3% |  37 |    6.92 |      9.85 |  PASS |  no |  identical |
| async      | Latent Mitosis Tree (EAGLE-2, leaves<=16, K=8, split<0.85) |    22.59 |    256 |  11.3 |   0.82x |  4.5% | 190 |    1.35 |      9.22 |  PASS |  no |    tok 144 |
| generator  |                                         Baseline (7B only) |    18.84 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |      8.63 |  PASS |  no |         -- |
| generator  |              Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.42 |    256 |  34.5 |   2.54x | 82.9% |  34 |    7.53 |      9.84 |  PASS |  no |     tok 16 |
| generator  | Latent Mitosis Tree (EAGLE-2, leaves<=16, K=8, split<0.85) |    22.23 |    256 |  11.5 |   0.85x |  4.8% | 187 |    1.37 |      9.23 |  PASS |  no |      tok 3 |
| decorator  |                                         Baseline (7B only) |    18.70 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.62 |  PASS |  no |         -- |
| decorator  |              Evolutionary Tree (leaves<=8, K=8, split<0.8) |     9.42 |    256 |  27.2 |   1.98x | 63.1% |  43 |    5.95 |      9.85 |  PASS |  no |     tok 48 |
| decorator  | Latent Mitosis Tree (EAGLE-2, leaves<=16, K=8, split<0.85) |    20.56 |    256 |  12.5 |   0.91x |  6.3% | 172 |    1.49 |      9.23 |  PASS |  no |    tok 102 |
| numeric    |                                         Baseline (7B only) |    18.78 |    256 |  13.6 |   1.00x |    -- | 256 |    1.00 |      8.62 |  PASS |  no |         -- |
| numeric    |              Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.01 |    256 |  36.5 |   2.68x | 88.2% |  32 |    8.00 |      9.85 |  PASS |  no |  identical |
| numeric    | Latent Mitosis Tree (EAGLE-2, leaves<=16, K=8, split<0.85) |    18.17 |    256 |  14.1 |   1.03x |  8.6% | 154 |    1.66 |      9.23 |  PASS |  no |     tok 12 |
| oop        |                                         Baseline (7B only) |    18.74 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.62 |  PASS |  no |         -- |
| oop        |              Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.30 |    256 |  35.1 |   2.57x | 83.5% |  34 |    7.53 |      9.83 |  PASS |  no |     tok 52 |
| oop        | Latent Mitosis Tree (EAGLE-2, leaves<=16, K=8, split<0.85) |    19.97 |    256 |  12.8 |   0.94x |  6.8% | 168 |    1.52 |      9.23 |  PASS |  no |     tok 52 |
| errors     |                                         Baseline (7B only) |    18.63 |    256 |  13.7 |   1.00x |    -- | 256 |    1.00 |      8.62 |  PASS |  no |         -- |
| errors     |              Evolutionary Tree (leaves<=8, K=8, split<0.8) |     7.81 |    256 |  32.8 |   2.39x | 77.8% |  36 |    7.11 |      9.85 |  PASS |  no |     tok 66 |
| errors     | Latent Mitosis Tree (EAGLE-2, leaves<=16, K=8, split<0.85) |    20.37 |    256 |  12.6 |   0.91x |  6.5% | 171 |    1.50 |      9.23 |  PASS |  no |     tok 66 |

## Summary

- **Best configuration**: Evolutionary Tree (leaves<=8, K=8, split<0.8) at **2.40x** the baseline
- **Baseline throughput**: 13.6 tok/s
- **Mean acceptance rate**: 41.6%
- **Peak VRAM**: 9.86 GiB of 15.9 GiB
- **Generated code parses**: 29/30 runs (1 truncated by the token budget, not scored)
- **Output identical to baseline**: 6/20 greedy runs

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

- Divergence begins at token **66** on average (earliest 3, latest 158) of 256 generated.
- Both continuations remain valid Python; the Code column is scored independently of whether the wording matched.

The practical consequence: with a reduced-precision target, treat speculative decoding as distribution-preserving up to that precision, not as a bitwise-identical drop-in. The speedups above are unaffected -- they are throughput measurements, and both arms decode the same kind of text at the same budget.
