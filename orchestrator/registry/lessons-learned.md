# Lessons Learned

Write durable cross-run lessons here.

Examples:

- which stacks repeatedly fail the wallclock budget
- which evaluation tricks are robust
- which Memento-style skills have high utility
- which prompts or reflection rules produce low-quality outputs

## Confirmed Lessons

### L01: Rotation-before-quantization is ineffective at int8 per-row precision (EXP-H38-A)
**Date:** 2026-03-28 | **Hypothesis:** H38

TurboQuant's Hadamard rotation trick (spreading outliers via WHT before quantization)
provides negligible benefit (+0.03% MSE) when using int8 per-row quantization with
99.99984th percentile clipping. The per-row scale already adapts to each output
channel's range, making rotation redundant.

**Key insight:** Per-row quantization is already "per-channel" — each row gets its own
scale. Rotation helps most when a single scale must cover an entire vector (like KV
cache per-token quantization). This is why TurboQuant works for KV cache but not weights.

**Non-power-of-2 dims:** MLP weights (512x1536) cannot use WHT without padding.
Padding breaks the round-trip guarantee. Block-diagonal rotation or careful padding
bookkeeping would be needed.

**When to revisit:** If moving to int6 or lower per-row quantization, or if implementing
per-tensor (not per-row) quantization where rotation could equalize row-wise ranges.
Also potentially useful combined with H37 (per-row adaptive bitwidth) where some rows
get int5.
