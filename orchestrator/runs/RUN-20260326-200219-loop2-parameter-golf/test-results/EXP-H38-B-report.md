# EXP-H38-B: TurboQuant Rotation at int6/int5 Export Precision

**Hypothesis:** H38 — revisited at lower bitwidths per EXP-H38-A recommendation
**Status:** MIXED — one actionable finding for QAT+int5 downquant

**Date:** 2026-03-28
**Platform:** Mac Mini M4 16GB (cyber02@192.168.23.25)

---

## Executive Summary

Tested Hadamard rotation across 5 bitwidths (int8→int4) and 2 weight types
(raw Gaussian with outliers, QAT-shaped int6). The key finding:

**Rotation provides +17.3% MSE improvement on QAT-shaped weights downquantized to int5.**

This is the one scenario where rotation meaningfully helps: weights trained with
int6 STE QAT are already snapped near int6 grid points. When you downquantize them
to int5 (half the grid points), the "between-grid" error is systematic and rotation
effectively randomizes it, reducing the correlated quantization noise.

For all other scenarios (raw weights at any bitwidth, QAT weights at their native
int6 or higher), rotation provides ≤0.22% improvement — not worth the overhead.

---

## Results Matrix

| Bitwidth | Gaussian+Outliers | QAT-Shaped (int6) |
|----------|------------------:|-------------------:|
| int8     |            +0.22% |             +0.00% |
| int7     |            +0.21% |             +0.00% |
| int6     |            +0.17% |             +0.00% |
| **int5** |          **+0.00%** | **+17.33%** |
| int4     |            +0.00% |             +0.00% |

### Key Observations

1. **QAT-shaped weights at int6 → near-zero error.** MSE = 1.08e-08, which is
   ~480x lower than raw Gaussian at int6 (8.70e-05). The QAT training perfectly
   conditions weights for their target bitwidth.

2. **QAT-shaped weights at int5 → rotation helps massively.** The int6 QAT grid
   has 63 levels; int5 has 31. The "half-step" errors are systematic (always in
   the same direction for adjacent grid points). Rotation randomizes these errors
   across dimensions, reducing correlated noise. 57/68 tensors chose rotation.

3. **QAT-shaped at int8/int7 → no help.** Higher bitwidths capture int6-grid
   weights perfectly. Nothing to improve.

4. **Raw Gaussian at int5/int4 → no help (0%).** This is surprising. For raw
   weights at very low bitwidth, the quantization error is dominated by large
   outliers that rotation can't fix with per-row scaling. The MSE-adaptive
   approach correctly falls back to baseline.

5. **Raw Gaussian at int8/int7/int6 → tiny help (~0.2%).** Consistent but
   negligible benefit from spreading outlier columns.

---

## Practical Implications for Parameter Golf

### The int5 Downquant Path (NEW)

If we train with int6 STE QAT (as the top solution does) and then export at int5
instead of int8, we save 3 bits/param:
- 77M params × 3 bits = **29MB savings** (before compression)
- With rotation, the int5 export MSE drops by 17.3%

**BUT**: all raw artifact sizes are well over 16MB (int5 = ~49MB for 77M params).
The actual artifact fits under 16MB only after zlib/lzma compression. The savings
from int5 vs int8 would be in compressed size, and compression already reduces
int8 quantized weights efficiently (many values cluster in the int6 range).

### Verdict: Rotation Alone is Not Enough

The +17.3% MSE improvement at int5 is real but does not translate to a practical
BPB improvement because:
1. The artifact must still fit in 16MB after compression
2. int5 export with rotation still has 40x higher MSE than int8 export
3. The STE QAT already optimizes for int6; downquanting to int5 loses that benefit

### What Would Actually Help

1. **Train with int5 STE QAT from the start** + rotation at export → would get
   the QAT benefit AND the rotation benefit at int5
2. **Mixed bitwidth per-row** (H37/H07): some rows at int5, others at int7,
   optimized for the 16MB budget — rotation helps the int5 rows
3. **Per-layer adaptive** (TurboQuant's layer-adaptive idea): int5 for early
   layers, int6 for late layers, with rotation on the int5 layers

---

## Updated Lesson

### L01 (updated): Rotation is precision-boundary-specific

Rotation helps **only when downquantizing across a precision boundary** — i.e.,
weights trained for one grid (int6) being exported at a coarser grid (int5).
It does NOT help when:
- Exporting at the same or higher precision than training (int6 → int8)
- Quantizing raw weights at any precision (per-row scaling already adapts)

The one actionable finding: if implementing mixed-precision per-row export (H37),
apply rotation specifically to rows that get assigned a lower bitwidth than
their QAT training precision.

---

## Artifacts

- Test script: `orchestrator/scripts/test_turboquant_rotation_multibit.py`
- JSON results: `test-results/EXP-H38-B-results.json`
- Previous: `test-results/EXP-H38-A-report.md` (int8-only benchmark)
