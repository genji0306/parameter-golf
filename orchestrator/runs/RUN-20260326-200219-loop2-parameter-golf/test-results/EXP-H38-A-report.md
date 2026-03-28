# EXP-H38-A: TurboQuant Hadamard Rotation Before Weight Quantization

**Hypothesis:** H38 — Applying Walsh-Hadamard rotation before int8 per-row quantization
spreads outlier values uniformly, reducing clipping loss.

**Status:** NEGATIVE RESULT (on synthetic weights) / NEEDS REAL-WEIGHT VALIDATION

**Date:** 2026-03-28
**Platform:** Mac Mini M4 16GB (cyber02@192.168.23.25)
**Runtime:** numpy 2.0.2, arm64

---

## Executive Summary

On **synthetic weights** with realistic outlier structure (Gaussian + 3-8x outlier
columns/rows), the Hadamard rotation shows a **+0.03% weighted MSE improvement** —
essentially noise. The technique is mathematically sound (round-trip error < 2e-15)
but the current per-row int8 quantization with 99.99984th percentile clipping already
handles outliers well enough that rotation provides negligible benefit.

**Verdict:** The rotation trick delivers massive gains (60-90%) when outlier
concentration is extreme, but our model's weight distributions — even with injected
outliers — don't hit that threshold at int8 precision. The technique may become
relevant at int6 or lower bitwidths where clipping loss is much larger.

---

## Results

### Overall

| Metric | Baseline | Rotated | Delta |
|--------|----------|---------|-------|
| Weighted avg MSE | 5.170e-06 | 5.169e-06 | **+0.03%** |
| Tensors rotated | - | 32/68 | 47% chose rotation |
| Export time | 790ms | 5796ms | **+634% overhead** |

### Per Tensor Type

| Type | Avg Improvement | Rotated | Cosine (base -> rot) |
|------|----------------|---------|---------------------|
| Q proj | +0.1% | 7/11 | 0.996704 -> 0.996709 |
| K proj | +0.2% | 6/11 | 0.996878 -> 0.996884 |
| V proj | +0.1% | 5/11 | 0.996735 -> 0.996739 |
| O proj | +0.1% | 7/11 | 0.996690 -> 0.996694 |
| MLP up | +0.0% | 0/11 | 0.997302 -> 0.997302 |
| MLP down | +0.1% | 6/11 | 0.996328 -> 0.996331 |
| Embedding | +0.0% | 1/1 | 0.993600 -> 0.993602 |
| LM head | +0.0% | 0/1 | 0.993605 -> 0.993605 |

### Key Observations

1. **MLP up weights (512x1536):** Never rotated. Non-power-of-2 last dim (1536)
   makes WHT inapplicable without padding, and padding breaks the round-trip guarantee.

2. **Attention projections:** Rotation helps marginally (0.1-0.2%) on some layers.
   The improvement is real but too small to matter for BPB.

3. **Export time overhead:** +634% is unacceptable for a 0.03% gain. The MSE-adaptive
   approach requires quantizing twice (baseline + rotated) for each tensor.

4. **Why the huge gains from earlier tests don't appear:** Those tests used extreme
   outliers (10x magnitude in specific rows/columns). The realistic weight generator
   uses milder outliers (3-8x), which the per-row clipping already handles. Real
   trained weights may have different outlier patterns — this needs validation on
   actual checkpoints.

---

## Analysis: Why TurboQuant Works for KV Cache but Not Weights

TurboQuant/PolarQuant's rotation is designed for **inner product preservation** in
attention score computation. The KV cache has fundamentally different statistics:

| Property | KV Cache | Weights |
|----------|----------|---------|
| Distribution | Concentrated, high kurtosis | Near-Gaussian, moderate tails |
| Outlier pattern | Channel-wise (a few dims dominate) | Row-wise (some output channels larger) |
| Quantization | Per-vector (all dims share one scale) | Per-row (each row gets its own scale) |
| Impact of rotation | Massive (equalizes per-vector max) | Minimal (per-row scale already adapts) |

The per-row int8 scheme already gives each row its own scale factor, which is
equivalent to "per-channel" quantization. Rotation helps most when a single scale
must cover a whole vector — which is the KV cache case, not the weight case.

---

## Recommendations

### Abandon for Current Stack (int8 export)
At int8 precision with per-row clipping, the quantization gap is already tiny
(MSE ~5e-06, cosine >0.993). Rotation cannot improve what's already near-lossless.

### Revisit if Moving to int6 or Lower
At int6 (clip_range=31 instead of 127), quantization error increases ~16x. The
rotation technique may become relevant there, especially for:
- Tensors with outlier columns (attention projections)
- Combined with STE QAT (H26) where the model adapts to quantized weights

### Combine with H37 (Per-Row Adaptive Bitwidth)
If we implement mixed int5/int6/int7 per-row quantization, lower-bitwidth rows
would benefit more from rotation pre-conditioning. This is a future combo to test.

### Real Weight Validation Still Needed
Trained transformer weights have different outlier structure than synthetic weights.
Run on an actual checkpoint (re-export with `ROTATE_QUANT_ENABLED=1`) before
fully closing this hypothesis.

---

## Artifacts

- Test script: `orchestrator/scripts/test_turboquant_rotation.py`
- JSON results: `test-results/EXP-H38-A-results.json`
- Code changes: `parameter-golf/train_gpt.py` (lines 297-350, 376-408)
- Env var: `ROTATE_QUANT_ENABLED=1` (off by default)

## Lessons Learned

1. **Per-row quantization already handles outliers well.** The biggest quantization
   gains come from training-time techniques (QAT, spectral reparam) not export-time
   rotation.

2. **Non-power-of-2 dimensions break WHT round-trip.** MLP weights (512x1536) cannot
   use Hadamard rotation without padding, which introduces errors. A real deployment
   would need block-diagonal rotation or padding-aware bookkeeping.

3. **MSE-adaptive approach is correct but doubles export cost.** If rotation were
   beneficial, a static heuristic (e.g., "always rotate attention, never rotate MLP")
   would eliminate the overhead.

4. **TurboQuant's innovation is KV-cache-specific.** The paper's contribution is
   combining PolarQuant + QJL for inner-product-preserving KV compression, not
   general weight quantization. Applying only the rotation component misses the
   synergy that makes TurboQuant work.
