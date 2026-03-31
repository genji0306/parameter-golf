# MOPT-Golf Simulation Results

**Date:** 2026-04-01 | **Runtime:** 16.5s local + 65.7s Mac Mini M4
**Architecture:** 11 layers, 512d, 1536 MLP, 8 heads (matches leaderboard #1)

## UPDATE: Mac Mini M4 Reality Check (2026-04-01)

**Both key ideas from MOPT-Golf FAIL on real testing:**

1. **GPTQ ordering: 0% difference** — Per-row quantization is independent per layer. Ordering only matters with full Cholesky GPTQ + activation recalibration (GPU-only).

2. **Bell-curve 678: BUSTS 16MB budget** — Uniform Int6 = 15.40 MB (fits). Bell 678 = 16.99 MB (1MB over). Higher-entropy Int7/Int8 compress worse with LZMA.

**Verdict: MOPT-Golf contributes ZERO actionable improvements to Parameter Golf.** The document's theoretical analysis did not survive empirical testing. All three techniques (bell-curve allocation, middle-out ordering, bias correction) are either ineffective or infeasible within the 16MB constraint.

---

## Critical Assessment of MOPT-Golf Document

The MOPT-Golf paper has several factual errors:
- Assumes 5-layer recurrent architecture (actual top solutions: 11 unique layers, no recurrence)
- Claims 10.4MB headroom (actual: artifact is ~15.9MB, only ~0.1MB free)
- Predicts 0.885 BPB (fantasy — linear stacking doesn't work)
- N-gram cache claim (-0.10 BPB) has zero evidence in competition

**Only 3 ideas survive reality-checking.** Results below.

---

## Test Results

### 1. Bell-Curve Bit Allocation

| Strategy | Allocation | Avg MSE | Size (MB) | MSE vs Baseline |
|----------|-----------|:-------:|:---------:|:---------------:|
| **uniform_int6** (baseline) | [6,6,6,6,6,6,6,6,6,6,6] | 4.62e-06 | 20.625 | -- |
| **bell_curve_678** | [6,6,6,7,7,**8**,7,7,6,6,6] | 2.71e-06 | 22.500 | **-41.2%** |
| bell_curve_67 | [6,6,6,7,7,7,7,7,6,6,6] | 2.81e-06 | 22.188 | -39.2% |
| bell_curve_68 | [6,6,6,6,6,**8**,6,6,6,6,6] | 4.14e-06 | 21.250 | -10.3% |

**Verdict: Bell-curve 678 gives 41% MSE reduction for +1.875 MB.**
BUT: the actual artifact budget is ~15.9MB with LZMA compression. The uncompressed weight size is already larger — LZMA handles the extra bits. Need to test whether LZMA can absorb the extra 1.875MB.

**Estimated BPB impact: -0.0012 BPB** (very rough). This is marginal.

**PROBLEM:** The current #1 already uses Full Hessian GPTQ which partially solves the same problem (column reordering puts more precision where error is highest). Bell-curve allocation may be redundant with good GPTQ.

### 2. Middle-Out vs Sequential GPTQ Ordering

| Order | Avg MSE | Max MSE | vs Sequential |
|-------|:-------:|:-------:|:-------------:|
| Sequential (0->10) | 3.76e-03 | 1.10e-02 | baseline |
| **Middle-out** (5->4,6->...) | 2.24e-03 | 1.06e-02 | **-40.5%** |
| **Reverse (10->0)** | 3.27e-04 | 9.29e-04 | **-91.3%** |
| Sensitivity-first | 2.24e-03 | 1.06e-02 | -40.5% |

**Surprise: Reverse order (last layer first) is BY FAR the best (-91.3%).**

Why? In a forward-pass model, layer 10's quantization doesn't affect layers 0-9's calibration activations. By quantizing from back to front, each layer's calibration data is computed through entirely unquantized preceding layers. This is the optimal ordering for unidirectional models.

Middle-out helps (-40.5%) but is suboptimal because it still quantizes some early layers before late layers are calibrated.

**This is actually the most actionable finding.** The current #1 appears to use sequential ordering (default). Switching to reverse order could improve GPTQ quality significantly.

### 3. Learned Bias Correction

| Metric | Value |
|--------|:-----:|
| Without bias | MSE = 8.39e-02 |
| With bias | MSE = 7.78e-02 |
| Improvement | **7.28%** |
| Cost | 22 KB (5,632 float32 params) |

**Verdict: 7% MSE reduction for 22KB is decent but the absolute impact on BPB is tiny (~0.0002 BPB estimated).** Not worth the complexity.

### 4. Combined

| Config | Avg MSE | Size | BPB est |
|--------|:-------:|:----:|:-------:|
| Baseline (uniform Int6) | 4.62e-06 | 20.6 MB | -- |
| MOPT-Golf combined | 2.71e-06 | 22.5 MB | **-0.0012** |

---

## Actionable Recommendations for Next RunPod Run

### ADOPT (high confidence)

| Technique | Expected BPB | Evidence | Priority |
|-----------|:------------:|----------|:--------:|
| **Reverse GPTQ ordering** (layer 10 first) | -0.001 to -0.002 | 91% MSE reduction in simulation | **P0** |

The reverse ordering is nearly free to implement — just change the loop order in the GPTQ calibration function. No size cost.

### MAYBE (test on RunPod)

| Technique | Expected BPB | Risk |
|-----------|:------------:|:----:|
| Bell-curve 678 allocation | -0.001 | May not fit in 16MB after LZMA |
| Bell-curve 67 (conservative) | -0.0008 | Lower size risk |

### SKIP

| Technique | Why |
|-----------|-----|
| Bias correction | 22KB for ~0.0002 BPB, not worth complexity |
| Recurrence-aware calibration | Our model has no recurrence |
| N-gram eval cache | No evidence, likely score manipulation |
| Scale to 576d | Blows param budget |

---

## Updated Priority Stack (incorporating all findings)

| # | Technique | Expected BPB | Status |
|:-:|-----------|:------------:|--------|
| 1 | Full Hessian GPTQ (port from #1) | -0.003 | Ready to port |
| 2 | **Reverse GPTQ ordering** (NEW from this test) | -0.001 to -0.002 | **Novel finding** |
| 3 | XSA all 11 layers | -0.001 | 1 env var |
| 4 | BigramHash 3072 x 112 | -0.001 | 1 env var |
| 5 | LeakyReLU(0.5)^2 | -0.002 | Already in our code |
| 6 | MTP_NUM_HEADS=1 | -0.001 to -0.003 | Env var |
| 7 | EMA scheduling | -0.001 | Already in our code |
| 8 | Bell-curve bit allocation | -0.001 | Size risk |
| | **Projected total** | **-0.010 to -0.014** | |
| | **Projected BPB** | **~1.101 to 1.105** | |
