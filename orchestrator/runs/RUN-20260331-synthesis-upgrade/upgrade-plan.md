# Orchestrator Synthesis: Upgrade Plan v4

**Date:** 2026-03-31 | **Target:** Beat 1.1147 BPB (current #1)
**Our best:** 1.1228 BPB (2026-03-22) | **Gap:** 0.0081 BPB

---

## New #1 Analysis (2026-03-25, abaybektursun, 1.1147 BPB)

### What changed from #2 (1.1194) to #1 (1.1147): -0.0047 BPB

| Technique | Delta | Nature |
|-----------|:-----:|--------|
| Full Hessian GPTQ (Cholesky + col reorder) | -0.003 est | Replaces GPTQ-lite diagonal approx |
| AR self-gen calibration (64 seqs × 2048 @ temp=0.8) | enables GPTQ | Legal workaround for calibration data |
| XSA on ALL 11 layers (was last 4) | -0.001 est | Zero-param improvement |
| BigramHash 3072 × dim=112 (was 1536) | -0.001 est | More capacity within budget |
| Selective ±1 pruning by recon error | 0 | Size reduction, not BPB |
| LZMA preset=9 (was zstd-22) | 0 | Better compression, frees param space |
| warmdown=4000 (was 3500) | -0.0005 est | Slightly longer cooldown |
| Dropped TTT | +0.002 | TTT no longer helps on this stack |

### Key insight: Full Hessian GPTQ is the dominant improvement

The Cholesky-based GPTQ with column reordering is a strictly better quantizer than the diagonal approximation. It compensates quantization error across columns, not just per-row. This is the single biggest delta.

---

## Gap Analysis: Our 1.1228 vs New #1's 1.1147

| Feature | Our Run (1.1228) | New #1 (1.1147) | Gap |
|---------|:----------------:|:----------------:|:---:|
| GPTQ | GPTQ-lite (diagonal) | Full Hessian (Cholesky) | **YES** |
| Calibration | Percentile search | AR self-gen 64×2048 | **YES** |
| XSA | Last 4 layers | All 11 layers | **YES** |
| BigramHash | 1536 | 3072 × 112 | **YES** |
| Activation | ReLU² | LeakyReLU(0.5)² | **YES** (our improvement) |
| MTP | Disabled (NUM_HEADS=0) | Disabled | tie |
| EMA | Fixed 0.997 | Fixed 0.997 | tie |
| TTT | Not used | Not used (confirmed negative) | tie |
| Compression | zstd-22 | LZMA preset=9 | **YES** |
| Pruning | None | Selective ±1 | **YES** |
| warmdown | 3500 | 4000 | small |

---

## Three Upgrade Paths (prioritized)

### Path A: Incremental — Port #1's techniques to our codebase
**Expected: 1.1228 → ~1.116 BPB** (close to or matching #1)

| Priority | Technique | Expected Δ | Effort |
|:--------:|-----------|:----------:|:------:|
| P0 | Full Hessian GPTQ + AR self-gen calibration | -0.003 | Port ~200 LOC |
| P0 | XSA on all 11 layers (`XSA_LAST_N=11`) | -0.001 | 1 env var |
| P0 | BigramHash 3072 × 112 | -0.001 | 1 env var |
| P1 | LZMA preset=9 (replace zstd-22) | size only | ~10 LOC |
| P1 | Selective ±1 pruning | size only | ~40 LOC |
| P1 | warmdown=4000 | -0.0005 | 1 env var |
| **Total** | | **-0.005 to -0.006** | |

### Path B: Novel — Stack our unique improvements on top of #1
**Expected: 1.1147 → ~1.110 BPB** (new SOTA)

| Priority | Technique | Expected Δ | Evidence |
|:--------:|-----------|:----------:|----------|
| P0 | LeakyReLU(0.5)² (already confirmed) | -0.002 | CAR test, #1 ablation |
| P1 | MTP_NUM_HEADS=1 | -0.001 to -0.003 | Infrastructure exists, never tested |
| P1 | EMA scheduling 0.99→0.9999 | -0.001 to -0.002 | "EMA Without the Lag" |
| P2 | QAT threshold sweep (0.10, 0.15, 0.20) | -0.001 | Never swept |
| P2 | Lloyd-Max centroids for weight quant | -0.001 | TurboQuant crossover idea |
| **Total** | | **-0.005 to -0.009** | |

### Path C: Moonshot — TurboQuant rotation for weight quantization
**Expected: unknown, high risk**

TurboQuant's core insight: random orthogonal rotation makes weights near-uniform, making quantization much more efficient. Currently applied to KV cache only.

**Hypothesis H39: Apply rotation BEFORE int6 weight quantization**

```python
# Before quantization, rotate weight matrix
R = random_orthogonal_matrix(cols, seed=42)
W_rotated = W @ R
# Quantize (now uniformly distributed = optimal for Lloyd-Max)
Q, S = quantize_int6(W_rotated)
# At inference: dequantize and inverse-rotate
W_hat = dequant(Q, S) @ R.T
```

Why this might work:
- After rotation, outlier columns are spread across all columns
- int6 uniform grid becomes near-optimal (no wasted codewords)
- Cholesky GPTQ may be **redundant** if rotation works (simpler pipeline)

Why it might fail:
- Rotation matrix R must be stored or regenerated (seed-based = free)
- Extra matmul at inference = slower eval
- Weight matrices may already be well-conditioned after STE QAT

**Risk: Medium | Potential: -0.002 to -0.005 BPB if outlier distribution matters**

---

## Recommended Execution Order

```
PHASE 1 (Day 1): Port Path A to our codebase
  ├── Copy Full Hessian GPTQ + AR self-gen from new #1
  ├── Set XSA_LAST_N=11, BIGRAM_VOCAB_SIZE=3072, BIGRAM_DIM=112
  ├── Switch to LZMA preset=9
  ├── Add selective ±1 pruning
  └── Expected: ~1.116 BPB (match #1)

PHASE 2 (Day 1-2): Stack Path B novel improvements
  ├── Apply LeakyReLU(0.5)² (already in our RUNPOD_READY)
  ├── Enable MTP_NUM_HEADS=1
  ├── Enable EMA scheduling
  └── Expected: ~1.110 BPB (new SOTA)

PHASE 3 (Day 3): Moonshot Path C if time permits
  ├── Implement rotation-before-quantization
  ├── Compare vs Full Hessian GPTQ
  └── Expected: unknown, but could stack on top of Phase 2
```

---

## TurboMOQ Relevance Assessment (from 2026-03-30 analysis)

**TurboMOQ is an INFERENCE optimization (KV cache compression), NOT a training/weight compression technique.**

| TurboMOQ Idea | Parameter Golf Relevance | Action |
|---|---|---|
| Random rotation → uniform distribution | **HIGH** — apply to weights before int6 quant (Path C) | H39 |
| Lloyd-Max centroids | **MEDIUM** — replace uniform int6 grid | H40 |
| QJL residual sign bits | **LOW** — adds bits, counterproductive for 16MB | Skip |
| Head importance scoring | **LOW** — we don't compress KV cache | Skip |
| Progressive compression | **NONE** — eval context is only 2048 | Skip |

**Bottom line**: TurboQuant's rotation trick is the only directly applicable technique, and it's worth testing as H39 (Path C moonshot).

---

## Updated Hypothesis Bank

| ID | Hypothesis | Priority | Status |
|----|-----------|:--------:|--------|
| H39 | Rotation before int6 weight quantization (TurboQuant crossover) | P2 | NEW |
| H40 | Lloyd-Max centroids for weight quantization | P2 | NEW |
| H41 | Full Hessian GPTQ + AR self-gen calibration (port from new #1) | P0 | NEW — CRITICAL |
| H42 | XSA on all 11 layers (port from new #1) | P0 | NEW — CRITICAL |
| H43 | BigramHash 3072 × 112 (port from new #1) | P0 | NEW |
| H44 | Selective ±1 pruning (port from new #1) | P1 | NEW |
| H45 | LZMA preset=9 compression (port from new #1) | P1 | NEW |
