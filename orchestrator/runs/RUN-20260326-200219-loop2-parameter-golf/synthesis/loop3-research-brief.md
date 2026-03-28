# Loop 3 Research Brief: Post-TurboQuant Synthesis

**Source:** EXP-H38-A/B results + Loop 2 debate
**Date:** 2026-03-29
**For:** AL (Solution Analyst), RL (Research Lead), M (Memento Designer)

---

## What Was Tested

**H38: TurboQuant-Inspired Hadamard Rotation Before Weight Quantization**

Applied Walsh-Hadamard rotation (random signs + butterfly WHT) to weight matrices
before per-row quantization at export. Tested across 5 bitwidths (int8→int4) and
2 weight distributions (raw Gaussian with outliers, int6-QAT-shaped).

Ran on Mac Mini M4 with 68 synthetic tensors (77M params) matching Stack A
architecture (11L/512d/MLP 1536).

---

## Results Summary

### What Failed

| Scenario | Improvement | Verdict |
|----------|:-----------:|---------|
| Raw weights at int8 | +0.22% | Negligible — per-row clipping already handles outliers |
| Raw weights at int7 | +0.21% | Same |
| Raw weights at int6 | +0.17% | Same |
| Raw weights at int5 | +0.00% | No help — outliers dominate at low bitwidth |
| Raw weights at int4 | +0.00% | Same |
| QAT-shaped at int8 | +0.00% | QAT weights are already perfectly int6-shaped |
| QAT-shaped at int7 | +0.00% | Same |
| QAT-shaped at int6 | +0.00% | MSE = 1.08e-08, near-perfect — nothing to improve |

### What Worked

| Scenario | Improvement | Verdict |
|----------|:-----------:|---------|
| **QAT-shaped (int6) → int5 export** | **+17.3%** | **57/68 tensors chose rotation** |

### Why

Rotation helps **only at precision boundaries** — when weights trained on one
quantization grid (int6 = 63 levels) are exported at a coarser grid (int5 = 31
levels). The "between-grid" errors are correlated; rotation randomizes them.

Per-row quantization already gives each row its own scale factor, making rotation
redundant when the bitwidth matches or exceeds the training precision.

---

## Confirmed Lesson (L01)

> Rotation-before-quantization is a **precision-boundary-specific** technique.
> It helps only when downquantizing across grid boundaries. It is NOT a general
> quantization improvement.

**Non-power-of-2 limitation:** MLP weights (512x1536) cannot use WHT. Only 512-dim
tensors (attention projections, down projections, embeddings) are eligible.

---

## Updated Hypothesis Rankings

H38 downgraded from rank 5 → rank 11 (contingent on H37). Current top 10:

| Rank | ID | Title | Priority | Risk | Status |
|-----:|:---|:------|:--------:|:----:|--------|
| 1 | H05 | MTP Auxiliary Loss | 0.95 | low | **READY** — zero code changes |
| 2 | H04 | Sequence Curriculum | 0.92 | low | **READY** — ~20 line change |
| 3 | H36 | Combo A (Recurrence+MoD+KV) | 0.88 | high | Needs architecture rewrite |
| 4 | H26 | Late QAT Threshold Tuning | 0.87 | low | **READY** — env var sweep |
| 5 | H32 | Mixture of Depth (MoD) | 0.84 | medium | Prerequisite for H36 |
| 6 | H08 | TTT Optimizer Upgrade | 0.83 | low | ~10 line change |
| 7 | H37 | Combo B (Spectral+QAT+Bitwidth) | 0.80 | medium | H38 rotation is sub-technique |
| 8 | H15 | EMA Decay Schedule | 0.78 | low | One-line change |
| 9 | H29 | SWA+EMA Ensemble Blending | 0.76 | low | Post-training, zero retrain |
| 10 | H35 | Full Depth Recurrence | 0.74 | high | Foundation for H36 |

---

## Implications for Next Hypotheses

### For RL (Research Lead)

1. **Quantization exploration is narrowing.** The quantization gap at int8 per-row
   is already tiny (~5e-06 MSE). Further quantization improvements should focus on:
   - **Training-time** techniques (QAT threshold, spectral reparam) not export-time
   - **Mixed bitwidth** (H37) where rotation becomes a valid sub-technique for
     downquantized rows

2. **Architecture and training efficiency are higher-value targets.** The remaining
   BPB gap (current: 1.1194, theoretical floor: ~1.08) is dominated by:
   - Model capacity (fixed by 16MB budget)
   - Training steps (fixed by 10-min budget)
   - H05/H04/H26 address these at near-zero risk

3. **New research directions to explore:**
   - **Learned codebook quantization:** Instead of uniform int6 grid, learn optimal
     centroid positions per-layer (inspired by TurboQuant's Lloyd's algorithm codebooks)
   - **KV cache quantization at inference:** TurboQuant's actual strength. Could enable
     longer sliding window eval (currently stride=64) by compressing KV cache, fitting
     more context in memory during TTT
   - **Rotation for activation quantization:** If implementing FP8 activations (like
     ternary stack), rotation may help there

### For M (Memento Designer)

1. **SK-QGR-004 (rotation) stays in catalog but marked contingent on H37.**
   Utility score remains 0.0. Only activate when H37 creates mixed-bitwidth rows.

2. **New skill opportunity: SK-QGR-005 (Lloyd's codebook quantization).**
   Replace uniform int6 grid with data-dependent centroids optimized via Lloyd's
   algorithm on actual weight distributions. TurboQuant's `codebook.py` provides
   reference implementation. Zero training change — pure export improvement.

3. **New skill opportunity: SK-INF-001 (KV cache compression for TTT).**
   Apply TurboQuant's actual KV cache compression during sliding window evaluation.
   3-bit KV cache = 4.6x compression → longer effective context window. This is
   TurboQuant's intended use case and may help TTT (H08) by enabling more context.

### For AL (Solution Analyst)

1. **Track whether any new leaderboard entries use rotation/Hadamard tricks.**
   QuIP# and similar rotation-based quantization methods may appear.

2. **The int6 QAT → int8 export path is confirmed near-optimal.** QAT-shaped
   weights at int8 have MSE = 3.3e-07, which is 260x lower than raw weights.
   The STE QAT is doing its job.

3. **Artifact size budget note:** Even at int5, raw model is ~49MB (before
   compression). The 16MB limit is met purely through compression (zlib/lzma),
   not lower bitwidth. Compression efficiency, not bitwidth, is the binding
   constraint.

---

## Execution Priority for Loop 3

**Phase 1 — Zero-cost wins (immediate, ~$5 RunPod):**
1. H05: `MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15`
2. H26: Sweep `LATE_QAT_THRESHOLD` ∈ {0.10, 0.20, 0.30, 0.50}
3. H15: EMA decay schedule (1-line)
4. H29: SWA+EMA blend sweep (post-training)

**Phase 2 — Training efficiency (~$10):**
5. H04: Sequence curriculum 512→1024→2048
6. H08: TTT optimizer Adam + per-layer LR

**Phase 3 — Architecture moonshots (~$30-80):**
7. H32: Mixture of Depth (prerequisite test for H36)
8. H37: Spectral reparam + progressive QAT + mixed bitwidth
   - Include H38 rotation as sub-technique for int5 rows

---

## Code Assets Available

| Asset | Path | Status |
|-------|------|--------|
| Rotation code (train_gpt.py) | `parameter-golf/train_gpt.py:297-350` | Merged, off by default (`ROTATE_QUANT_ENABLED=1`) |
| Int8 benchmark | `orchestrator/scripts/test_turboquant_rotation.py` | Complete |
| Multibit benchmark | `orchestrator/scripts/test_turboquant_rotation_multibit.py` | Complete |
| TurboQuant+ reference | `/tmp/turboquant_plus/` (local clone) | Available for codebook/KV cache reference |
| EXP-H38-A results | `test-results/EXP-H38-A-results.json` | Int8 benchmark data |
| EXP-H38-B results | `test-results/EXP-H38-B-results.json` | Multibit benchmark data |
