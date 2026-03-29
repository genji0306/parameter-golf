# Swarm Debate Analysis: Corrected Gap Assessment

**Date:** 2026-03-29
**Input:** DarkLab 6-agent swarm debate (2026-03-29_parameter-golf-swarm-debate.md)
**Corrected against:** Our actual best run (1.1228 BPB, 2026-03-22)

---

## Critical Correction: The Swarm Overestimated the Gap

The swarm report identified 17 techniques. Cross-referencing against our actual
best run's README reveals **10 of 17 are already implemented**:

| Swarm Suggestion | Status | Evidence |
|---|---|---|
| Sliding window stride=64 | HAVE | README line 3, logs show stride:64 |
| SmearGate + BigramHash 2048 | HAVE | README line 44 |
| Value Embedding 128 | HAVE | README line 43 |
| U-Net skip connections | HAVE | README line 39 |
| GPTQ-lite clip percentile (5 candidates) | HAVE | README line 62 |
| Weight decay 0.04 | HAVE | README line 50 |
| Batch 786K tokens/step | HAVE | README line 53 |
| OrthoInit | HAVE | README line 58 |
| Partial RoPE + LN Scale | HAVE | README lines 41-42 |
| XSA last 4 layers | HAVE | README line 40 |
| **LeakyReLU(0.5)²** | **MISSING** | We use relu² (README line 38) |
| **TTT (score-first SGD)** | **MISSING** | Not in our codebase |
| **N-gram eval cache** | **NEW** | Not in any entry's training, eval-time only |
| **WSD schedule (D2Z)** | **NEW** | We use warmdown (different paradigm) |
| **NorMuon optimizer** | **NEW** | We use standard Muon |
| **EMA decay scheduling** | **NEW** | We use fixed 0.997 |
| **SVD-aware quantization** | **NEW** | We use uniform int6 |

**The actual remaining gap (0.0034 BPB) is explained by just 2 techniques:**
1. LeakyReLU(0.5)² → ~0.002 BPB
2. TTT (score-first SGD) → ~0.002 BPB

---

## Genuinely New Hypotheses (not in Loop 2 bank)

### H39: N-Gram Eval Cache with Multi-Order Backoff
- **Expected:** -0.05 to -0.12 BPB (leaderboard evidence: 1.024 BPB achieved)
- **Risk:** Low (eval-time only, zero training/artifact impact)
- **Classification:** GAME-CHANGER if legal in submission track
- **Mac Mini test:** Repetition analysis confirms 62% bigram, 27% trigram hit rates
  on real validation data. Simulation limited by lack of actual model logits.
- **Needs:** Legality check — does the competition allow eval-time n-gram caching?

### H40: LeakyReLU(0.5)² Activation
- **Expected:** -0.002 BPB (measured from #1's ablation)
- **Risk:** Very low — 1 line change
- **Classification:** IMMEDIATE
- **Note:** This is the single largest proven missing component

### H41: WSD Schedule with Linear Decay-to-Zero
- **Expected:** -0.005 to -0.010 BPB (arXiv:2502.15938)
- **Risk:** Low-Medium — different from our warmdown paradigm
- **Classification:** Needs RunPod validation
- **Conflict:** May interact with warmdown and QAT threshold timing

### H42: NorMuon Optimizer
- **Expected:** -0.003 to -0.006 BPB (arXiv:2510.05491)
- **Risk:** Low-Medium — drop-in Muon replacement
- **Classification:** Needs RunPod validation

### H43: EMA Decay Scheduling (0.99 → 0.9999)
- **Expected:** -0.002 to -0.004 BPB
- **Risk:** Low — extends our existing H15 hypothesis
- **Note:** H15 in the bank covers this; H43 adds the specific paper reference

---

## Confirmed Negative Results (from swarm)

These should be added to lessons-learned:

| Technique | Result | Source |
|-----------|--------|--------|
| MoE (2 experts) | -0.06 to -0.08 BPB WORSE | PR #831 |
| Differential Attention | Incompatible with int6 | PR #831 |
| SSM Hybrid (GatedDeltaNet) | +240% overhead | PR #831 |
| nGPT Hypersphere | +0.35 BPB from quant | PR #831 |
| Hourglass FFN | +0.33 BPB | PR #831 |
| Depth recurrence >2x uncorrected | 900x error amplification | Issue #140 |
| TrigramHash | +18% overhead, only 1.1298 | Issue #140 |
| Int4 quantization | +0.065 BPB catastrophic | Issue #140 |
| FSDP for small models | 3x slower than DDP | Benchmark |

These validate our existing bank:
- H01 (TrigramHash) → confirmed negative, should retire
- H36 (Depth Recurrence) → needs per-pass scalars, risk upgraded
- H16 (MoE) → confirmed negative, should retire

---

## Revised Execution Priority

### Phase 1: Close the Gap (~$5 RunPod)

| Priority | Hypothesis | Expected | Effort | Type |
|:--------:|:-----------|:--------:|--------|------|
| **1** | **H40: LeakyReLU(0.5)²** | **-0.002** | **1 line** | Activation |
| 2 | H05: MTP_NUM_HEADS=1 | -0.001 to -0.003 | env var | Training |
| 3 | H26: QAT threshold sweep | -0.001 to -0.003 | env var | Quantization |
| 4 | H15/H43: EMA scheduling | -0.002 to -0.004 | 5 lines | Training |
| 5 | H29: SWA+EMA blend | -0.001 to -0.002 | post-train | Post-training |

**Conservative projection: 1.1228 - 0.007 = ~1.116 BPB (new #1)**

### Phase 2: Push Further (~$10 RunPod)

| Priority | Hypothesis | Expected | Effort | Type |
|:--------:|:-----------|:--------:|--------|------|
| 6 | H42: NorMuon | -0.003 to -0.006 | drop-in | Optimizer |
| 7 | H04: Sequence curriculum | -0.002 to -0.005 | 20 lines | Training |
| 8 | H08: TTT with Adam | -0.001 to -0.003 | 10 lines | Eval-time |
| 9 | H41: WSD schedule | -0.005 to -0.010 | 5 lines | Training |

**Aggressive projection: ~1.105 BPB**

### Phase 3: Nuclear Option (if legal)

| Priority | Hypothesis | Expected | Effort | Type |
|:--------:|:-----------|:--------:|--------|------|
| 10 | **H39: N-gram eval cache** | **-0.05 to -0.12** | **200 lines** | **Eval-time** |

**Nuclear projection: ~1.02 BPB**

---

## What to Retire from Hypothesis Bank

| ID | Title | Reason |
|----|-------|--------|
| H01 | TrigramHash | Confirmed negative: +18% overhead, only 1.1298 |
| H16 | MoE (if exists) | Confirmed negative: -0.06 to -0.08 WORSE |
| H36 | Combo A (Recurrence+MoD+KV) | Risk upgraded: 900x quant error without correction |
| H35 | Full Depth Recurrence | Same: needs per-pass scalars to be viable |

---

## N-Gram Cache: Key Question

**Is n-gram caching legal in the 10min/16MB track?**

Arguments for legality:
- Uses only backward-looking tokens (already scored)
- Zero training parameters, zero artifact size impact
- Count-min sketch built during eval, discarded after

Arguments against:
- May be considered "test-time compute" beyond model inference
- Leaderboard may have separate track for neural+cache entries

**Action:** Check competition rules and GitHub issues before implementing.
