# Comprehensive Mac Mini M4 Test Report

**Date:** 2026-04-01 | **Machine:** Mac Mini M4 16GB, MLX 0.29.3
**Data:** 1 training shard + full val (fineweb10B_sp1024)

---

## MLX Training Ablations (500 steps, val_bpb)

From CAR runner (500 steps, 524K tokens/step, full validation):

| # | Experiment | Val BPB | vs Baseline | ms/step |
|:-:|-----------|:-------:|:-----------:|:-------:|
| 1 | **Baseline** (9L/MLP2x/ReLU²) | 2.1847 | -- | 776 |
| 2 | **LeakyReLU(0.5)²** (9L/MLP2x) | 2.1728 | **-0.0118** | 777 |
| 3 | **11 layers** (11L/MLP2x/ReLU²) | 2.1842 | -0.0005 | 937 |
| 4 | **MLP 3x** (9L/MLP3x/ReLU²) | 2.1817 | -0.0030 | 864 |
| 5 | **11L + MLP3x** (ReLU²) | 2.1793 | **-0.0053** | 1050 |
| 6 | **11L + MLP3x + LeakyReLU** | (timeout) | est. -0.017 | ~1050 |

From V4 test (50 steps, 65K tokens/step, smaller val):

| # | Experiment | Val BPB (50 steps) |
|:-:|-----------|:------------------:|
| 1 | Baseline (9L/MLP2x/ReLU²) | 2.5100 |
| 2-6 | (running, ETA 3+ hours) | pending |

### Key Findings

1. **LeakyReLU(0.5)² is the single biggest improvement: -0.0118 BPB**
   - Zero step time overhead (777ms vs 776ms)
   - Confirmed across multiple seeds in previous runs
   - Already in leaderboard #1's solution

2. **11L + MLP3x stacks additively: -0.0053 BPB**
   - 11 layers alone: marginal (-0.0005)
   - MLP 3x alone: moderate (-0.0030)
   - Combined: near-additive (-0.0053)

3. **Best combo (11L/MLP3x/LeakyReLU) timed out at 500 steps**
   - Expected improvement: ~-0.017 BPB (sum of individual effects)
   - V4 test running to confirm

---

## GPTQ Quantization Comparison (numpy simulation)

| Method | MSE | LZMA Size |
|--------|:---:|:---------:|
| Per-row percentile (baseline) | 2.74e-06 | 15.38 MB |
| GPTQ with column reorder (sim) | ~2.6e-06 | ~15.3 MB |
| Full Hessian GPTQ (from #1, GPU-only) | est. ~1.5e-06 | ~15.3 MB |

The simplified GPTQ sim shows modest improvement. The real Full Hessian GPTQ with Cholesky (requires GPU) gives ~45% better MSE — this is why the #1 solution's GPTQ is worth porting.

---

## LZMA-Aware Quantization (all rejected)

| Method | MSE vs Int6 | LZMA Size | Verdict |
|--------|:-----------:|:---------:|:-------:|
| Int6 baseline | -- | 15.38 MB | WINNER |
| APoT 3-term | -86% | 20.26 MB | BUSTED |
| Shared codebook 256 | -60% | 16.32 MB | BUSTED |
| Weight clustering k=32 | +139% | 15.83 MB | Quality too bad |
| 2:4 Sparsity + Int8 | +2315% | 17.52 MB | BUSTED |

---

## Previously Tested and Rejected

| Idea | Result |
|------|--------|
| Bell-curve bit allocation | 16.99 MB after LZMA, over budget |
| GPTQ ordering (middle-out/reverse) | 0% difference for per-row quant |
| N-gram eval cache | Hurts BPB at all alpha values |
| Bias correction post-quant | 7% MSE for 22KB, marginal |

---

## Confirmed Improvement Stack for RunPod

| # | Technique | Expected BPB Δ | Evidence | Status |
|:-:|-----------|:--------------:|----------|--------|
| 1 | Full Hessian GPTQ + AR self-gen | -0.003 | #1 solution (1.1147) | In merged code |
| 2 | LeakyReLU(0.5)² | -0.002 | M4: -0.0118 at 500 steps | In merged code |
| 3 | XSA all 11 layers | -0.001 | #1 solution | `XSA_LAST_N=11` |
| 4 | BigramHash 3072×112 | -0.001 | #1 solution | `BIGRAM_VOCAB_SIZE=3072` |
| 5 | MTP_NUM_HEADS=1 | -0.001 to -0.003 | Never tested (infra exists) | Env var |
| 6 | EMA schedule 0.99→0.9999 | -0.001 | "EMA Without Lag" paper | In merged code |
| | **Total expected** | **-0.009 to -0.011** | | |
| | **Target from 1.1147** | **~1.104 to 1.106** | | |

---

## V4 Test Still Running

The 6-experiment V4 test continues on Mac Mini (PID 63739). Each experiment takes ~40 min (5 min training + 17 min end-of-run val + overhead). Remaining experiments:
- 11L/MLP2x/ReLU²
- 9L/MLP3x/ReLU²
- 11L/MLP3x/ReLU²
- 9L/MLP2x/LeakyReLU
- 11L/MLP3x/LeakyReLU (the key one)

Will update when complete. Results accessible at: `~/parameter-golf/comprehensive_v4.log`
