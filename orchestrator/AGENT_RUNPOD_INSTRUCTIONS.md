# Agent Instructions: RunPod Testing of Merged Solution

**For:** Codex, Antigravity, or any autonomous agent
**Goal:** Run the merged solution on RunPod 8xH100 and beat leaderboard #1 (1.1147 BPP)
**Solution:** `parameter-golf/records/track_10min_16mb/2026-04-01_MERGED_FullGPTQ_LeakyReLU_MTP_EMASchedule/`

---

## Context

You are testing our merged solution that combines:
- **Base:** Leaderboard #1 (abaybektursun, 1.1147 BPB, Full Hessian GPTQ + XSA-all + BigramHash 3072)
- **Our additions:** MTP enable + EMA decay scheduling (5 lines changed)

The solution is a single file `train_gpt.py` that runs on 8xH100 SXM with PyTorch 2.9.1+cu128.

---

## Step 0: Environment Setup

```bash
# RunPod 8xH100 SXM (80GB each)
# PyTorch 2.9.1+cu128 required

# Install Flash Attention 3 (Hopper kernels required)
pip install --break-system-packages flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece zstandard

# Verify
python3 -c "from flash_attn_interface import flash_attn_func; import sentencepiece, zstandard; print('deps OK')"

# Clone and enter repo
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Copy our solution
cp records/track_10min_16mb/2026-04-01_MERGED_FullGPTQ_LeakyReLU_MTP_EMASchedule/train_gpt.py .
```

---

## Step 1: Baseline Reproduction (MUST DO FIRST)

Reproduce the #1 solution to verify environment works. Default env vars = exact #1 behavior.

```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Expected:** val_bpb ~1.1147 (±0.0004). If not within range, DO NOT proceed — debug first.

**Checkpoints:**
- Step ~6920: training complete
- ~86ms/step expected
- Pre-quant BPB: ~1.135
- AR self-gen calibration: 64 seqs × 2048 tokens
- Final artifact: <16MB

---

## Step 2: Full Stack Run (MTP + EMA Schedule)

```bash
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
EMA_DECAY_START=0.99 EMA_DECAY_END=0.9999 \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Expected:** val_bpb ~1.104-1.113 (target: beat 1.1147)

**Record the result.** If val_bpb < 1.1147, this is a new SOTA. Run seeds 42 and 999 to get 3-seed mean.

---

## Step 3: Ablation Runs (if Step 2 improves)

Run each change independently to measure contribution:

### 3a: MTP only (no EMA schedule)
```bash
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### 3b: EMA schedule only (no MTP)
```bash
EMA_DECAY_START=0.99 EMA_DECAY_END=0.9999 \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Step 4: MTP Loss Weight Sweep (if MTP helps)

```bash
for W in 0.05 0.10 0.15 0.20 0.30; do
  echo "=== MTP_LOSS_WEIGHT=$W ==="
  MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=$W \
  EMA_DECAY_START=0.99 EMA_DECAY_END=0.9999 \
  BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
  TARGET_MB=15.9 SEED=314 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tail -5
done
```

---

## Step 5: Multi-Seed Validation (for best config)

Once you identify the best config from Steps 2-4:

```bash
for SEED in 314 42 999; do
  echo "=== SEED=$SEED ==="
  MTP_NUM_HEADS=<best> MTP_LOSS_WEIGHT=<best> \
  EMA_DECAY_START=<best> EMA_DECAY_END=<best> \
  BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
  TARGET_MB=15.9 SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

**Report:** 3-seed mean BPB, std, Welch t-test vs 1.1147.

---

## Decision Tree

```
Step 1: Baseline reproduces 1.1147?
  NO  → Debug environment. Check FA3, PyTorch version, data path.
  YES → Continue to Step 2.

Step 2: Full stack (MTP + EMA) beats 1.1147?
  YES → Run ablations (Step 3) to identify which change helped.
        Run 3-seed validation (Step 5).
        Report: new SOTA with 3-seed mean + std + t-test.
  NO  → Run ablations anyway to check individual contributions.
        If one ablation helps but the other hurts, try winning ablation alone.

Step 3a: MTP alone beats 1.1147?
  YES → Sweep MTP_LOSS_WEIGHT (Step 4).
  NO  → MTP is neutral/negative on this stack. Drop it.

Step 3b: EMA schedule alone beats 1.1147?
  YES → Try different decay ranges: (0.99,0.999), (0.995,0.9999), (0.997,0.9995).
  NO  → EMA schedule is neutral. Revert to fixed 0.997.

Neither ablation helps?
  → Our additions don't improve on the #1 stack.
  → Report: "#1 solution is already near-optimal. MTP and EMA schedule
     contribute 0 or negative on this architecture."
  → This is still a valid result — it eliminates two hypotheses.
```

---

## Output Format

Write results to `orchestrator/runs/RUN-<date>/runpod-results.json`:

```json
{
  "baseline_reproduction": {
    "seed": 314,
    "val_bpb": 1.1151,
    "artifact_bytes": 15863278,
    "steps": 6927,
    "step_avg_ms": 86.6
  },
  "full_stack": {
    "seed": 314,
    "val_bpb": "<result>",
    "mtp_num_heads": 1,
    "mtp_loss_weight": 0.15,
    "ema_decay_start": 0.99,
    "ema_decay_end": 0.9999,
    "artifact_bytes": "<size>",
    "steps": "<steps>"
  },
  "ablation_mtp_only": { ... },
  "ablation_ema_only": { ... },
  "best_config": {
    "3_seed_mean_bpb": "<value>",
    "3_seed_std": "<value>",
    "delta_vs_sota": "<value>",
    "welch_t": "<value>"
  }
}
```

---

## Constraints (DO NOT VIOLATE)

1. **16MB artifact budget** — measured AFTER LZMA-9 compression. `TARGET_MB=15.9` leaves margin.
2. **600s training time** — the script auto-stops. Do not modify the timer.
3. **No external data during quantization** — AR self-gen calibration only. Do not access val/train data post-training.
4. **Flash Attention 3 required** — the script imports `flash_attn_interface` directly. H100 Hopper kernels only.
5. **Do NOT modify quantization** — Int6 per-row with Full Hessian GPTQ is optimal. All LZMA-friendly alternatives have been tested and rejected (APoT, clustering, sparsity, mixed-precision all bust the 16MB budget).

---

## What Was Already Tested and Rejected

Do NOT re-test these — they have been empirically eliminated on Mac Mini M4:

| Idea | Result | Test Date |
|------|--------|-----------|
| Bell-curve bit allocation | 16.99 MB LZMA, over budget | 2026-04-01 |
| APoT quantization | 20.26 MB LZMA | 2026-04-01 |
| Weight clustering | 139% worse MSE | 2026-04-01 |
| Shared codebook | 16.32 MB LZMA, over budget | 2026-04-01 |
| 2:4 structured sparsity | 2315% worse MSE | 2026-04-01 |
| GPTQ ordering (reverse/middle-out) | 0% difference | 2026-04-01 |
| N-gram eval cache | Hurts BPB at all alphas | 2026-03-30 |
| TurboQuant rotation (weights) | Redundant with Full GPTQ | 2026-03-30 |
| Bias correction post-quant | 0.0002 BPP, not worth it | 2026-04-01 |

---

## Estimated Costs

- RunPod 8xH100 SXM: ~$20/hr
- Each run: ~12 min (10 min train + 2 min quant/eval)
- Baseline + full stack + 2 ablations: 4 runs = ~48 min = ~$16
- With sweeps and 3-seed validation: ~10-15 runs = ~$50-75

---

## Emergency Rollback

If something goes wrong, the default env vars reproduce #1 exactly:
```bash
# This IS the #1 solution — no MTP, no EMA schedule
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
