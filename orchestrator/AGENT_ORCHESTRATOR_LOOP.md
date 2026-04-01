# Agent Instructions: Orchestrator Research Loop

**For:** Codex AL, RL, M agents and DarkLab swarm
**Goal:** Find improvements to beat the current best BPB on the Parameter Golf leaderboard
**Current #1:** 1.1147 BPB (abaybektursun, 2026-03-25)
**Our target:** <1.110 BPB

---

## Architecture of the Orchestrator

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  RL Agent    │  │  AL Agent    │  │  M Agent     │
│  (Research)  │  │  (Analysis)  │  │  (Design)    │
│              │  │              │  │              │
│ Hypotheses   │  │ Solution     │  │ Skills &     │
│ Literature   │  │ Catalog      │  │ Innovations  │
│ Frontiers    │  │ Gap Analysis │  │ Candidate    │
│              │  │ Ablations    │  │ Designs      │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────┬────────┘────────┬────────┘
                │    Synthesis    │
                ▼                 ▼
        ┌───────────────────────────┐
        │     Test Queue            │
        │  (Mac Mini M4 first,      │
        │   then RunPod 8xH100)     │
        └───────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │     CAR Agent             │
        │  (Controlled Autoresearch │
        │   Runner)                 │
        └───────────────────────────┘
```

---

## Hard Constraints (MEMORIZE THESE)

Every hypothesis MUST satisfy ALL of these or it is dead on arrival:

### 1. LZMA-9 Compressed Size ≤ 16 MB
- The artifact is `train_gpt.py` source code + LZMA-compressed model weights
- **Higher precision = higher entropy = LARGER after LZMA**
- Int6 per-row with percentile search is the Pareto-optimal quantization
- TESTED AND REJECTED: APoT, weight clustering, shared codebook, mixed-precision, bell-curve bit allocation, structured sparsity. ALL bust the 16MB LZMA budget.

### 2. Training Time ≤ 600 seconds on 8xH100 SXM
- ~6900 steps at ~87ms/step
- Any technique that adds >5ms/step must save >0.002 BPB to be worthwhile

### 3. No External Data During Quantization
- AR self-generated calibration only (model generates its own data)
- Cannot access train or val data after training ends

### 4. Flash Attention 3 (Hopper) Required
- The script imports `flash_attn_interface` directly
- Must run on H100 or newer

---

## What's Already in the Solution

| Component | Setting | Status |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA heads, 4 KV) | Fixed |
| MLP | 3x (1536), LeakyReLU(0.5)² | Fixed |
| Attention | XSA on all 11 layers | Fixed |
| BigramHash | 3072 × dim=112 | Fixed |
| RoPE | Partial (16/64 dims) | Fixed |
| Optimizer | Parallel Muon + AdamW | Fixed |
| Quantization | Full Hessian GPTQ (AR self-gen) | Fixed |
| Compression | LZMA preset=9 | Fixed |
| EMA | 0.997 (or scheduled 0.99→0.9999) | Testing |
| MTP | 0 or 1 head | Testing |
| SmearGate + VE128 + U-Net | Yes | Fixed |

---

## Where to Look for Improvements

### Tier 1: Training-side (no size impact)
These improve the model WITHOUT changing artifact size:

- **Optimizer tuning** — learning rate schedule, weight decay, momentum
- **MTP (Multi-Token Prediction)** — train to predict next 2-4 tokens, discard MTP heads before export. Zero size cost.
- **EMA scheduling** — ramp decay from 0.99 to 0.9999 over training
- **Sequence length curriculum** — start short, ramp to 2048
- **Batch size schedule** — start small, increase during training
- **Better initialization** — orthogonal init variants
- **Data ordering/curriculum** — present training data in optimal order

### Tier 2: Architecture (must fit param budget)
These change the model but must still fit in 16MB after Int6+LZMA:

- **Attention variants** — different XSA configurations, head dimensions
- **Activation functions** — other ReLU variants (SiLU, GELU, Swish)
- **Skip connection variants** — different U-Net configurations
- **Embedding tricks** — value embedding dimensions, shared embeddings

### Tier 3: Eval-time (no training impact)
These improve BPB at evaluation without changing training:

- **Sliding window stride** — already at 64, could try 32 or 128
- **Test-Time Training (TTT)** — already tried, negative on current stack
- **Ensemble/averaging** — if multiple seeds fit in budget

### DO NOT PURSUE (empirically eliminated)
- Any quantization change (Int6 per-row is optimal for LZMA budget)
- Mixed-precision anything (higher bits = bigger LZMA)
- KV cache compression (irrelevant — eval context is 2048)
- N-gram cache (hurts BPB)
- GPTQ ordering changes (0% difference for per-row quant)

---

## Research Loop Protocol

### Phase 1: Hypothesis Generation (RL Agent)

1. Read current hypothesis bank: `orchestrator/registry/hypothesis-bank.md`
2. Read recent experiment results: `orchestrator/runs/RUN-*/`
3. Search literature for techniques that satisfy the hard constraints
4. Generate new hypotheses with:
   - Clear mechanism of action
   - Expected BPB improvement (with evidence)
   - Size impact estimate
   - Whether it can be tested on Mac Mini M4 first

Output: `outbox/rl/hypothesis-bank.json`

### Phase 2: Solution Analysis (AL Agent)

1. Read the current top solution: `parameter-golf/records/track_10min_16mb/2026-04-01_MERGED_*/train_gpt.py`
2. Diff against leaderboard entries for techniques we're missing
3. Identify specific code locations where improvements can be applied
4. Rank hypotheses by expected impact and implementation effort

Output: `outbox/al/solutions-catalog.json`

### Phase 3: Design (M Agent)

1. Take top-ranked hypotheses from AL
2. Design minimal code changes (prefer env var control)
3. Write test plans for Mac Mini M4 validation
4. Write RunPod run commands

Output: `outbox/m/candidate-skill-list.json`

### Phase 4: Mac Mini M4 Pre-Test

**EVERY hypothesis must pass Mac Mini testing before RunPod.**

Protocol:
1. SSH to Mac Mini: `sshpass -p 'Opensens26' ssh cyber02@192.168.23.25`
2. Activate venv: `source ~/parameter-golf/.venv/bin/activate`
3. Run numpy simulation testing the idea
4. **ALWAYS check LZMA compressed size** — if >16MB, hypothesis is DEAD
5. Save results: `~/parameter-golf/EXP-{name}-results.json`

### Phase 5: RunPod Execution (CAR Agent)

Follow `AGENT_RUNPOD_INSTRUCTIONS.md` for RunPod testing.

### Phase 6: Result Integration

1. Record results in `orchestrator/runs/RUN-*/`
2. Update hypothesis bank with confirmed/rejected status
3. Commit to git
4. Feed results back to Phase 1 for next iteration

---

## Communication Protocol

### Agent → Agent Messages

Write to `orchestrator/runs/RUN-*/inbox/{agent}/` or `outbox/{agent}/`.

### File Formats

- Hypotheses: JSON with `{id, title, mechanism, expected_bpb, evidence, status}`
- Results: JSON with `{experiment_id, val_bpb, config, artifacts}`
- Plans: Markdown with run commands and decision trees

### Status Updates

Write to `orchestrator/runs/RUN-*/status.json`:
```json
{
  "phase": "testing",
  "current_experiment": "EXP-H46-A",
  "best_bpb": 1.1147,
  "target_bpb": 1.110,
  "experiments_run": 15,
  "experiments_remaining": 3
}
```

---

## Quick Start (Codex)

```bash
# Initialize a new run
python3 orchestrator/scripts/control_plane.py init-run --label loop5-beat-1.1147

# Seed with current state
python3 orchestrator/scripts/control_plane.py seed-briefs --run RUN-<id>

# Launch agents (one Codex session each)
python3 orchestrator/scripts/control_plane.py print-bootstrap --run RUN-<id> --agent rl
python3 orchestrator/scripts/control_plane.py print-bootstrap --run RUN-<id> --agent al
python3 orchestrator/scripts/control_plane.py print-bootstrap --run RUN-<id> --agent m

# After synthesis, launch CAR
python3 orchestrator/scripts/control_plane.py synthesize --run RUN-<id> --limit 3
python3 orchestrator/scripts/control_plane.py print-bootstrap --run RUN-<id> --agent car
```

---

## Quick Start (Antigravity / Standalone Agent)

If running outside the orchestrator framework:

1. Read this file + `AGENT_RUNPOD_INSTRUCTIONS.md`
2. Read current best solution: `parameter-golf/records/track_10min_16mb/2026-04-01_MERGED_*/`
3. Pick ONE hypothesis from the "Where to Look" section
4. Test on Mac Mini M4 first (LZMA size check mandatory)
5. If passes, run on RunPod following the decision tree
6. Report results as JSON to `orchestrator/runs/`
