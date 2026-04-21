# Hypothesis Bank — Parameter Golf Sub-1.05 (v3, Post Go/No-Go)

**Updated:** 2026-04-18
**Target:** < 1.050 BPB
**SOTA:** 1.0810 BPB (bigbag PR #1493)
**Deadline:** April 30, 2026

---

## Binding Rules (from Go/No-Go v2)

1. No compound stacks before isolated H87 or H88 results exist
2. No reuse of old H85 branch as evidence for or against GDN
3. No reopening H89 (Late-SAM) or H91 (AdamW TTT) without fundamentally different mechanism
4. No additive BPB projections without same-stack validation
5. Track A (GDN) blocked until scorer audit clears

---

## Full Kill List

| ID | BPB | Kill Reason | Lesson |
|---|---|---|---|
| BQ1 | 1.2321 | 4L/448d capacity-starved | L20 |
| H85 | 1.2227 | Scorer bug (model may be valid → H96) | L21, L28 |
| H91/PQ1 | 1.1757 | AdamW TTT failed empirically | L35 |
| H89/PQ2 | 1.1015 | Late-SAM still above baseline | L36 |
| MTP | - | 0.000 BPB at this scale | L23 |
| MoE | - | Optimal sparsity = 0 below 500M | L23 |
| Mamba | - | +2.7% BPB at 512d | L23 |
| MoD | - | Saves FLOPs not params | L24 |
| KD | - | +0.003 worse (I/O) | L25 |
| BLT | - | Infeasible in 600s | L26 |
| H-100..H-106 | - | All based on killed 4L architecture | L20 |

---

## Track A: GDN — DEAD (2026-04-18)

### H85/H96: GatedDeltaNet — CONFIRMED KILLED
**Final status:** Dead at 1.223 BPB (3-seed canonical) — 0.14 behind SOTA

**What happened:** H85 initial run scored 1.034 under a buggy scorer (non-canonical
LUT pattern: wrong `is_boundary` defaults, missing `is_byte()`/`is_unused()`, bad
leading-space stripping). Same bug closed PR #1687. Canonical re-score: 1.223 BPB.

**Systemic finding (L45-L46):** The leaderboard top-5 (PR #1711, #1698, #1672, etc.)
appears to use the same non-canonical LUT. Their 1.01 scores are likely invalid
under canonical scoring. **Track A is closed — no further GDN RunPod spend.**

**Archived only:** The GDN artifact remains for forensic reference. It will not
be submitted or re-tested.

---

## Track B: Isolated Recurrence Differentiation (NOW THE ONLY PATH)

All probes run on the **exact** bigbag 11L/512d stack. ONE change per experiment.

### H87: Per-Loop LoRA Rank-4 Deltas — EXP-H87-SOTA-PerLoopLoRA
**Status:** PRIORITY-1 (next RunPod lane per queue update 2026-04-18)
**Platform:** MLX probe → RunPod validation
**Source:** Relaxed Recursive Transformers (ICLR 2025); PR #1552 RecurLoRA

**What:** Layers 3-5 are reused 3 times. Currently all 3 passes use identical weights.
Add rank-4 LoRA (A @ B^T) to passes 1 and 2 on the QKV projections only.
Pass 0 = base weights (identity).

**Training params:** ~49K LoRA params (negligible during training).

**STORAGE WARNING (L41):** Merge-before-quantize means storing 3 complete
quantized QKV copies per recurrence layer (one per pass). This adds **~3.5 MB**
to the artifact. Must verify this fits the bigbag budget before RunPod.
Alternative: rank-2 (halves LoRA params but same 3-copy storage cost).

**Throughput:** ~0.02ms (precomputed merges). BPB cost: 0.0001 (negligible).

**Test protocol:**
1. Establish exact baseline on MLX (unmodified bigbag stack)
2. Add LoRA to QKV only. Measure val_bpb. ONE CHANGE.
3. If positive on MLX: RunPod 3-seed validation

---

### H88: Per-Loop Unique LayerNorm + Level Embedding
**Status:** PRIORITY-2 (queued behind H87)
**Platform:** MLX probe → RunPod validation
**Source:** RingFormer (EMNLP 2025); PR #1552 pass index embeddings

**What:** Each recurrence pass gets its own LayerNorm parameters (gamma, beta) and
a learned level embedding added to the residual stream.

**Parameter overhead:**
- Per-loop LN: 3 passes × 3 layers × 2 LNs × (512 + 512) = ~18K params (fp16, excluded from Int6)
- Level embedding: 3 passes × 512 = 1.5K params
- Total: ~20K params, ~27 KB fp16. Budget-invisible.

**Throughput:** <0.3% overhead (LN is memory-bound, not compute-bound on H100).

**Test protocol:**
1. Establish exact baseline on MLX
2. Add per-loop LN + level embed. Measure val_bpb. ONE CHANGE.
3. If positive on MLX: RunPod 3-seed validation

---

## Track C: Codex Chunking (NEW, promoted after GDN kill)

The Codex chunking family (H-081, H-082, H-083) was promoted alongside H87+ after
the GDN kill. These are representation-shift hypotheses targeting token sequence
compression via learned chunk boundaries — a direction that previously seemed less
urgent when GDN appeared to be the solution.

### H-081: Codex Chunking Variant 1
**Status:** PRIORITY-3 (MLX probe)
**Details:** TBD — awaiting Codex team specification

### H-082: Codex Chunking Variant 2
**Status:** PRIORITY-4 (MLX probe)
**Details:** TBD — awaiting Codex team specification

### H-083: Codex Chunking Variant 3
**Status:** PRIORITY-5 (MLX probe)
**Details:** TBD — awaiting Codex team specification

All three must pass the same isolated-probe protocol as H87/H88:
single change on exact bigbag 11L/512d stack, measured against unmodified baseline.

---

## Stacking

### H98: SpinQuant (Hadamard Rotation Before GPTQ)
**Status:** PENDING (stacks with any winner from either track)
**Expected ΔBPB:** -0.005
**Source:** PR #1695

Hadamard rotation spreads weight outliers before GPTQ Int6 quantization.
Not a compression trick (cf. killed H38-A at per-row precision) — this is
a quantization prep step that improves Int6 fidelity.

---

## Deferred / Low EV

| ID | Reason |
|---|---|
| H94 (Engram n-gram) | Later fallback only; gating mandatory; compression risk |
| H86 (trajectory readout) | Only -0.0022, 3ms overhead, confounded |
| H93 (LeakyReLU 0.5²) | MLX favored plain ReLU² |
| H90 (length curriculum) | Good prior, not headline novelty |
| H95 (EMA self-distil) | Previous KD negative |

---

---

## Track D — Novel Ideas (NV-xxx, 2026-04-18)

Generated against the full kill list. Each idea either EXCLUDES killed families or
COMBINES lessons in a novel way. Full design in [novel-ideas.md](novel-ideas.md).

**Top-5 novel probes (priority order):**

| Rank | ID | Name | Strategy | Research |
|---|---|---|---|---|
| D1 | NV-002 | MLA (Multi-head Latent Attention) | EXCLUDE | Running |
| D2 | NV-003 | Selective short-window (non-recurrence layers) | COMBINE | Running |
| D3 | NV-004 | QA-LoRA + Int6 merge (may replace H87 storage cost) | COMBINE | Running |
| D4 | NV-001 | Differential Transformer | EXCLUDE | Running |
| D5 | NV-005 | Per-pass ALiBi slopes | COMBINE | Running |

**Secondary probes (no-research, direct MLX):**

| ID | Name | Cost | Status |
|---|---|---|---|
| NV-008 | BitFit-style TTT (LN-only) | Zero | Queue after H88 |
| NV-010 | Pass-indexed dropout masks | Zero | Queue after H88 |

**Speculative / later:**

| ID | Name | Status |
|---|---|---|
| NV-006 | Mixture of per-loop LoRAs | Wait for H87 result first |
| NV-007 | Hidden-state bottleneck | Wait |
| NV-009 | Scorer-aware training loss | Needs rules audit (compliance) |

---

## Execution Sequence (Track A dead — single path forward)

```
Phase 1: H87 isolated EXP-H87-SOTA-PerLoopLoRA ($15-24 RunPod)
    ↓ positive measured delta?
    ├── YES → Phase 2: H88 isolated probe ($8-15 RunPod)
    │         ↓ both positive?
    │         ├── YES → Phase 3: H87+H88 compound ($15 RunPod)
    │         │         → add H98 SpinQuant → submit
    │         └── NO → submit H87 only + H98
    └── NO → Phase 2: H88 isolated (fallback)
            ↓ positive?
            ├── YES → submit H88 + H98
            └── NO → escalate to Codex chunking (H-081/082/083 MLX probes)
                    ↓ positive?
                    ├── YES → RunPod validate → submit
                    └── NO → escalate: H94 or novel hypothesis generation
```

**Budget:** ~$40-60 total (single-track spend without GDN lane).
