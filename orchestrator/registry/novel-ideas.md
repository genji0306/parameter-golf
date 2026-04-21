# Novel Ideas Framework — Parameter Golf Sub-1.05

**Created:** 2026-04-18
**Purpose:** Generate genuinely novel hypotheses that respect the kill list.
Each idea either EXCLUDES killed approaches or COMBINES lessons in new ways.

---

## Design Rules

Every novel idea must satisfy:

1. **Not in any kill family (A-L)** — see [lessons-learned.md](lessons-learned.md)
2. **Stay on 11L/512d SOTA family** (L20)
3. **Measured under canonical scorer** (L49)
4. **Throughput-accountable** — 1ms cost = 0.006 BPB (L22)
5. **Isolated validation before compounding** (L38)
6. **No additive projections without same-stack validation** (L37)

---

## Two Generation Strategies

### Strategy A — EXCLUDE (fresh directions)
Ideas from completely different paper families than prior attempts.

### Strategy B — COMBINE (lesson-aware recomposition)
Ideas that take the *insight* from a killed approach and apply it differently.

---

## Track D — Novel Hypotheses (NV-xxx)

### NV-001: Differential Transformer Attention — **KILLED** 2026-04-18
**Source:** Microsoft 2024 (arXiv:2410.05258)
**Research finding:** 4 prior Parameter Golf submissions tested it:
- PR #932 (CoDA-GQA): val_bpb **1.1580** (+0.012 vs own baseline, +30% step time)
- PR #418 (last 2 layers): val_bpb 1.1715
- PR #345 (DART recurrent): val_bpb 1.8522 (irrelevant setup)
- PR #542 (DG Attention): val_bpb 1.1898 (+67 ms/step; β-gate collapse)

Root cause: **+30% step time → ~0.18 BPB headwind under L22 (1ms = 0.006 BPB).**
Throughput penalty swamps any quality gain at sub-10M scale. modded-nanogpt community
also rejected it for the same reason.

**Status:** KILLED. Do not pursue.

### NV-002: Multi-head Latent Attention (MLA) — **KILLED** 2026-04-18
**Source:** DeepSeek V2 (arXiv:2405.04434), V3 (arXiv:2412.19437)
**Strategy:** EXCLUDE — novel KV parameterization
**Research finding:** **Classic MLA is LARGER than GQA-4 at head_dim=64.** The KV up-projection
matrices exceed what's saved by compression. MLA saves params only at head_dim ≥ 128.
- GQA-4 baseline: 786K params/layer
- MLA latent=128: 1,114K params/layer (+328K per layer!)
- Step-time overhead: +40ms = +0.24 BPB under L22 rule
- Prior attempts (issues #354, #1589): pre-quant BPB worse by 0.013-0.07

**Status:** KILLED. MLA is a KV-cache optimization, not a param optimization at our scale.

### NV-002b: Cross-Layer Attention (CLA / K_KVShare_Wider) — **REPLACEMENT**
**Source:** Brandon et al. 2024 (arXiv:2405.12981); PR #1687/#1711 attempts
**Strategy:** EXCLUDE — different param-sharing axis (spatial, not temporal)
**Mechanism:** Layers 0,2,4,6,8,10 project their own K,V; odd layers reuse previous layer's K,V.
**Budget angle:** Frees ~1.3M params (~1 MB Int6) for reinvestment into width or depth.
**Critical interaction with depth recurrence:** Share K,V **spatially** (across adjacent layer indices),
NOT **temporally** (across recurrent iterations). Layer 4 reuses layer 3's K,V within the same pass.
**Reinvest:** width 512→544, or +1 unique layer, or mlp_mult 2→2.5.
**Expected ΔBPB:** -0.03 to -0.06 based on issue #1589 scaling data.
**Throughput:** Zero or negative (fewer ops).
**Competition evidence:** PR #1711 uses this + GDN (now killed under canonical scorer), but
CLA itself is architecture-independent and survives under canonical scoring.
**Status:** **PROMOTED — HIGH PRIORITY**. CLA without GDN is genuinely novel on the softmax stack.

### NV-003: Selective Short-Window (Non-Recurrence Layers Only) — **RESEARCH-VALIDATED**
**Source:** COMBINE — BQ1 (short window on 4L) + L20 (keep 11L/512d)
**Research finding (2026-04-18):**
- Strong precedent: Gemma 2/3 (5:1 local:global), SWAA (arXiv:2512.10411), Samba/Jamba/Hymba all validate
- BQ1 death was capacity starvation (4L/448d), not SWA per se — this proposal keeps 11L/512d intact
- Parallel residuals L7+ are *favorable* for SWA (attention's long-range contribution is already weak there)
- **FLOP savings: ~35% attention, ~9-10% total wall-time → ~55-60s extra training**
- Throughput is **negative overhead** (this is a win, not a cost)
- Expected net ΔBPB: **-0.003 to -0.008** (quality penalty -0.001 to +0.004 + throughput gain -0.005 to -0.010)

**Implementation:** `attention_window = [256,256,256,0,0,0,256,256,256,256,256]` (0 = full).
Use FlashAttention-2 `window_size=(W, 0)` for W≤512 (strictly faster).

**Status:** **PROMOTED — HIGH PRIORITY.** First MLX probe to run.

### NV-004: QA-LoRA with Unmerged Int6 — **RESEARCH-VALIDATED**
**Source:** COMBINE — H87 LoRA (L41 storage issue) + L14 post-quant calibration need
**Research finding (2026-04-18):**
- QA-LoRA (arXiv:2309.14717, ICLR 2024) uses STE quant during LoRA training so deltas are Int6-robust
- **Critical insight:** store base Int6 + LoRA Int6 UNMERGED — saves the entire 3.5 MB H87 overhead
- Total LoRA storage: ~28 KB Int6 (rank-4 × 3 QKV × 3 passes)
- **NOT** in Kill Family D (L14) because STE quantizes during the forward pass — model learns against
  the same grid that deploys. Verification gate: STE grid must exactly match GPTQ output.
- Quality delta over standard LoRA+PTQ at Int6: **-0.003 to -0.008 BPB**
- With storage savings reinvested: total **-0.005 to -0.015 BPB**
- Throughput: +0.05 ms unmerged path → +0.0003 BPB (negligible)

**Status:** **PROMOTED — HIGH PRIORITY.** Likely supersedes H87 as PRIORITY-1.

### NV-005: Per-Pass ALiBi Slopes — **RESEARCH-VALIDATED, NOVEL**
**Source:** COMBINE — H88 per-pass differentiation + ALiBi (Press et al.)
**Research finding (2026-04-18):**
- No prior publication varies ALiBi slopes per recurrence pass — **apparently novel**
- RoPE + ALiBi coexist (modify different parts of attention — multiplicative vs additive)
- Gemma 3 and Command-R7B both use hybrid positional schemes
- Parameter cost: 8 heads × 3 passes = 24 scalars = 48 bytes fp16 (budget-invisible)
- Throughput: +0.01 ms (one broadcast-add) → +0.00006 BPB (trivial)
- Expected ΔBPB: **-0.002 to -0.006** (analogy from per-pass LoRA's 1-2% PPL gain, scaled down)
- Complementary to H88: H88 differentiates residual normalization; NV-005 differentiates attention range
- Risk: slopes may collapse during training. Mitigation: orthogonality regularizer, small init scale

**Implementation:** `alibi_slopes[3, 8]` learned scalars, added to QK logits after RoPE.
Per-pass init: `2^(-8(h+1)/n_heads) * pass_scale[pass]`.

**Status:** **PROMOTED — cheap probe, run after H88.**

### NV-006: Mixture of per-loop LoRAs (MoLoRA)
**Source:** COMBINE — H87 LoRA + gated expert selection
**Mechanism:** Per recurrence pass, have 2-3 rank-2 LoRA "experts." A tiny router
(linear 512→2-3) selects expert per position.
**Parameter cost:** 2-3x H87 LoRA params + tiny router. Still <100K.
**Storage cost:** Same 3.5 MB issue as H87 unless NV-004 (QA-LoRA) is stacked.
**Expected ΔBPB:** Potentially larger than H87 if position-adaptive specialization helps.
**Status:** Speculative — needs research on whether tiny routers converge in 600s.

### NV-007: Hidden-State Bottleneck Between Recurrence Passes
**Source:** COMBINE — depth recurrence + forced compression (BLT-like insight without BLT)
**Mechanism:** Between passes, compress hidden state: 512 → 256 → 512 via tiny linear layers.
Forces compressed representations to flow between passes.
**Parameter cost:** 2 × (512×256 + 256×512) = 524K params → ~390 KB Int6. Significant but bearable.
**Risk:** Bottleneck may destroy information flow → BPB regression.
**Expected ΔBPB:** Highly uncertain. Could be -0.010 or +0.020.
**Status:** Speculative. Lower priority than above ideas.

### NV-008: BitFit-Style TTT (Bias-Only)
**Source:** COMBINE — H91 AdamW TTT kill (L35) + BitFit (Ben-Zaken et al. 2022)
**Mechanism:** At eval time, update ONLY LayerNorm gamma/beta (1.5K params).
Extremely cheap, may not trigger the H91 failure mode (which updated all params).
**Throughput:** Near-zero overhead (gradient through LN is tiny).
**Expected ΔBPB:** -0.001 to -0.003 at best.
**Status:** Low-effort probe. Could slot in as a cheap additive eval-time technique.

### NV-009: Scorer-Aware Training Objective
**Source:** COMBINE — L43-L49 canonical scorer findings + direct optimization
**Mechanism:** During training, the cross-entropy loss is computed in nats per token.
For Parameter Golf canonical scorer, BPB = loss / log(2) × tokens / canonical_bytes.
Since canonical byte count differs from our training tokenization byte count, we can
re-weight the per-token loss to directly optimize canonical BPB.
**Risk:** Violates competition rules if the re-weighting uses val-set info.
**Expected ΔBPB:** -0.003 to -0.010 if compliant. Risk of DQ if not.
**Status:** Needs rules audit. Requires reading Issue #1017 compliance doc.

### NV-010: Parameter-Free Signal Injection via Pass-Indexed Dropout Masks
**Source:** EXCLUDE — novel
**Mechanism:** At each recurrence pass, apply a FIXED pseudo-random dropout mask
(seed = pass_idx). Different masks per pass give different regularization per pass.
Zero parameters. Masks are generated deterministically at both train and eval time.
**Expected ΔBPB:** Highly speculative. Probably zero or slightly negative from noise.
**Status:** Cheap probe, low priority.

---

---

## Track E — PR-Inspired Combinations (PRC-xxx, 2026-04-18)

Added after analysis of competition PRs #1698, #1693, #1700, #1716, #1670.
Full audit in [PR-ANALYSIS-2026-04-18.md](../PR-ANALYSIS-2026-04-18.md).

### PRC-001: BigramHashEmbedding d=32 on Bigbag SOTA
**Source:** PR #1716 (himanshudongre, 2026-04-18)
**Strategy:** COMBINE — legal n-gram mechanism at tight size
**Mechanism:** Reduce bigram hash embedding dim from common 48/64 to 32.
Legal because it's a learned embedding indexed by bigram hash, not a lookup table
with random keys (not in Kill Family C n-gram hash concerns).
**Parameter cost:** ~200 KB at d=32 (vs ~300-400 KB at d=48/64).
**Throughput:** Zero (smaller embedding = cheaper gather).
**Measured in PR #1716:** bundled with Path A v3; -0.00218 BPB combined.
**Status:** MLX probe direct — cheapest safest change.

### PRC-002: Path A v3 Passthrough Quantization
**Source:** PR #1716
**Strategy:** EXCLUDE — novel quantization targets not in prior kill families
**Mechanism:**
- Per-tensor int8 for 5 control-tensor families: `attn_scale`, `mlp_scale`, `resid_mix`,
  `skip_gates`, `skip_weights` (previously fp16 passthrough)
- Per-row int8 for 3 small 2-D matrices: `bigram.proj`, `attn_gate_proj`, `smear_gate.weight`
- Combined with LZMA self-extracting code wrapper (18,097 bytes packed from 53,514 bytes source)
**Budget impact:** Saves ~0.5-1 MB that can be reinvested into width or +1 layer.
**Kill family check:** Not in Family A (no distribution-matching quantizer on weights).
Not in Family D (happens at artifact export, not pre-quant calibration).
**Status:** PROMOTED — direct MLX + RunPod validation.

### PRC-003: AttnOutGate (PR #1667 → #1693) — **per-loop variant recommended**
**Source:** PR #1667 @MarioPaerle, used in #1693
**Strategy:** COMBINE — zero-init gate that stacks safely with recurrence differentiation
**Mechanism:** `y * (2.0 * sigmoid(Linear(x[:,:,:12])))` — per-head multiplicative gate.
Zero-init → identity at init.
**Baseline (shared-across-passes):** 1,056 params (12 × 8 heads × 11 layers).
**Per-loop extension (recommended by research):** +288 extra params on recurrence layers
(3-5) × 3 passes. Total 1,344 params (~0.8 KB) — still trivial.
**Research finding:** Per-loop preferred because gate would otherwise average across
semantically different passes. Keep as fp16 (sigmoid gates quantization-sensitive
at derivative peak near 0.5).
**Throughput:** ~0.5-1.0 ms for 33 gate calls (11 layers × 3 passes) → +0.003-0.006 BPB
cost under L22.
**Expected isolated ΔBPB:** -0.0008 to -0.0012 (per-loop variant).
**Status:** PROMOTED — direct MLX probe, per-loop variant from launch.

### PRC-004: SmearGate (modded-nanogpt → #1693) — **per-loop variant recommended**
**Source:** @kellerjordan modded-nanogpt, reintroduced by @MarioPaerle
**Strategy:** COMBINE — input-dependent per-channel residual mixer
**Mechanism:** Mixes current token with previous token (strictly causal). Zero-init lambda.
**Baseline (shared):** 13 params. **Per-loop (recommended):** 39 params (13 × 3 passes).
**Research finding:** Per-loop ~free cost-benefit: Loop 1 smears raw embeddings (local
n-gram-like), Loop 2 smears partially-contextualized states, Loop 3 smears near-final
states — different functional operations against different input distributions.
**Expected isolated ΔBPB:** -0.0003 to -0.0006 (per-loop variant).
**Status:** PROMOTED — near-free probe, stack with AttnOutGate.

### PRC-005: Multi-Phase Global SGD TTT — **RESEARCH-VALIDATED**
**Source:** PR #1626 → #1670 → #1700 (code audited directly at lines 2188, 2475-2545)
**Strategy:** COMBINE — explicitly different mechanism from killed H91 (AdamW all-param)
**Verification:** Research agent audited PR #1700 `train_gpt.py` source (2976 lines).
- PR #1700 uses SGD momentum 0.9, cosine-decayed from lr=0.001, grad_clip=1.0
- Phase pause via `pause_flag_path` + `dist.barrier()` + `dist.all_gather_object(local_scored_docs)`
- C3 guard: `activate_chunk_mask = (num_chunks_t - 1 > ci).float()` — only already-scored chunks backward
- H91's failure mode (AdamW all-param drift) is NOT present
**Mechanism difference from H91 on TWO axes (satisfies L39):**
- Optimizer: SGD not AdamW
- Protocol: score-first phased pause with all-gather barrier, not interleaved
**Refinement only:** single-phase score-first SGD (PR #1610) already legal at -0.00056 BPB;
multi-phase refines by ~0.001 BPB. Bulk of gain (-0.005 to -0.010 per L33) is from standard
score-first SGD which is already in bigbag SOTA.
**N=2 recommended** (boundaries [1000, 2000]) over N=3 — saves 1 phase-pause of wallclock.
**Expected ΔBPB:** -0.001 incremental over SOTA's existing score-first SGD TTT.
**Status:** PROMOTED — adopt PR #1700 mechanism as primary eval-time TTT.

### PRC-006: Phased LoRA TTT — **RESEARCH-VALIDATED + CRITICAL STACKING WARNING**
**Source:** PR #1700 @jorge-asenjo (audited: `BatchedTTTLoRA` rank-96 at lines 1137-1205)
**Strategy:** COMBINE — parameter-efficient TTT (different from H91's all-param)
**Mechanism:** At eval time, `BatchedTTTLoRA` attaches FRESH rank-96 adapters to Q/K/V/O/MLP
of base blocks. Base weights are READ-ONLY. Adapters are re-instantiated per-document,
updated per-document, and discarded at doc/phase end.
**Why not blocked by L35:** Only ~30K LoRA params update at eval, not 8M base. Parameter
scope fundamentally different from killed H91.
**Stacking with NV-004 QA-LoRA — COMPOUND (verified):**
- QA-LoRA trains its own LoRA adapters during pretraining, merged/stored Int6
- Phased LoRA TTT attaches ORTHOGONAL live adapters at eval
- Non-overlapping parameter populations → legitimate compound
**CRITICAL: do NOT stack QA-LoRA with Global SGD base update leg (PRC-005).**
Global SGD updates base weights → invalidates QA-LoRA's Int6 grid. L63 territory.
Either freeze base during NV-004 runs (lose -0.0008 SGD gain), or run separately.
**Expected ΔBPB:** -0.005 to -0.010 (LoRA TTT standalone); compound with QA-LoRA pretraining:
full -0.005 to -0.015 NV-004 gain + ~-0.003 to -0.008 Phased LoRA TTT.
**Status:** PROMOTED — primary eval-time mechanism. Adopt N=2 variant.

### PRC-007: Int7 Embeddings
**Source:** PR #1700
**Strategy:** EXCLUDE — novel bit-width (between our int6 matrices and int8 embeddings)
**Mechanism:** Embeddings quantized to int7 (saves ~200 KB over int8).
**Kill family check:** Not Family A (int7 is uniform grid, not distribution-matching).
**Expected ΔBPB:** -0.000 to -0.001 (pure budget savings, quality likely neutral).
**Status:** PROMOTED — trivial probe; gain if budget savings used for reinvestment.

### PRC-008: Adaptive Per-Component GPTQ Sigma Clip
**Source:** PR #1670 (adaptive clip MLP=12σ, ATTN=13σ)
**Strategy:** EXCLUDE — per-component sigma tuning vs fixed SDClip
**Mechanism:** Different clipping sigma per tensor type during GPTQ.
MLP tensors clip at 12σ, attention tensors clip at 13σ.
**Kill family check:** Not pre-quant calibration (L14). Happens during GPTQ rounding.
**Critical interaction with SpinQuant (research finding):** If stacked with SpinQuant
Hadamard rotation, sigma values MUST be re-tuned post-rotation. Rotation redistributes
outliers, making 12/13σ too loose. Post-rotation optimum typically 7-9σ for MLP,
8-10σ for ATTN. **Reusing PR #1670 values naively will UNDERPERFORM.**
**Expected ΔBPB:** -0.000 to -0.002 standalone; additional -0.001 from post-rotation re-tune.
**Status:** PROMOTED — but requires re-tune protocol if stacked with H98 SpinQuant.

---

## Combined Novel + PR-Inspired Stack (NV × PRC)

The strongest legal compound stack combining our Track D + Track E:

| Component | Source | Expected ΔBPB | Artifact Impact |
|---|---|---|---|
| NV-002b CLA stride=2 | Our research + PR #1687 validated arch | -0.030 to -0.060 | Frees 1.3M params |
| NV-004 QA-LoRA unmerged Int6 | Our research | -0.005 to -0.015 | Saves 3.5 MB vs H87 |
| NV-003 Selective short-window | Our research | -0.003 to -0.008 | Zero params, throughput WIN |
| PRC-002 Path A v3 passthrough quant | PR #1716 | -0.001 to -0.003 | Saves 0.5-1 MB |
| PRC-003 AttnOutGate | PR #1667/1693 | -0.001 to -0.002 | +1,056 params |
| PRC-004 SmearGate | modded-nanogpt | -0.0003 to -0.0007 | +13 params |
| H88 per-loop LN | Our research + RingFormer | -0.005 to -0.010 | 27 KB fp16 |
| NV-005 per-pass ALiBi | Our research (novel) | -0.002 to -0.006 | 48 bytes |
| PRC-005 Multi-phase SGD TTT | PR #1626 | -0.002 to -0.005 | Zero |
| PRC-006 Phased LoRA TTT | PR #1700 | -0.001 to -0.003 | Zero (eval-only) |
| H98 SpinQuant | Our research + PR #1695 | -0.005 | Zero |

**Raw sum:** -0.055 to -0.115 BPB
**Compound (×0.5 conservative):** -0.028 to -0.058 BPB
**Compound (×0.6 moderate):** -0.033 to -0.069 BPB
**Projected BPB:** 1.012 to 1.056

**Reality check:** Per L37, no additive projections without same-stack validation.
Every compound pair must be measured. The upper-bound exists only if every component's
own mechanism survives isolated validation under canonical scorer.

---

## Final Priority Ranking — POST RESEARCH (Track D)

**4 hypotheses promoted, 2 killed, 5 deferred/secondary.**

| Rank | ID | Name | Expected Net ΔBPB | Param/Storage | Throughput | Status |
|---|---|---|---|---|---|---|
| **D1** | **NV-002b** | **CLA (Cross-Layer KV Share stride=2)** | **-0.030 to -0.060** | Frees 1.3M params | Negative (faster) | PROMOTED |
| **D2** | **NV-004** | **QA-LoRA + unmerged Int6** | **-0.005 to -0.015** | ~28 KB (saves 3.5 MB vs H87) | +0.05 ms | PROMOTED (supersedes H87?) |
| **D3** | **NV-003** | **Selective short-window (non-recurrence)** | **-0.003 to -0.008** | Zero param change | -5 to -8 ms (win) | PROMOTED |
| **D4** | **NV-005** | **Per-pass ALiBi slopes** | **-0.002 to -0.006** | 48 bytes fp16 | +0.01 ms | PROMOTED (novel) |
| D5 | NV-008 | BitFit-TTT (LN-only eval) | -0.001 to -0.003 | Zero (eval) | Near-zero | Direct probe |
| D6 | NV-010 | Pass-indexed dropout masks | Uncertain | Zero | Zero | Direct probe |
| ~~-~~ | ~~NV-001~~ | ~~Differential Transformer~~ | **+0.077 (killed)** | - | +30ms → 0.18 BPB tax | KILLED |
| ~~-~~ | ~~NV-002~~ | ~~MLA (classic)~~ | **negative (killed)** | Larger than GQA-4 | +40ms → 0.24 BPB | KILLED |
| D7 | NV-006 | MoLoRA | Unknown | Medium | Unknown | Deferred |
| D8 | NV-007 | Hidden-state bottleneck | Speculative | 390 KB | Unknown | Deferred |
| D9 | NV-009 | Scorer-aware training loss | Potentially -0.003 to -0.010 | Zero | Zero | COMPLIANCE AUDIT REQUIRED |

**Cumulative upper-bound (if all 4 promoted stack with 60% compound factor):**
- Raw sum: -0.089 to -0.044 BPB
- Compound: -0.053 to -0.026 BPB
- **Projected BPB: 1.028 to 1.055** ← potentially below target

(Per L37, do NOT trust additive projections without same-stack validation. This is
a pre-experimental upper bound only.)

---

## Integration with Existing Queue

Track D runs in **parallel** with Tracks B (H87, H88) and C (Codex chunking):
- **MLX probes for NV-003, NV-005, NV-008 can start immediately** ($0, no research needed)
- **NV-002, NV-004, NV-001 wait for research completion** (agents running now)
- **NV-006, NV-007, NV-009, NV-010 deferred to Wave 2**

**No compounding of Track D ideas until isolated validation** (L38).
**All Track D probes must use canonical scorer** (L49).

---

## Success Criteria for Track D

- Minimum: Any NV-xxx probe produces a positive measured delta on isolated MLX probe
- Target: At least one NV-xxx probe reaches <1.060 on 3-seed RunPod
- Primary: Compound of a NV-xxx winner + H87 or H88 reaches <1.050 BPB
