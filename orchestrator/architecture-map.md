# Parameter Golf — Complete Architecture Map

**Source:** openai/parameter-golf, 1,668 PRs scanned, 2026-04-19
**Current locked benchmark:** `K_KVShare_Wider FLA (Opensens reproduction)` at `1.03385760 BPB` (3-seed mean, no TTT).
**Canonical control reference:** Bigbag softmax baseline (11L/512d, GQA-4, depth-rec L3-5, parallel-res L7+, QK-gain 5.25,
MuonEq-R, SDClip Int6, SP8192, legal TTT).
**Systems note:** `TMA G3` already passed on the control line at `415.3298 ms -> 374.5369 ms` (`-9.8218%`).

Status codes:
- `BENCHMARK` — strongest verified local record; default comparison target
- `SOTA` — in merged bigbag baseline
- `PROMOTED` — in v6 plan, awaiting MLX or RunPod
- `FRONTIER` — canonical-valid, not yet in our plan
- `PENDING` — legal ruling required
- `KILLED` — empirically or throughput-killed with evidence
- `INVALID` — scorer-poisoned or rules-invalid

---

## 1. ATTENTION MECHANISMS

### Softmax / Standard
| Mechanism | Status | Best BPB | Source |
|---|---|---|---|
| GQA-4 (4 KV / 8 Q heads) | SOTA | — | baseline |
| Partial RoPE (16/64 dims) | SOTA | — | baseline |
| QK-gain 5.25 per-head | SOTA | — | #1334 [M] |
| QK-gain 5.4 / 5.5 | FRONTIER | 1.0815/1.0810 | #1706/#1714 |
| YaRN-2048 positional | SOTA | — | #640 [M] |
| CLA stride-2 (NV-002b) | KILLED (M) | +0.000082 vs NS0 | empirical 2026-04-20 — frees bytes, zero BPB recovery |
| Per-pass ALiBi (NV-005) | PROMOTED | -0.002–0.006 | novel |
| Prefix absolute pos table 64× (NV-009) | PROMOTED | -0.003–0.008 | novel |
| AttnOutGate per-head zero-init (PRC-003) | PROMOTED | ~-0.001 | #1667/#1693 |
| K_KVShare_Wider (wider CLA variant) | BENCHMARK | 1.03385760 | #1687 arch + 2026-04-17 Opensens reproduction |
| Sliding-window only on non-rec layers | PROMOTED (NV-003) | -0.003–0.008 | novel |
| VarLen flash attention | FRONTIER | 1.0740–1.077 | #1626/#1560 |
| Krylov gated attention | FRONTIER | 1.0960 | #1446 [O] |
| Breadcrumb gating | FRONTIER (low) | 1.1803 | #1724 [O] |
| QuantGate (gating on quant path) | FRONTIER | 1.0655 | #1736 [O] |
| Attention sink tokens (NV-012) | PROMOTED (legality?) | -0.004–0.012 | novel |
| McGilchrist Register Token + FiLM | FRONTIER | unclear | #1022 [O] |
| Gemma-style attention + Gram NS | FRONTIER | unclear | #1674 [O] |

### Linear / Sub-quadratic
| Mechanism | Status | Best BPB | Source |
|---|---|---|---|
| GatedDeltaNet / GDN | INVALID | ~1.177 canonical | #1711/#1698 [C] |
| FLA / Flash Linear Attention | INVALID | ~1.177 canonical | Issue #1719 |
| GDN-Hybrid [GDN×5→SWA×1 repeating] | INVALID (score) / FRONTIER (arch) | 1.01195 claimed | #1672 [C] |
| DeltaNet plain | INVALID | same family | #1632 |
| RWKV token-shift hybrid | FRONTIER (weak) | uncompetitive | #1112 [O] |
| MASA low-rank shared attention | FRONTIER (weak) | 1.3579 | #1025 [O] |

### Sparse / Windowed
| Mechanism | Status | Best BPB | Source |
|---|---|---|---|
| XSA4 / Sliding-window | SOTA | — | #315 [M] |
| XSA-all (all layers windowed) | FRONTIER | 1.06160 | #1694/#1473 [O] |
| XSA5LastGated | FRONTIER | competitive | #1495/#1448 |
| XSA6 / XSA7 / XSA-11 | FRONTIER | various | #776/#1182/#1324 |
| Tight SWA w=256 bank | FRONTIER | competitive | #1512 [O] |

---

## 2. RECURRENCE / DEPTH

| Mechanism | Status | Best BPB | Source |
|---|---|---|---|
| 3-layer depth recurrence L3-5 (frac=0.35) | SOTA | — | #1437 [M] |
| Per-loop LN + level embed (H88) | KILLED (empirical) | +0.000109 vs NS0 | Variant A missed 2026-04-20; Variant B size-blocked |
| Selective short-window on non-rec (NV-003) | PROMOTED | -0.003–0.008 | novel |
| 4-layer recurrence (loop_end=6) | FRONTIER | unknown | #1678 [O] |
| Triple loop depth recurrence | FRONTIER | 1.0760–1.0848 | #1450/#1420/#1555 |
| Loop45 (45-iteration depth loop) | PROMOTED (G4 next) | 1.0655 | #1736 [O] — next only after clean TMA transfer on target stack |
| Depth-recurrence ×2 stacked | FRONTIER | 1.0797 | #1572 [O] |
| Depth-rec with adaptive frac (learned) | FRONTIER | untested | novel |
| Two independent recurrence groups | FRONTIER | untested | novel |
| Universal Transformer (all layers loop) | KILLED (J) | throughput | #1640 [O] |
| Mamba-3 SSD hybrid / Nemotron-H | FRONTIER (weak) | 1.1473 | #1644 [O] |
| H-Net dynamic chunking | FRONTIER (weak) | 1.2070 | #1305 [O] |
| Mixture of Depths (MoD) | KILLED | — | #957 [O] |
| Multi-Token Prediction (MTP) | KILLED | — | #1691 [C] |
| ACT Adaptive Computation Time | FRONTIER (weak) | uncompetitive | #1293 [O] |
| Fractal recurrent primitive | FRONTIER (low) | — | #1569 [C] |

---

## 3. MLP / FEEDFORWARD

| Mechanism | Status | Best BPB | Source |
|---|---|---|---|
| LeakyReLU(0.5)² 4× | SOTA | — | baseline |
| Fused Triton MLP kernel | FRONTIER → PROMOTED | 1.0720 | #1626/#1700 |
| TMA Megakernel (fused FC+activation) | PROMOTED (G3 passed) | 415.3298 -> 374.5369 ms | #1672/#1555/#1450 — passed systems gate on 2026-04-20 |
| Fused Softcap+CE megakernel | FRONTIER | 1.94× speedup | #915 [O] |
| SwiGLU activation | FRONTIER | uncompetitive so far | #73 [M], #505 |
| Poly5 / Softcap | FRONTIER | competitive | #1325 |
| SmearGate per-channel (PRC-004) | PROMOTED | ~-0.0005 | #1667/#1693 |
| MoE (full) | KILLED (E/B) | artifact too large | — |
| Mini-MoE (small random) | FRONTIER (weak) | 1.117 | #1692 [O] |
| HedgeMixer 6-expert | FRONTIER | 1.1105 legal | #849 [O] |
| BankLinear cross-layer shared | FRONTIER (weak) | 1.1091 | #1315/#1089 |
| DeltaShare MLP (NV-015) — V7 Phase 2 | V7 QUEUED | -0.002+ est | DeltaLLM-inspired: shared middle-block MLP + rank-4 deltas |
| Bilinear logit head rank-4 (NV-010) | PROMOTED | -0.004–0.010 | novel |
| KAN Kolmogorov-Arnold | KILLED | negative result | #1537 [O] |
| Bottleneck MLP projection | FRONTIER | untested | novel |
| Gated MLP conditioned on rec-pass index | FRONTIER | untested | novel |

---

## 4. TOKENIZER / VOCABULARY

| Mechanism | Status | Best BPB | Source |
|---|---|---|---|
| SP8192 + BigramHash d=32 | SOTA | 1.07882 | #1716 [O] |
| CaseOps reversible lossless (PRC-010) | PENDING | 1.03540 | #1738 [O] |
| Casefold V4 (PRC → legal pending) | PENDING | 1.05733 | #1693/#1670 |
| Embedding factorization r=128 (NV-013) | PROMOTED | -0.010–0.025 | novel |
| Int7 embeddings (PRC-007) | PROMOTED | ~-0.001 | #1700 |
| Fractional-byte vocab / BPE-merge (NV-007) | PROMOTED (scorer test first) | -0.010–0.030 | novel |
| Larger vocab SP16384 | KILLED (B) | artifact | — |
| VE / Value Embeddings (various dims) | FRONTIER (weak) | 1.1588 | #1009/#586 |
| BESE 288-vocab novel alphabet | FRONTIER (weak) | 1.1276 | #1327 [O] |
| Geodesic Topological Tokenizer | FRONTIER (exotic) | untested | #1571 [O] |
| BigramHash d=48/64 (larger) | FRONTIER (regress) | worse than d=32 | — |
| TrigramHash / QuadgramHash | FRONTIER (weak) | 1.0887 | #486/#882 |
| Vocab4096 + MLP4× | FRONTIER (weaker base) | 1.0979 | #1218 [M] |
| Hashed n-gram cache | INVALID | Issue #677 | — |

---

## 5. ARCHITECTURE TOPOLOGY

| Mechanism | Status | Best BPB | Source |
|---|---|---|---|
| Parallel residuals L7+ GPT-J | SOTA | — | #1412 [M] |
| Skip gates sigmoid U-Net | SOTA | — | baseline |
| U-Net Transformer symmetric | FRONTIER | 1.1239–1.1570 | #640/#641 [M] |
| Value Residuals | FRONTIER | 1.0887 | #486/#1182 |
| DenseNet-style layerwise skip | FRONTIER | untested | novel |
| Asymmetric parallel residuals | FRONTIER | untested | novel |
| Attention sink tokens (NV-012) | PROMOTED (legality?) | -0.004–0.012 | novel |
| Trajectory-State Readout (Parcae) | FRONTIER | 1.0788 | #1676/#1703 |
| HELIX MoR K7R2 U-Net | FRONTIER (exotic) | unclear | #1600 [O] |
| LLM-JEPA alternative objective | FRONTIER (weak) | 1.2699 | #1277/#1480 |
| MDLM masked diffusion | FRONTIER (weak) | 1.3428 | #1582 [O] |

---

## 6. TEST-TIME ADAPTATION

| Mechanism | Status | Best BPB | Source |
|---|---|---|---|
| Score-first SGD TTT (3 ep, 32K chunks) | SOTA | — | #1413 [M] |
| Multi-phase global SGD TTT N=2 (PRC-005) | PROMOTED | ~-0.001 | #1700 [O] |
| Phased LoRA TTT (PRC-006) | PROMOTED | -0.005–0.015 | #1700 [O] |
| Parallel pre-quant TTT (PRC-009) | PENDING | 1.0429 | #1735 [O] |
| Pre-quant TTT 10ep + QK5.25 | FRONTIER | 1.0600 | #1487 [C] |
| VarLen doc-TTT chunk-48 | FRONTIER | 1.0741 | #1560 [O] |
| MP-SGD TTT + SpinQuant V1 | FRONTIER | 1.0759 | #1695 [O] |
| AdamW TTT all-param (H91) | KILLED | 1.1757 | empirical |
| LoRA TTT (merged precedent) | FRONTIER | 1.195 merged | #77 [M] |
| FiLM-only TTT | FRONTIER (weak) | 1.3151 | #1383 [O] |
| SLOT per-doc adaptation | FRONTIER (legality unclear) | 0.65–0.94 | #1324+ |
| SAM inner loop for meta-TTT | FRONTIER | unclear | #1601 [O] |
| FOMAML meta-TTT | FRONTIER (weak) | unclear | #494 [O] |

---

## 7. QUANTIZATION / COMPRESSION

| Mechanism | Status | Best BPB | Source |
|---|---|---|---|
| GPTQ SDClip Int6 matrices | SOTA | — | #1394 [M] |
| Int8 SDClip embeddings | SOTA | — | #1394 [M] |
| Path A v3 passthrough int8 (PRC-002) | PROMOTED | ~-0.002 | #1716 [O] |
| Byte-shuffle + Brotli-11 | SOTA | — | baseline |
| LZMA self-extracting code wrapper | SOTA | — | baseline |
| SpinQuant Hadamard (H98) | PROMOTED | ~-0.005 | novel |
| Adaptive sigma per-component (PRC-008) | PROMOTED | ~-0.001 | #1670 [O] |
| QA-LoRA STE Int6 (NV-004) | PROMOTED | -0.005–0.015 | novel |
| Lloyd-Max 6-bit embeddings (H64) | FRONTIER | -214KB | loop-6 |
| AWQ + Hessian mixed precision (PRC-011) | FRONTIER | 1.0785 | #1732 [O] |
| FreqGPTQ (frequency-aware GPTQ) | FRONTIER | WIP | #1743 [O] |
| Hadamard-rotated GPTQ | FRONTIER | 1.1035 | #1400 [C] |
| Hessian-aware GPTQ block-diagonal | FRONTIER | competitive | #1689 [O] |
| ANS weight compression (lossless) | FRONTIER | -13.9% vs LZMA | #1510 [O] |
| Ternary / BitNet 1.58 | FRONTIER (weak) | 1.2029–1.2196 | #1273 [O] |
| fp8 e4m3 | FRONTIER (weak) | used in MDLM | #1699 [O] |
| Custom Brotli dictionary | KILLED | insufficient patterns | H68 |
| QJL 1-bit JL residual | FRONTIER (parked) | implementation bug | H63 |

---

## 8. OPTIMIZER / TRAINING DYNAMICS

| Mechanism | Status | Best BPB | Source |
|---|---|---|---|
| MuonEq-R (Newton-Schulz 5 steps) | SOTA | — | #1285 [M] |
| AdamW for scalars/embeddings | SOTA | — | baseline |
| NeoMuon variant | FRONTIER | competitive | #1641 [O] |
| FlashMuon + LinearScaleInit | FRONTIER | competitive | #1495/#1448 |
| Turbo-Muon | FRONTIER (weak) | 1.1091 | #1089 [O] |
| QK-gain as training-time variable | SOTA | — | baseline |
| QAKD-OS FP16 snapshot KD (NV-011) | PROMOTED | -0.006–0.015 | novel |
| Entropy-conditional token curriculum (NV-006) | PROMOTED | -0.008–0.020 | novel |
| Seqlen curriculum 512→2048 (NV-014) | PROMOTED | -0.010–0.020 | novel |
| Stochastic depth gradient avg (NV-008) | PROMOTED | -0.005–0.012 | novel |
| Compressibility regularization | FRONTIER (weak) | 1.1135 | #1508 [C] |
| TWEO early-cosine outlier reg | FRONTIER | unclear | #1636 [O] |
| Norm-PCT-Dropout | FRONTIER | 1.0824 | #1520 [O] |
| Label smoothing + Bank QAT | FRONTIER (weak) | 1.1352 | #667 [O] |

---

## 9. NET-NEW FINDINGS vs V6 PLAN (additions from full PR scan)

These were NOT in the v6 plan registry and have measurable canonical-valid evidence:

| ID | Mechanism | BPB | Why interesting |
|---|---|---|---|
| **NEW-1** | Loop45 (45-iteration depth loop, #1736) | 1.0655 | Same PR as CaseOps; isolate the loop contribution |
| **NEW-2** | Triple loop depth recurrence (#1450/#1555) | 1.076–1.085 | 3 loops of L3-5 vs single 3-pass — more recurrence without GDN |
| **NEW-3** | TMA Megakernel fused FC+activation (#1555) | 1.0764 | Systems gain that enables more arch complexity without throughput cost |
| **NEW-4** | Pre-quant TTT 10ep + QK5.25 (#1487) | 1.0600 | Independent pre-quant TTT data point before #1735's 21ep |
| **NEW-5** | ANS weight compression lossless (#1510) | −13.9% artifact | Frees ~2.2MB vs current Brotli-11 — same info, smaller bytes |
| **NEW-6** | FreqGPTQ frequency-aware GPTQ (#1743) | WIP 2026-04-19 | Newest quantization variant; published same day as this sweep |
| **NEW-7** | Krylov gated attention (#1446) | 1.0960 | Novel attention class, not killed, measurable BPB |
| **NEW-8** | Parcae Trajectory-State Readout (#1676) | 1.0788 | Novel readout path (4-PR family); distinct from standard LM head |
| **NEW-9** | VarLen FA3 attention (#1626/#1560) | 1.0740 | Variable-length chunked attention; throughput-positive |
| **NEW-10** | XSA-all / XSA5LastGated (#1694/#1495) | ~1.06 | All-layer sparse attention variants beyond baseline XSA4 |
| **NEW-11** | QK5.5 scan (#1714/#1715) | 1.0810 | Higher QK-gain still at frontier; 5.5 not yet tried on our stack |
| **NEW-12** | Norm-PCT-Dropout (#1520) | 1.0824 | Combined with legal TTT + gated attn; novel regularizer |

---

## 10. PRIORITY ADDITIONS FOR V6 PLAN

Ranked by expected BPB × novelty vs current stack:

1. **Benchmark-first composition** — treat `1.03385760` from `K_KVShare_Wider` as the default comparison line, not the older softmax-only trust anchor.
2. **ANS compression (NEW-5)** — lossless 13.9% artifact savings → frees ~2.2MB for more capacity or tighter quantization. Zero BPB cost at inference. Add to Phase 0 preflight as a free multiplier on all other gains.
3. **TMA Megakernel (NEW-3)** — fused FC+ReLU²+gradient kernel already passed the systems gate on the control line. Next job is transfer onto the benchmark family.
4. **Loop45 / deep depth recurrence (NEW-1)** — conditional on a clean TMA transfer. If 45 loops run in the same wall-clock as current 3 loops via fusion, expected BPB gain is significant.
5. **Pre-quant TTT 10ep stack (#1487) (NEW-4)** — independent data point confirming PRC-009's pre-quant TTT direction. 1.0600 BPB vs our old softmax anchor = strong mechanism signal, but it is still legal-gated.
6. **VarLen FA3 (NEW-9)** — variable-length chunked attention gives better TTT chunk boundaries and is throughput-positive on long sequences.
7. **Krylov gated attention (NEW-7)** — 1.0960, no kill family hit, novel mechanism class worth one MLX probe.
8. **QK-gain 5.5 (NEW-11)** — monotonic gain curve not yet exhausted; trivial to test.
