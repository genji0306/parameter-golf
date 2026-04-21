# Lessons Learned — Parameter Golf

**Last updated:** 2026-04-18 (post Go/No-Go v2)

## Index

| ID | Summary | Source |
|---|---|---|
| L01 | Rotation redundant at per-row precision; exception: bit-width boundaries | H38-A |
| L02 | Per-row Lloyd-Max codebook overhead (128 B/row) kills multi-matrix workloads | H72 |
| L03 | Lloyd-Max indices near-uniform (H=5.996/6.0); entropy coders can't help | H78 |
| L04 | Per-row norm + any global codebook → uniform indices | H71 |
| L05 | SDClip center-peak: H=2.53 bits/sym; distribution-matching is inverse | H70/71/72/78 |
| L06 | All Mac-Mini non-embedding quant improvements exhausted | Loops 5-7 |
| L07 | Gemma PLE/nested-width lose to shorter local window | autoresearch |
| L08 | Chunk-JEPA aux on existing recurrence hurts | autoresearch |
| L09 | SLSL alternating schedule loses to short-window-only | autoresearch |
| L10 | All S/L permutations exhausted; only short local span helps | L07+L09 |
| L11 | SOTA source already LZMA2-golfed in b85; 0 attack surface | H76 |
| L12 | Representation shift (chunking) is next credible novelty direction | Loop 8 synthesis |
| L12b | SEML per-row params scramble compressibility | H-EML probe |
| L13 | Cross-loop proxy claims must re-verify under current compressor | H64 overturn |
| L14 | Pre-quant calibration != post-quant eval | H74 |
| L15 | Shared grammar cell: -6 MB but +14.3% BPB (capacity starvation) | H-084 |
| L16 | Proxy compressor != real pipeline (bit-shuffle+Brotli) | H64+H-R4P2 |
| L20 | **Do NOT shrink below 11L/512d** — BQ1 4L/448d scored 1.2321 | BQ1 RunPod |
| L21 | Always verify BPB under corrected byte-accounting path | H85 scorer bug |
| L22 | **Each 1ms step overhead → 0.006 BPB** — throughput is multiplicative | Competition data |
| L23 | MTP confirmed 0.000 at this scale; MoE sparsity=0 below 500M; Mamba +2.7% | Loop 10 dead list |
| L24 | MoD saves FLOPs not params — wrong constraint for artifact-budget comp | Loop 10 |
| L25 | KD adds +0.003 BPB worse; quality curriculum zero effect | Loop 10 |
| L26 | Full BLT infeasible in 600s training | Loop 10 |
| L27 | **GDN dominates sub-1.05** — top-5 all GDN, best 1.00980 (PR #1711) | Leaderboard Apr 2026 |
| L28 | H85 GDN failure was SCORER not model — real GDN reaches 1.01 | Corrected leaderboard |
| L29 | O(n) linear attention is the structural advantage — more tokens/step | Competition arch |
| L30 | SpinQuant gives -0.005 BPB from quant quality alone (PR #1695) | Competition data |
| L31 | Brotli-11 saves ~700KB over zstd-22 | PR #1672 |
| L32 | Late QAT (Int6 STE final steps) reduces quant degradation | Top-5 common |
| L33 | Score-first TTT (SGD lr=0.005, 3 epochs) gives -0.005 to -0.010 | Top-10 common |
| L34 | PR #1676 trajectory readout only -0.0022, confounded with Muon momentum | Leaderboard |
| L35 | **H91 AdamW TTT: 1.1757 BPB — KILLED** | PQ1 RunPod |
| L36 | **H89 Late-SAM: 1.1015 BPB — KILLED** (still above 1.081 baseline) | PQ2 RunPod |
| L37 | **Do NOT trust additive BPB projections** without same-stack validation | Go/No-Go v2 |
| L38 | **Do NOT compound before isolated single-change validation** | Go/No-Go v2 |
| L39 | **Do NOT reopen H89/H91 without fundamentally different mechanism** | Go/No-Go v2 |
| L40 | MLX sandbox favored plain ReLU² over LeakyReLU(0.5)² — H93 deprioritized | MLX probe |
| L41 | **Merge-before-quantize unrolls shared layers** — H87 LoRA adds ~3.5 MB for 3 QKV copies | H87 research |
| L42 | RecurLoRA (PR #1552) attempted per-pass LoRA but was not adopted into SOTA — informative | Competition |
| L43 | **[CORRECTED] Our scorer fix was RIGHT** — buggy LUT deflated GDN by ~0.19 BPB, not the other way around | GDN audit |
| L44 | **H85 GDN confirmed KILLED at 1.223 BPB (3-seed)** under canonical scorer; 1.034 reading was the bug | GDN audit |
| L45 | **The ENTIRE Parameter Golf top-5 leaderboard uses the same buggy LUT** — their 1.01 scores are invalid | H96 audit |
| L46 | **Same bug closed PR #1687** — competition-wide scoring issue, not isolated to our run | Audit history |
| L47 | Non-canonical LUT pattern: wrong is_boundary defaults, missing is_byte/is_unused, bad leading-space stripping | LUT audit |
| L48 | GDN under corrected scoring starts 0.14 BPB behind SOTA — architecturally viable but not the free win | Corrected standings |
| L49 | Always verify BPB under canonical `build_sentencepiece_luts` BEFORE trusting new architecture scores | L44 |
| L50 | H88 per-loop LN adds only 27 KB fp16 (budget-invisible); zero quantization interaction | H88 research |
| L51 | H87 LoRA adds ~3.5 MB via merged-quantize unrolling (3 QKV copies) — budget constraint | H87 research |
| L52 | Codex chunking family (H-081/082/083) promoted alongside H87+ after GDN kill | Queue update 2026-04-18 |
| L53 | **MLA classic is LARGER than GQA-4 at head_dim=64** — MLA is KV-cache optimization, not param optimization at our scale | NV-002 research |
| L54 | **CLA (Cross-Layer KV Share) frees ~1.3M params** — share K,V spatially across adjacent layers, NOT temporally across recurrence | NV-002b research |
| L55 | **Differential Transformer KILLED at sub-10M** — 4 prior PRs (#932, #418, #345, #542) all worse; +30% step time = 0.18 BPB tax | NV-001 research |
| L56 | **Selective short-window on non-recurrence layers** is NOT in BQ1 kill family — BQ1 died from capacity loss (4L), not SWA itself | NV-003 research |
| L57 | **QA-LoRA with unmerged Int6** saves H87's 3.5 MB overhead; STE grid MUST match GPTQ grid exactly to avoid L14 kill family | NV-004 research |
| L58 | **Per-pass ALiBi is apparently novel** — no prior publication varies ALiBi per recurrence pass; RoPE + ALiBi coexist (Gemma 3 precedent) | NV-005 research |
| L59 | SOTA already uses GQA-4 (not MHA) — "moving to GQA" is already spent; must go further (GQA-2, MQA, or CLA) | NV-002b research |
| L60 | **New legal SOTA is 1.07882 (PR #1716)** — not 1.0810; target gap is now 0.029 BPB | PR analysis 2026-04-18 |
| L61 | PR #1716 Path A v3 passthrough int8 on control tensors (attn_scale/mlp_scale/resid_mix/skip_gates/skip_weights) is NOVEL quant target, not in any kill family | PR analysis |
| L62 | LZMA self-extracting code wrapper saves ~3× on loader code (53KB→18KB); keep LZMA on CODE segment only, Brotli terminal on WEIGHTS | PRC-002 research |
| L63 | **SpinQuant + adaptive sigma clip: MUST re-tune sigma post-rotation** (7-9σ MLP, 8-10σ ATTN, not 12/13σ) — rotation redistributes outliers | PRC-002 research |
| L64 | CLA-shared K,V must use SAME Hadamard rotation across sharing group when combined with SpinQuant — otherwise producer/consumer basis mismatch | PRC-002 research |
| L65 | AttnOutGate + SmearGate per-loop variants ~free (288+39 params extra) — recommended over shared | PRC-003/004 research |
| L66 | Sigmoid gates (AttnOutGate) are quantization-sensitive near derivative peak (0.5) — keep fp16, do NOT quantize | PRC-003 research |
| L67 | Quantization stack alone reaches only ~1.069 — need architectural reinvestment (CLA freed 1.3M params) to close to 1.050 | PRC-002 projection |
| L68 | Gate_in slice width in AttnOutGate (first 12 channels of x_orig) is likely empirical not load-bearing — sweep 8/12/16/24 | PRC-003 research |
| L69 | Casefold ablation decomposition: PR #1693 -0.00237 combined contains ~-0.0012 casefold + ~-0.0012 gates (estimate, HIGH uncertainty) | PRC-003 research |
| L70 | **PRC-005 multi-phase SGD TTT refines SOTA by only ~0.001 BPB** — the bulk of TTT gain (-0.005 to -0.010) is already in bigbag's existing score-first SGD | PRC-005 research |
| L71 | **N=2 phases sufficient** (boundaries [1000, 2000]) — captures ~70% of N=3 gain at lower phase-pause overhead | PRC-005 research |
| L72 | **QA-LoRA × Phased LoRA TTT = COMPOUND** (non-overlapping LoRA populations: trained-merged Int6 + live eval adapters) | PRC-006 research |
| L73 | **QA-LoRA × Global SGD base update = CONFLICT** — SGD on base invalidates QA-LoRA's Int6 grid. Freeze base or run separately | PRC-006 research |
| L74 | PR #1700 code audit: `activate_chunk_mask` is C3 guard; `all_gather_object` gathers only already-scored docs for legal base SGD | PRC-005 research |
| L75 | PR #1700 mechanism satisfies L39 on TWO axes (optimizer + protocol) — defensible "fundamentally different mechanism" claim | PRC-005 research |
| L76 | **CLA stride-2 KILLED** — Variant A (3 recipients): 1.07923041 (+0.000082 vs NS0, -297KB); Variant B (1 recipient): 1.07949064. Gate 1.058 missed by 0.021. Frees bytes, not BPB. | CLA RunPod 2026-04-20 |
| L77 | CLA's shared K/V freed ~300KB artifact (useful headroom!) but no quality gain — bytes freed, model capacity unchanged, BPB unchanged. | L76 derived |
| L78 | **NS1 import is structurally blocked** — PR #1700 depends on qo_bank/kv_bank, forward_ttt, parallel lambda tensors absent from NS0; requires full surgical port, not diff apply | NS1 audit 2026-04-20 |

## Kill Families

| Family | Mechanism | Victims |
|---|---|---|
| A | Distribution-matching quantizer → incompressible | H70,H71,H72,H78,H-EML |
| B | Per-row codebook overhead exceeds savings | H72 |
| C | Proxy compressor != real pipeline | H64, H-R4P2 |
| D | Pre-quant calibration != post-quant eval | H74 |
| E | Capacity starvation from aggressive weight sharing | H-084, BQ1 |
| F | Source code already maximally compressed | H76 |
| G | Rotation redundant at per-row precision | H38-A |
| H | Low-capacity reshapes bust BPB | BQ1 |
| I | Byte-accounting artifacts in scorers | H85 original |
| J | Throughput-negative (overhead > gain × 0.006/ms) | H86 |
| K | Scorer bugs masking valid architectures | H85 → H96 |
| L | **Scorer bugs INFLATING BPB — trusting buggy scores across leaderboard** | H85/H96/PR#1687 |
| M | **Weight sharing frees bytes but not BPB at this scale** — capacity unchanged, BPB unchanged; structural not quality win | CLA, H-084 |
