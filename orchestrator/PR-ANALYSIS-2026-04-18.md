# Competitive PR Analysis — 2026-04-18

Five PRs analyzed against our kill list and canonical-scorer discipline.

---

## Summary Table

| PR | Author | Claimed BPB | Family | Canonical-Valid? | Legal? | Notable |
|---|---|---|---|---|---|---|
| #1698 | arsenis-cmd | 1.00995 | GDN+FLA | **NO** (scorer bug — same as H85) | Yes | K_KVShare_Wider arch, legal TTT |
| #1693 | dexhunter | 1.05733 | Softmax | Likely yes | Casefold pending #1604 | AttnOutGate+SmearGate zero-init gates |
| #1700 | jorge-asenjo | 1.07219 | Softmax | Likely yes | **Yes (no casefold)** | Multi-phase global SGD + phased LoRA TTT |
| #1716 | himanshudongre | **1.07882** | Softmax | **Yes** | **Yes** | BigramHash d=32 + Path A v3 passthrough quant |
| #1670 | dexhunter | 1.05970 | Softmax | Likely yes | Casefold pending #1604 | Casefold + multi-phase global SGD TTT |

**Our confirmed baseline: 1.08100 (bigbag) or 1.0810 canonical reference.**

---

## PR #1698 — GatedDeltaNet FLA + Legal TTT (1.00995)

**Verdict: POISONED by H85/L28 finding.**

- Architecture: K_KVShare_Wider (10L/544d/8 heads, KV stride=2), FLA library
- Same scorer vulnerability as our killed H85 (L44-L46)
- Non-canonical LUT pattern likely inflates the apparent 1.01 score
- Uses zstd-22 (not Brotli-11) — inferior compression by ~700KB
- Artifact 16,600,916 bytes reported as "15.83 MiB / 16 MiB" → using MiB (16,777,216) not MB (16,000,000)

**Kill families triggered:**
- Family I (byte-accounting artifact) — inherited from H85
- Family K (scorer bugs masking valid architectures)

**Salvageable components:**
- **KV sharing stride=2** (= CLA / our NV-002b) — validates our promoted idea
- **Legal score-first TTT recipe** (SGD lr=0.005, 3 epochs, 32K chunks, freeze first 2 blocks)
- Int6 matrices + Int8 embeddings — already in our stack

**Action:** Treat the 1.00995 number as likely invalid. Mine the CLA + TTT recipe only.

---

## PR #1716 — BigramHash d=32 + Path A v3 Quantization (1.07882) — 2026-04-18

**Verdict: STRONGEST LEGAL SUBMISSION. Measured under canonical scorer.**

This is the single most important PR. Published TODAY. Two changes on the exact bigbag
SOTA stack:

### Change 1: BigramHashEmbedding d=32
- Reduced from common d=48 or d=64 to d=32
- Less embedding overhead → more budget for weights

### Change 2: Path A v3 Aggressive Passthrough Quantization
- **Per-tensor int8** for 5 control-tensor families:
  - `attn_scale`, `mlp_scale`, `resid_mix`, `skip_gates`, `skip_weights`
- **Per-row int8** for 3 small 2-D matrices:
  - `bigram.proj`, `attn_gate_proj`, `smear_gate.weight`
- These were previously fp16 passthrough (uncompressed)
- Combined with **LZMA self-extracting code wrapper**
- Full int8 embeddings + int6 matrices recipe now fits ≤16 MB

### Results (3-seed canonical scorer)
- **Post-TTT BPB: 1.07882** (3-seed mean, std 0.000143)
- Artifact: 15,991,203 - 15,996,103 bytes
- Training: ≤588s. Eval: ≤481s.
- **Delta vs bigbag: -0.00218 BPB = -0.00564 nats/token**
- **Z = -2.998, p = 0.00136** (clears 0.005-nat significance bar)

### Compliance
- All 4 Issue #1017 conditions verified
- No casefold, no SLOT, no n-gram cache, no ETLB
- Seeds match bigbag convention (42, 314, 999)

**Novel insights for our campaign:**
- Control tensors that were "untouchable" passthroughs are quantizable to int8
- LZMA self-extracting code wrapper saves code bytes
- d=32 BigramHash is tighter than literature assumes for this scale

---

## PR #1693 — Casefold V4 + AttnOutGate + SmearGate + Phased SGD (1.05733)

**Verdict: PARTIALLY USABLE. Casefold is pending legality; gates are universally legal.**

### Components (isolated impact):

**AttnOutGate (@MarioPaerle, PR #1667):**
- Per-head multiplicative sigmoid gate on attention output
- Zero-init → identity at init (safe to stack)
- 12 × 8 heads × 11 layers = **1,056 new parameters**
- Implementation: `y * (2.0 * sigmoid(Linear(x[:,:,:12])))`
- Applied in standard, parallel-residual, and depth-recurrent paths
- <2% throughput cost

**SmearGate (@kellerjordan, modded-nanogpt):**
- Input-dependent per-channel residual mixer
- 13 parameters total
- Mixes current token with previous token (strictly causal)
- Zero-init lambda → identity start

**Casefold V4:** Pending organizer review #1604. If denied, kill.

**Multi-Phase Global SGD TTT:** 3 phases, score-first, identical to #1670.

**Stacking analysis:**
- AttnOutGate + SmearGate = 1,069 params, both zero-init, mechanism-diverse → safe to stack
- Both are "pure training-time architectural additions with trained weights" per PR text
- Neither depends on casefold

**Salvageable without casefold:**
- AttnOutGate (isolated impact ~-0.001 BPB)
- SmearGate (isolated impact ~-0.0005 BPB based on 13-param scale)
- Combined with non-casefold tokenizer: expected -0.001 to -0.002 BPB

---

## PR #1700 — SP8192 Multi-Phase Global SGD + Phased LoRA TTT (1.07219)

**Verdict: LEGAL, softmax, novel TTT mechanism.**

- No casefold (uses standard SP-8192)
- **Int7 embeddings** (novel — between int6 matrices and int8 embeddings)
- Per-layer GPTQ with sigma clipping
- **Multi-phase global SGD at test-time + phased LoRA TTT combined**
- Depth recurrence, VarLen flash attention, fused triton MLP
- 1.07219 BPB (3-seed mean)

**Novel TTT mechanism vs killed H91:**
- H91 killed: AdamW on all params during eval (all-param full adaptation) → 1.1757
- PR #1700: Multi-phase SGD on scored tokens ONLY + phased LoRA
- **Different mechanism entirely** — score-first + LoRA-restricted
- Per L39 (don't reopen H89/H91 without new mechanism): this IS a new mechanism

**Salvageable:**
- Int7 embeddings (save ~200 KB over int8, cost ~50 KB accuracy)
- Multi-phase global SGD (score-first + phased, legal under #1017)
- Phased LoRA TTT (only LoRA params update at eval, not full model)

---

## PR #1670 — Casefold V4 + Multi-Phase Global SGD TTT (1.05970)

**Verdict: Similar to #1693 but without AttnOutGate/SmearGate. Casefold-dependent.**

- Casefold V4 + Multi-Phase Global SGD TTT (3 phases, 2000 prefix docs)
- Adaptive GPTQ clip: MLP=12σ, ATTN=13σ (vs fixed σ)
- Builds on PR #1530 base

**Salvageable without casefold:**
- **Adaptive GPTQ clip per-component**: MLP=12σ, ATTN=13σ — novel per-component sigma tuning
- Multi-phase global SGD TTT (also in #1693, #1700)

---

## Cross-PR Kill-Family Audit

| Component | Kill Family Check | Verdict |
|---|---|---|
| GDN (FLA) | I + K (scorer artifacts) | BLOCKED |
| KV sharing stride=2 (CLA) | None | LEGAL (our NV-002b) |
| Legal score-first TTT | None | LEGAL (already in SOTA) |
| BigramHash d=32 | None (n-gram → bias, not as learned lookup) | LEGAL |
| Path A v3 passthrough quant | None (L14 OK — happens AFTER dequant of weights) | LEGAL |
| LZMA self-extracting wrapper | L11 (source already LZMA golfed) | CHECK — may be redundant |
| AttnOutGate | None | LEGAL |
| SmearGate | None | LEGAL |
| Casefold V4 | Pending Issue #1604 | CONDITIONAL |
| Int7 embeddings | None (different bit-width from killed codebooks) | LEGAL |
| Multi-phase SGD TTT | L35 (H91 killed) — but DIFFERENT mechanism | LEGAL (mechanism differ) |
| Phased LoRA TTT | L39 (don't reopen without new mech) — IS new | LEGAL |
| Adaptive GPTQ clip | L14 if pre-quant; OK if per-component sigma | CHECK |

---

## Measured Deltas vs 1.0810 Bigbag Baseline (canonical scorer)

| Component | Measured ΔBPB | Isolated? |
|---|---|---|
| BigramHash d=32 + Path A v3 quant (PR #1716) | **-0.00218** | Yes (cleanest isolated improvement) |
| AttnOutGate alone | Unknown (bundled) | No |
| SmearGate alone | Unknown (bundled) | No |
| Multi-phase SGD TTT vs standard TTT | Unknown (not isolated from casefold in public PRs) | No |
| Casefold alone | Unknown (bundled) | No |
| Adaptive GPTQ clip | Unknown (bundled) | No |
| CLA stride=2 on softmax (no GDN) | Unknown (only tested with GDN) | No |

**Critical gap:** Only PR #1716 provides an ISOLATED measured delta that passes our
canonical-scorer discipline. Everything else requires isolation.

---

## Implication for Our Campaign

1. **PR #1716 is the legal SOTA candidate (1.07882).** We should aim to beat this, not 1.0810.
2. **Casefold PRs (1693, 1670) are a coin flip** — if legality denied, their gains vanish.
3. **GDN PRs (1698) are likely poisoned** under canonical scorer.
4. **Stackable legal components we haven't used:**
   - AttnOutGate (1,056 params, zero-init)
   - SmearGate (13 params, zero-init)
   - BigramHash d=32
   - Path A v3 passthrough quantization
   - Int7 embeddings
   - Multi-phase SGD TTT
   - Phased LoRA TTT
   - Adaptive GPTQ clip per-component
5. **Our NV-002b CLA is validated** by PR #1698's K_KVShare_Wider — the arch works,
   the 1.01 score was the scorer bug.

**Updated target:** Beat PR #1716's 1.07882 by ≥0.029 BPB to reach <1.050.
