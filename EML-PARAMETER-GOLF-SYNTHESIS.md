# EML × Parameter Golf — Disruptive Angle Analysis

**Date:** 2026-04-16
**Source:** `eml_operator_research_memo.md` (EML: $\mathrm{EML}(x,y) = e^x - \ln y$, arXiv 2603.21852v2)
**Target stack:** SOTA 1.0810 BPB (2026-04-09), Loop 8 target 1.077 BPB, competition goal 1.050 BPB

---

## 1. TL;DR — Honest reading

The EML framework is a **representation** idea. Parameter golf has already solved the
representation problem at the architecture level (11L transformer + 3-layer recurrence +
SDClip + LegalTTT). Where EML can actually intersect with parameter golf reduces to
**three specific angles**, two of which will likely hit the existing lesson gates (L02–L05)
and one of which is genuinely novel but high-risk.

**Genuinely disruptive angle (one, not many):**
> Procedural weight generation — replace stored Int6 indices with a tiny symbolic program
> that computes weights deterministically at load time. The program is the artifact.

This is the only path where EML's "one operator, recursive composition" philosophy
attacks the dominant byte cost (~15.9 MB of Int6 weights) rather than the dead-cost
overhead (16 KB source, few-KB metadata). Risk: the EML paper's own finding that
exact recovery fails at depth 4–6 suggests a 17M-parameter target is out of reach.

---

## 2. Why most EML applications fail existing lesson gates

Before proposing anything, the EML framework must be filtered through L01–L11.

| EML idea | Naive appeal | Gate that kills it |
|---|---|---|
| EML-generated centroids as Lloyd-Max codebook substitute | "64 centroids from 4 params — kills L02 codebook overhead" | **L03/L05:** Distribution-matching centroids produce near-uniform index distribution → LZMA 1.19:1 → worse than SDClip's 1.88:1 |
| EML-parameterized quantization grid (shared across rows) | "One global EML tree instead of per-row scales" | **L04:** Global codebook + per-row normalization → uniform indices → same failure as H71 |
| EML as activation function replacing LeakyReLU² | "Universal operator → richer gradients" | **L04/L09:** The SOTA LeakyReLU(0.5)² is already the empirical winner; `exp(x)` blows up, `log(y)` singular — optimization-hostile exactly as the memo warns |
| EML-tree encoding of train_gpt.py source | "Collapse source to a smaller symbolic tree" | **L11:** Source already LZMA2-raw inside b85 — 0 bytes strippable before re-compression |
| Symbolic regression on per-layer Hessian trace to allocate bits | "Use EML tree to parameterize bit allocation" | **L02:** Per-row Lloyd-Max already fails because of codebook overhead; symbolic generator adds ~0 bytes but saves ~0 bytes too |

**Anything that touches MLP/attention weight quantization and goes through LZMA must
produce a Gaussian-center-peak index distribution.** EML-generated centroids are
distribution-matching by nature — same failure mode as H70/H71/H72/H78.

---

## 3. Where EML can actually apply — three real angles

### Angle A (LOW–MEDIUM reward, LOW risk): Procedural positional encoding via EML constant tower

**Hypothesis H-EML-A:** Replace RoPE's fixed-base (10000) sinusoidal positional encoding
with an EML-constant-tower encoding. Zero artifact cost; zero learnable parameters
for position.

**Mechanism:**
- EML(1, 1) = e − ln(1) = e ≈ 2.7183
- EML(EML(1,1), 1) = e^e ≈ 15.15
- EML(1, EML(1,1)) = e − 1 ≈ 1.7183
- A depth-6 tree yields ~2⁶ = 64 deterministic constants with irrational, non-periodic values
- Use these as the RoPE base frequencies (one per RoPE dim)

**What it saves:** 0 bytes directly (RoPE is already parameter-free). What it might
improve: BPB by providing richer, non-periodic position signals.

**Where it gates:** Must improve BPB by ≥ 0.0005 on RunPod. The memo's own caveat
("optimization-hostile") applies — but here we don't train the EML values; they're
fixed constants. Only the downstream model trains.

**Why it's not obviously killed by prior loops:** RoPE base is conventionally 10000.
The Loop-3 catalog did not propose evaluating alternate base schedules.

**Risk profile:**
- Low — zero byte cost, zero training change
- Kill criterion: BPB ≥ SOTA at 3-seed median
- Cost: single H100 run × 3 seeds ≈ 30 min

### Angle B (MEDIUM reward, MEDIUM risk): EML-tree-based tokenizer vocabulary generator

**Hypothesis H-EML-B:** Replace the trained SP8192 tokenizer with a **symbolic vocabulary
constructor** where each vocabulary entry is an EML composition over a tiny set of
byte seeds.

**Mechanism:**
- Train SP8192 normally (SOTA path)
- Post-training: for each of the 8192 token strings, find the shortest EML-tree
  representation over `{byte_literal_0..255}` primitives (the EML framework generalized
  to byte concatenation: `EML_BYTES(x, y) = concat(x, y[len(y)-len(x):])` or similar)
- Store only the EML-tree index per token — if median tree-depth is 4, each token
  needs ~20 bits → 8192 × 20 / 8 = 20 KB vs current ~50–80 KB SentencePiece model

**Saved bytes:** 30–60 KB per artifact.

**Where it gates:**
- Only wins if the median EML-tree depth is shallow (3–4). English tokens often have
  shared prefixes/suffixes → trees may compress. Rare tokens (hex, code tokens) may
  require depth 8+ → kill.
- Mac-Mini testable: take the real SOTA SP8192 vocab, brute-force EML-tree search
  per token, measure total encoded bytes.

**Risk profile:**
- Medium — the competition specifies tokenizer as "stored in the artifact", meaning
  this optimizes one of the smaller segments (vocab is ~50–80 KB, not 15 MB weights)
- Kill criterion: encoded vocab bytes ≥ current SP8192 model bytes
- Cost: ~2 hours Mac Mini (brute-force search)
- **Kill if:** tokens that benefit most from SP8192's merge-rule semantics cannot be
  expressed as shallow EML trees

### Angle C (HIGH reward, VERY HIGH risk): Procedural weight generation — the disruptive angle

**Hypothesis H-EML-C:** The model artifact is **a tiny EML-tree program plus a seed
parameter vector**. At load time, the program procedurally generates all ~17M model
weights.

**Mechanism:**
- Learn a "weight generator" G parameterized by θ (few KB)
- G: (layer_id, row_id, col_id) → float — the weight value at that position
- G is structured as a composition of stable EML variants (SEML = softplus-based to
  avoid exp blow-up and log singularity — directly from §7.3 of the memo)
- Training: instead of storing weights, train θ to match the full-precision weights
  of a teacher model (distillation)
- Artifact: ~100 KB (θ + program) instead of ~15.9 MB (Int6 weights)

**Theoretical max savings:** 15.9 MB → 100 KB = 99%+ artifact reduction = ~15 MB free
budget to add more capacity → conceivably 1.050 BPB or lower.

**What the memo itself says about this:** §4.1 (basin fragmentation), §4.5
(continuous-to-discrete mismatch), §2.4 (recovery fails at depth 4–6). A weight generator
for 17M values is radically deeper than the regime where EML-style recovery was
demonstrated.

**Mitigations to consider:**
- SEML (softplus-based, §8.A.1 of memo) instead of raw EML
- MDL/complexity regularization during training (§8.B.2)
- Curriculum over tree depth (§8.B.3) — start with a shallow generator, deepen over training
- Symbolic-snap-in-the-loop (§8.B.5) — alternate continuous optimization with discrete projection
- Hybrid approach: generator produces weight REFERENCES into a small shared bank;
  weights = bank[generator_output]; this constrains the output to a discrete codebook
  and removes the continuous-to-discrete mismatch

**Kill criteria (must all hold):**
- Generator + θ size < 1 MB
- Teacher-student MSE on weights < 2× SDClip reconstruction MSE
- Downstream BPB within 0.02 of teacher BPB

**Risk profile:**
- Very high. The EML paper's own §2.4 finding is the lurking wall.
- If it works even partially (say, compresses MLPs by 50% at equal quality), that's
  already a disruptive competitive advantage.
- Cost: H100 weeks of training experiments, not hours.

**This is the only EML angle where the memo's philosophy meets parameter golf's
core cost structure.** The other two angles attack <100 KB segments; this attacks
the 15 MB segment.

---

## 4. Concrete disruptive program — if you had 1 RunPod week

Rank-ordered by expected value vs cost:

### Step 1 (Day 1): H-EML-A (RoPE constant tower)
- Zero-risk, zero-byte experiment
- If it gains any BPB, cheap win; if not, rules out a direction
- 1×H100 × 3 seeds × 30 min

### Step 2 (Days 1–3): H-EML-C Phase 1 (hybrid weight-bank generator)
- Build the weakest version of H-EML-C: keep current Int6 weights, but replace the
  per-row SDClip SCALE (currently 1 fp16 per row = 2 bytes) with an EML-tree-generated
  scale from a shared θ.
- Saves: 67 KB (scales) → ~5 KB (tree + θ). Negligible saving but proves the
  training-time EML mechanism works.
- If this trains stably → Phase 2 (generate codebook too); if not → the memo's §4
  pathology is already biting.

### Step 3 (Days 3–5): H-EML-C Phase 2 (full procedural MLP weights)
- Only proceed if Phase 1 trained stably.
- Replace ONE MLP matrix (22 layers × MLP = pick one) with a procedural generator.
- If reconstruction MSE within 2× SDClip and downstream BPB within 0.005 of SOTA → scale up.

### Step 4 (Days 5–7): H-EML-B (tokenizer vocabulary EML encoding)
- Only if Steps 1–3 have freed budget. This recovers 30–60 KB which can be reinvested
  in more weights.

### Parallel (background): H-EML-C fallback
- If the paper's §4 pathology kills H-EML-C at Phase 2, pivot to:
  **EML-generated quantization CODEBOOK for the embedding matrix (H64 extension).**
  Replaces H64's 1 MB fp16-codebook with an EML-tree that generates 64 centroids from
  4 per-row params. Bytes saved: up to 960 KB beyond current H64.
  **WARNING:** This hits L03/L04 unless the EML-generated centroids are non-uniform
  in a way that preserves Gaussian-center-peak index distribution. Analyse centroid
  spacing before running.

---

## 5. What the EML memo correctly warns about that applies here

The memo's §4 (Mathematical issues during training) is essentially a pre-written kill
list for optimistic EML applications. Map it to parameter-golf's existing lessons:

| Memo concern | Parameter-golf analogue |
|---|---|
| §4.1 Basin fragmentation | L05 — quality improvements don't monetize if bytes don't shrink |
| §4.2 Symmetry, non-identifiability | L04 — multiple codebooks give similar MSE but uniform indices |
| §4.3 Stiff exp/log composition | Not yet encountered in parameter golf (we don't use exp/log in quant); real if EML enters the training loop |
| §4.4 Domain constraints | Relevant for SEML design — softplus variants needed |
| §4.5 Continuous-to-discrete mismatch | L03 — optimal quantizers produce near-uniform indices; symbolic snap post-training is the actual failure mode |
| §4.6 Poor asymptotic behaviour | L05 — distribution-matching is the trap |

The memo's §7.2 four properties (expressive, stable, domain-safe, snap-friendly) are
a reasonable design gate for any successor operator. For parameter golf specifically,
add a 5th property:

> **§7.2.5 Compression-aware** — the operator's output distribution, after per-row
> normalization and uniform-grid discretization, must have Shannon entropy ≤ 4.5
> bits/symbol so that LZMA/Brotli can achieve ≥ 1.80:1 compression on the index stream.

This is the real lesson from Loop 7. EML-successor design that ignores L05 will
produce theoretically beautiful operators that are practically byte-negative.

---

## 6. Is this actually disruptive?

Honest answer: **No, unless H-EML-C Phase 2 works.**

- H-EML-A (RoPE tower) is incremental — saves 0 bytes, maybe −0.001 BPB.
- H-EML-B (tokenizer EML) is incremental — saves 30–60 KB, probably fails on rare tokens.
- H-EML-C Phase 2 (procedural weight generation) is potentially revolutionary (could
  reach ≤ 1.050 BPB by freeing ~14 MB of budget), but the EML paper's own empirical
  finding suggests the optimization wall is at depth 4–6 — well below what 17M
  parameters would need.

The EML framework's real contribution to parameter golf is **a well-articulated prior
on what an ideal compression operator should look like**. Its five desiderata (expressive,
stable, domain-safe, snap-friendly, and — my addition — compression-aware) could guide
the design of custom quantization schemes in future loops, even if EML itself is not
adopted as-is.

---

## 7. Action items for the incoming agent

1. **Read** [HANDOFF-LOOP8-AGENT-BRIEF.md](HANDOFF-LOOP8-AGENT-BRIEF.md) first —
   understand the SOTA and the lesson gates.
2. **Do NOT** propose EML-as-codebook variants that don't first analyse the output
   centroid distribution. They will hit L03/L04/L05.
3. **Consider** H-EML-A as a cheap Loop-9 experiment after Loop-8 RunPod work.
4. **Flag** H-EML-C Phase 1 (hybrid weight-bank with EML-generated SDClip scales) as
   a plausible Loop-10 exploratory experiment. Tiny byte saving, but validates
   whether EML-style trainable operators work in this model's regime.
5. **Update** `lessons-learned.md` with L12 capturing §7.2.5 (compression-aware
   operator design) once H-EML-A is run.
6. **Add** hypotheses H-EML-A, H-EML-B, H-EML-C to the hypothesis-bank.md with
   status `mac_mini_queued` (A, B) and `runpod_parked` (C).

---

## 8. Bottom line

EML's conceptual core — "universal compression via a single repeated operator" — is
genuinely aligned with parameter golf's goal. But the ONE place where this alignment
could produce 10× artifact reduction (H-EML-C procedural weights) is exactly where
the EML paper's own empirical results say it fails.

If someone makes H-EML-C work at 17M parameters, that is the disruptive result. If it
doesn't, EML contributes to parameter golf as a **design prior**, not a direct win.

The current Loop-8 plan (H64 graft + H74 + H73 + H79) is the safer, higher-probability
path to 1.077 BPB. EML-inspired work should be staged in parallel as Loop-9/Loop-10
exploratory experiments, not substituted for the Loop-8 queue.
