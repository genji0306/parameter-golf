---
id: pgolf-sub100-v7
title: "Parameter Golf Sub-1.00 — V7 Compression-First, Systems-Unlocked, Sharing-Aware"
author: codex
intent: research
mode: sequential
budget_usd: 80.0
priority: critical
tags: [parameter-golf, sub-100, v7, ans, tma, loop45, delta-sharing, pre-quant-ttt]
deadline: "2026-04-30T23:59:59Z"
---

# Objective

Achieve `< 1.000 BPB` on Parameter Golf (`openai/parameter-golf`, `track_10min_16mb`).
Primary milestone: `< 1.030 BPB`.

Locked current benchmark:

- `KKVShare_Wider FLA (Opensens reproduction)`: `1.03385760 BPB` 3-seed mean, no TTT

Historical controls retained for transfer and hygiene:

- `NS0`: PR `#1716` style canonical softmax line
- `NS1`: PR `#1700` style legal multi-phase SGD + phased LoRA TTT on top of `NS0`

V7 changes the order of attack:

1. lock the current best verified result as the benchmark that all new work must beat
2. treat artifact compression headroom as a first-class unlocker
3. require a systems unlock before any deep recurrence escalation on a new stack
4. add a new structured sharing + low-rank delta lane inspired by DeltaLLM
5. move `NV-002b` from unquestioned main bet to a secondary structural lane

# What Changed From V6

## 1. ANS is now mandatory Phase 0 preflight

Reason:

- PR `#1510` suggests a lossless packaging win large enough to change what is even legal to try.
- The incremental-snapshot compression paper (`arXiv:2505.09810`) reinforces the byte-grouping + entropy-coding direction, even though its checkpoint-delta setting is not our deployment setting.

Action:

- Compare `brotli-11`, `rANS`, and block-adaptive Huffman-style coding on the exact current quantized payload.
- Do this before spending architectural budget.

## 2. TMA -> recurrence is now the main evidence-backed architecture branch

Reason:

- PR `#1555` makes deep recurrence a systems question, not just an architecture question.
- PR `#1736` shows `1.0655 BPB`, but the loop contribution is confounded with CaseOps.
- Therefore the right next move is not "assume Loop45 works"; it is "unlock throughput, then isolate Loop45 cleanly."

Action:

- The softmax control line already passed `G3` at `415.3298 ms -> 374.5369 ms` (`-9.8218%`).
- No `Loop45` or triple-loop spend on the benchmark family without reproducing that throughput story there first.

## 3. New lane: DeltaShare (cross-layer sharing + low-rank deltas)

Reason:

- DeltaLLM (`arXiv:2501.18596v1`) shows that cross-layer sharing with learned low-rank deltas can remove parameters while preserving most performance.
- The paper's full training recipe depends on teacher distillation and progressive module replacement, which is too heavy to import blindly into Parameter Golf.
- But its structural lesson is directly transferable: keep specialized edge blocks unique, share middle-layer weights, and restore flexibility with tiny low-rank deltas.

Action:

- Add an isolated Parameter Golf-specific sharing lane that does **not** depend on PMR or a teacher.

## 4. PRC-009 moves from passive watchlist to active prep

Reason:

- PR `#1735` was already strong.
- PR `#1487` is a second independent pre-quant TTT data point on the same mechanism family.

Action:

- Legal gating still holds.
- Implementation prep, compliance scaffolding, and cost accounting should start early instead of waiting for the final week.

# Source Transfer Verdict

## arXiv:2505.09810 — Lossless Compression for LLM Tensor Incremental Snapshots

Directly useful:

- byte-grouping / byte-shuffle remains a good prior
- adaptive entropy coding is worth testing on the exact mixed-quant payload
- blockwise Huffman / ANS style coding belongs in packaging preflight

Not directly useful:

- checkpoint-to-checkpoint delta compression
- training-time snapshot assumptions
- throughput claims measured on checkpoint streams rather than final challenge artifacts

V7 implication:

- import the compression lesson, not the incremental snapshot framing

## arXiv:2501.18596v1 — DeltaLLM

Directly useful:

- preserve the first and last specialized blocks
- share middle-layer weights only where redundancy is plausible
- attach tiny learned low-rank deltas to soften hard sharing
- treat quantization compatibility as a first-class check

Not directly adopted as-is:

- teacher distillation
- progressive module replacement as a mandatory recipe
- downstream benchmark numbers as if they were directly comparable to canonical FineWeb BPB

V7 implication:

- create a Parameter Golf-native `DeltaShare` lane, not a literal DeltaLLM port

## Awesome-LLM-Compression repo

Useful signal:

- compression work is clustering around four families relevant to us:
  - quantization
  - weight sharing / decomposition
  - low-rank compensation
  - vocabulary compression

Mostly irrelevant to Parameter Golf:

- KV-cache compression
- prompt compression
- latency-only decode tricks
- serving-only memory optimizations that do not reduce final artifact bytes or BPB

V7 implication:

- keep the frontier focused on artifact bytes, parameter sharing, and post-compression recovery

# Immutable Rules

1. Canonical scorer only. Any scorer deviation is invalid.
2. No compounding before isolated validation.
3. No additive BPB projections; only same-stack measurements.
4. Keep dead lanes dead: `GDN`, `H87`, `H89`, `H91`, `BQ1`, `NV-001`, `NV-002`.
5. `PRC-009` and `PRC-010` remain legal-gated.
6. No deep recurrence expansion beyond current trusted loop depth without a systems proof.
7. Compression wins count only if net bytes improve after wrapper/code overhead.
8. Final artifact must stay `<= 16,000,000` decimal bytes; train and eval must each stay `<= 600s`.

# New Gates

| Gate | Requirement | Effect if failed |
|---|---|---|
| `G0a` payload audit | exact round-trip + net positive byte gain on real quantized payload | keep Brotli path, no ANS promotion |
| `G0b` ANS unlock | at least one blocked architecture variant becomes size-legal | treat ANS as useful; otherwise compression stays optional |
| `G1` NS0 trust | seed-42 `<= 1.0800`, canonical scorer, within size/time budget | stop and fix control line |
| `G2` NS1 trust | seed-42 `<= 1.0740`, canonical scorer, within size/time budget | stop and fix TTT control import |
| `G3` TMA systems gate | fused-kernel path keeps step time within `+2%` of parent or improves it | no Loop45 / triple-loop on that stack |
| `G4` Loop45 isolate | seed-42 beats parent by `>= 0.006 BPB` and stays within budget | kill Loop45 lane |
| `G5` DeltaShare isolate | seed-42 beats parent by `>= 0.002 BPB` or frees enough bytes to fund a better same-stack reinvestment | kill DeltaShare lane |
| `G6` compound | seed-42 `<= 1.0520` before any 3-seed spend | stop compounding |
| `G7` sub-1.03 | 3-seed mean `< 1.030` with all budget gates met | submission go |

# Phase 0: Compression, Scorer, and Legal Preflight

## Step 0.1 — Keep NS0 trusted as a canonical control

Keep the V6 `NS0` requirement unchanged: canonical reproduction of the trusted softmax line first.

## Step 0.2 — Keep NS1 trusted as a canonical TTT control

Keep the V6 `NS1` requirement unchanged: import legal multi-phase SGD + phased LoRA TTT on top of `NS0`.

## Step 0.3 — Packaging audit on the exact payload

Run the packaging stack on the actual mixed-quant payload:

- `bit-shuffle + brotli-11`
- `bit-shuffle + rANS`
- `bit-shuffle + block-adaptive Huffman`
- keep `LZMA` on code wrapper only if still net-positive

Deliverables:

- exact round-trip proof
- payload-only bytes
- total submission bytes including code/wrapper
- unlock decision for blocked variants
- run this first on the `1.03385760` benchmark artifact family, then on softmax control artifacts only if a transfer check is needed

## Step 0.4 — Legal prep for PRC-009 / PRC-010

Do not spend RunPod on them yet, but do prepare:

- compliance checklist for Issue `#1017`
- code path inventory
- eval/train budget accounting
- artifact accounting for pre-quant and CaseOps branches

# Phase 1: Systems Unlock Before Deep Recurrence

## Step 1.1 — TMA Megakernel transfer onto the live benchmark family

The softmax control path already passed `G3`.

Test the fused FC + activation + gradient path on the `1.03385760` benchmark family next.

Goal:

- determine whether the current best family can absorb the same systems gain without losing its BPB edge

Non-goals:

- no BPB claims from TMA alone
- no compound with Loop45 in the same first pass

## Step 1.2 — Loop45 isolated only after TMA passes

If `G3` passes, run a clean seed-42 isolate:

- exact parent stack
- only the deeper recurrence change
- no CaseOps
- no pre-quant TTT
- no extra packaging changes beyond already-trusted Phase 0 compressor

Priority order:

- benchmark family first if the TMA transfer is clean
- softmax control line only as a sanity cross-check

## Step 1.3 — Triple-loop fallback

If Loop45 is unstable or too risky, test triple-loop depth recurrence as the lower-complexity recurrence escalation.

# Phase 2: DeltaShare — New V7 Structural Lane

## Step 2.1 — MLP-only DeltaShare isolate

New hypothesis ID: `NV-015 DeltaShare`

Design:

- keep first two and last two blocks unique
- do not touch the scorer or tokenizer
- share selected middle-block MLP weights via anchors
- attach tiny rank-`4` or rank-`8` low-rank deltas per shared block
- no teacher, no PMR, no separate distillation stage in the first isolate

Default first isolate:

- share MLP weights only on middle non-specialized blocks
- keep recurrence-critical blocks conservative unless the first result is positive

## Step 2.2 — DeltaShare on hard-sharing candidates

If `NV-015` wins in isolation:

- attach low-rank deltas to hard-sharing candidates already in the roadmap
- first target: CLA-style shared `K/V`
- second target: selected shared MLP lanes

This turns "hard share or not" into a softer continuum.

## Step 2.3 — Quantization-aware DeltaShare audit

Use the survey/repo lesson here:

- test whether the anchor weights and delta weights should quantize identically
- allow "anchor-skip" style exceptions only if total bytes still win

# Phase 3: Secondary Structural Frontier

## Step 3.1 — NV-002b becomes secondary, not primary

`NV-002b` stays alive, but it is no longer the uncontested main bet.

Run it after:

- Phase 0 payload headroom is known
- `NS1` is trusted
- the TMA/Loop45 branch is resolved enough to judge opportunity cost
- the first `DeltaShare` isolate is resolved

## Step 3.2 — Continue cheap local H88 / NV-005 / NV-003 probes

These remain valuable as low-cost, low-byte, orthogonal probes, but not as the campaign-defining bet.

# Phase 4: Legal-Conditional Adaptation Lane

## Step 4.1 — PRC-009 staged ladder

Do not jump straight to the expensive final form.

Use a two-step legal-conditional ladder:

1. `10ep` style budgeted reproduction path
2. `21ep` full PR `#1735` style path only if the ruling is favorable and the shorter path is already promising

## Step 4.2 — PRC-010 CaseOps remains legal-conditional

CaseOps still matters, but v7 treats its byte-sidecar legality as external to the core architecture plan.

# Phase 5: Controlled Compounds

Only compound after isolated wins from:

- one systems-backed recurrence lane (`Loop45` or triple-loop), or
- one `DeltaShare` lane, or
- one proven structural alternative (`NV-002b`)

Safe first compounds:

- `ANS` + anything, because packaging-only
- `DeltaShare` + trusted quant stack, if net bytes remain positive
- `Loop45` + trusted packaging, if throughput and artifact budgets remain valid

Unsafe early compounds:

- `Loop45` + `PRC-009`
- `Loop45` + `NV-002b`
- `DeltaShare` + `NV-002b`

Do not collapse multiple structural uncertainties into one first compound run.

# Revised Priority Queue

| Priority | ID | Why |
|---|---|---|
| 1 | `benchmark lock` | `1.03385760` is the live number to beat |
| 2 | `ANS` payload preflight | free multiplier if real on exact benchmark payload |
| 3 | `TMA` transfer onto benchmark family | systems donor already passed on control line |
| 4 | `Loop45` isolate | strongest public deep-loop signal once deconfounded |
| 5 | `PRC-009` prep | second independent public signal now exists |
| 6 | `NV-015 DeltaShare` | new structural lane supported by DeltaLLM |
| 7 | `NV-002b` | still attractive, no longer lone main bet |
| 8 | `H88` / `NV-005` / `NV-003` | cheap orthogonal locals |
| 9 | `PRC-010` | still large upside, still legal-conditional |

# V7 Decision Summary

V7 is stricter than V6 about what counts as actionable evidence:

- compression papers change packaging order, not model BPB directly
- DeltaLLM changes how we think about sharing, not which exact recipe we import
- the survey repo is useful as a taxonomy, not as evidence that every compression family belongs in Parameter Golf

So the V7 campaign is:

1. lock `1.03385760` as the benchmark
2. keep `NS0` and `NS1` trusted as control lines
3. measure real artifact headroom with `ANS`
4. transfer the passed `TMA G3` systems gain onto the benchmark family
5. isolate `Loop45`
6. add `DeltaShare` as the new v7-specific architecture lane
7. keep `PRC-009` warm and ready, but legal-gated

## PR Direction

The clean next PR path is:

1. benchmark-family payload audit
2. benchmark-family `TMA` transfer
3. one isolated recurrence or sharing delta on top of that line

That keeps the repo aligned with the strongest verified local result instead of reopening older `NS0`-anchored queues as the mainline.

If you only adopt one change from V7, it should be this:

**Stop treating architecture and compression as separate queues. For Parameter Golf, lossless artifact bytes are architectural budget.**
