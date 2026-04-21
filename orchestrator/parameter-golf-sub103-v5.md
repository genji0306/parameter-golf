---
id: pgolf-sub103-v5
title: "Parameter Golf Sub-1.03 - Canonical Softmax Reinvestment Campaign"
author: codex-autoresearch
intent: research
mode: staged
budget_usd: 120.0
priority: critical
tags: [parameter-golf, sub-103, canonical-scorer, cla, qlora, softmax]
deadline: "2026-04-30T23:59:59Z"
---

# Objective

Deliver a scorer-clean plan for reaching `< 1.030 BPB` under the canonical scorer by building on the legal softmax base (`PR #1716` and `PR #1700`) and then validating a narrow set of surviving novel lanes. The plan explicitly excludes the invalidated or killed families: `H85/H96` GDN, `H89`, `H91`, `BQ1`, and the now-closed `H87`.

# Strategy

Use one large swing plus orthogonal add-ons:

- Main swing: `NV-002b` CLA / cross-layer KV sharing on the canonical softmax base.
- Adaptation lane: `NV-004` plus `PRC-006`, with `PRC-005` allowed only when the base is frozen.
- Throughput lane: `NV-003` selective short-window on non-recurrence layers.
- Recurrence micro-lane: `NV-005`, `H88`, `PRC-003`, and `PRC-004`.
- Final budget layer: `H98` plus `PRC-008` after post-rotation sigma retuning.

# Steps

1. Reproduce `PR #1716` locally as NS0 with the canonical scorer and retain it only if seed-42 is `<= 1.0800`, artifact is `<= 16,000,000` bytes, train is `<= 600s`, and eval is `<= 600s`.
2. Import `PR #1700` as NS1 on top of the NS0 base using `PRC-005` and `PRC-006`. Retain it only if seed-42 is `<= 1.0740` with zero scorer drift and the same artifact / train / eval caps.
3. Run parallel `MLX` isolates for `NV-002b`, `NV-003`, the `NV-004` STE-grid parity check, and the `NV-005` / `H88` / `PRC-003` / `PRC-004` bundle.
4. Promote `NV-004` only if its STE Int6 grid exactly matches GPTQ Int6. If that check fails, kill the lane instead of patching around it.
5. Launch a `RunPod` isolated validation for `NV-002b`. Keep it only if seed-42 reaches `<= 1.0580` and there is no temporal KV sharing.
6. Launch exactly one secondary `RunPod` isolate from the surviving `NV-003`, `NV-004`, or `NV-005/H88/PRC-003/PRC-004` family. Keep it only if the isolated seed-42 result improves the parent retained score by at least `0.0020` BPB while staying within artifact, train, and eval limits.
7. Compound only measured winners. Start with `NV-002b + NV-003`, then `NV-002b + NV-004 (+ PRC-006 with frozen base)`, then `NV-002b + H88/NV-005/PRC-003/PRC-004`.
8. Do not compound `NV-004` with unfrozen `PRC-005`. Do not combine `H98` with `PRC-008` until sigma is re-tuned after rotation.
9. Promote a compound candidate to three seeds only if seed-42 reaches `<= 1.0520`.
10. Add `H98` and `PRC-008` only after the architecture and compound stack already work. Keep the same Hadamard basis across any CLA sharing group.
11. Run the final three-seed canonical scorer validation and retain the stack only if the mean is `< 1.030`, artifact is `<= 16,000,000`, train is `<= 600s`, and eval is `<= 600s`.
12. Package the final `submission` artifact and evidence chain without reopening any killed lane for rescue deltas.

# Constraints

- Canonical scorer only. All BPB numbers must use the repo-root `build_sentencepiece_luts` semantics.
- `PRC-001` and `PRC-002` are part of the trust base, not optional garnish.
- Do not reopen `H85`, `H96`, `H89`, `H91`, `BQ1`, or `H87`.
- Do not shrink below the `11L/512d` family except inside the explicit `NV-002b` sharing plan.
- No compound before isolated single-change validation.
- No additive BPB projection is accepted as proof.
- `NV-004` is invalid if STE Int6 and GPTQ Int6 disagree.
- `NV-004 x PRC-005` is forbidden unless the base is frozen.
- `PRC-003` remains fp16.
- `H98 x PRC-008` requires post-rotation sigma re-tuning.
- Artifact must stay `<= 16,000,000` bytes under the submission compressor.
- Train and eval must each stay `<= 600s`.
- Final decision uses seeds `{42, 314, 999}`.

# Success Criteria

- Primary: final three-seed mean `< 1.030 BPB` under the canonical scorer.
- Trust milestone: NS0 reproduces `PR #1716` locally and NS1 reproduces the `PR #1700` line without scorer drift.
- Architecture milestone: `NV-002b` reaches seed-42 `<= 1.0580`.
- Compound milestone: one measured compound reaches seed-42 `<= 1.0520`.
- Process milestone: every retained lane is scorer-clean, isolated before compound use, and documented against the closed-lane list.

# Budget

- NS0 and NS1 trust runs: about `$16`.
- Parallel `MLX` isolates: `$0`.
- Two isolated `RunPod` validations beyond NS1: about `$30-45`.
- Two compound `RunPod` validations: about `$30`.
- Final three-seed validation and packaging: about `$15-20`.
- Total working budget: about `$90-111`, with a `budget_usd` cap of `$120` for re-runs or one failed isolate.
