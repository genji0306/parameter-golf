# Campaign Synthesis V5 - Sub-1.03 Canonical Reinvestment Plan

**Date:** 2026-04-19
**Campaign ID:** `pgolf-sub103-v5`
**Status:** Locked planning synthesis
**Target:** `< 1.030 BPB` under the canonical scorer
**Live canonical SOTA:** `1.07882` BPB (`PR #1716`, 2026-04-18)
**Canonical challenger:** `1.07219` BPB (`PR #1700`)

## Headline

Sub-1.03 is only credible if the campaign stops chasing scorer-poisoned or low-capacity lanes and instead uses a scorer-clean softmax spine: trust `PR #1716` locally, import the legal `PR #1700` phased TTT path, then push one large novel architecture swing (`NV-002b`) plus only orthogonal, measured add-ons (`NV-003`, `NV-004`, `NV-005`, `H88`, `PRC-003`, `PRC-004`, `H98`, `PRC-008`). The plan is intentionally narrower than v4: no additive projection is treated as evidence, no compound is allowed before isolated wins, and the closed lanes remain closed.

## Baseline

- Canonical byte accounting is non-negotiable. All retained BPB numbers must follow the repo-root `build_sentencepiece_luts` semantics: skip `sp.is_control`, `sp.is_unknown`, and `sp.is_unused`; count `sp.is_byte()` as exactly `1` byte; strip the leading `▁`; and avoid the non-canonical `+1` leading-space inflation that killed the GDN lane.
- The current trust anchors are `1.07882` BPB from `PR #1716` and `1.07219` BPB from `PR #1700`. The older `1.0810` bigbag score remains historical context, not the live target.
- `H85`, `H96`, `H89`, `H91`, and `BQ1` are closed. `H85/H96` are scorer-contaminated GDN lanes; `H89` and `H91` were empirically killed; `BQ1` proved that shrinking below the `11L/512d` family is a dead end. `H87` is also closed after its `1.08395` seed-42 miss, `16,050,968` B artifact, and throughput regression.
- The campaign therefore starts from a legal softmax family only. `PRC-001` and `PRC-002` are the measured byte-budget base from `PR #1716`; `PRC-005` and `PRC-006` are the only legal TTT import path from `PR #1700`.

## Novel Lanes

| Lane | Why it survives | Role in sub-1.03 plan | Kill / keep rule |
|---|---|---|---|
| `NV-002b` CLA softmax spine | Only surviving swing with expected magnitude large enough to matter; frees about `1.3M` params without inheriting GDN scorer poison | Primary architecture bet and reinvestment spine | Keep only if sharing stays spatial, not temporal, and the retained seed-42 result reaches `<= 1.0580` before compounds |
| `NV-004` QA-LoRA + `PRC-006` phased LoRA TTT | Captures the H87 adaptation idea without the `~3.5 MB` merged-LoRA tax | Main adaptation lane on top of the trusted softmax base | Kill immediately if STE Int6 does not exactly match GPTQ Int6, or if `PRC-005` touches the base without freezing it |
| `NV-003` selective short-window | Escapes the `BQ1` failure because it keeps the `11L/512d` family intact and buys wallclock | Throughput-positive secondary lane for extra training headroom | Keep only if it improves BPB while preserving long-range behavior and budget gates |
| `NV-005` + `H88` + `PRC-003` + `PRC-004` | Tiny, orthogonal recurrence-differentiation adds; `PRC-003` / `PRC-004` are already documented as stack-safe with care | Micro-gain lane that can pair with the main swing after isolated validation | Keep only measured winners; `PRC-003` remains fp16 because the sigmoid gate is quantization-sensitive |
| `H98` + `PRC-008` | Known quantization stacker, but only after the architecture is already working | Final budget-recycling / submission layer | Retune sigma after rotation and keep the same Hadamard basis across any CLA sharing group |

The plan is novel because it is not a broad "try everything" stack. It is a reinvestment doctrine: `PRC-001` + `PRC-002` establish the legal byte budget, `PRC-005` + `PRC-006` establish the legal TTT ceiling, `NV-002b` carries the main structural delta, and only then do `NV-003`, `NV-004`, and `NV-005` compete for the remaining headroom.

## Execution Order

1. `MLX` / local trust preflight: confirm the canonical scorer path and restate the closed-lane policy in every experiment brief.
2. `RunPod` NS0: reproduce `PR #1716` locally with `PRC-001` + `PRC-002` as the trusted softmax base.
3. `RunPod` NS1: import `PRC-005` + `PRC-006` onto the NS0 base and verify the `PR #1700` line without scorer drift.
4. `MLX` screening in parallel: isolate `NV-002b`, `NV-003`, the `NV-004` STE-grid precheck, and the `NV-005` / `H88` / `PRC-003` / `PRC-004` recurrence bundle.
5. `RunPod` isolated validation: first validate `NV-002b`; only validate a secondary lane if it beats its parent retained score cleanly.
6. `compound` phase: pair the retained `NV-002b` spine with exactly one validated secondary family at a time. Recommended order: `NV-002b + NV-003`, then `NV-002b + NV-004 (+ PRC-006 with frozen base)`, then `NV-002b + H88/NV-005/PRC-003/PRC-004`.
7. `RunPod` final stack: add `H98` + `PRC-008` only after post-rotation sigma retuning.
8. `submission` phase: run the final three-seed canonical scorer, byte-budget audit, and packaging pass. No killed lane is reopened to rescue a near-miss.

## Gates

| Gate | Requirement |
|---|---|
| `G0 scorer` | Use the canonical scorer only. Any non-canonical LUT path is an automatic kill. |
| `G1 NS0 trust` | Seed-42 `<= 1.0800`, artifact `<= 16,000,000` bytes, train `<= 600s`, eval `<= 600s`. |
| `G2 NS1 trust` | Seed-42 `<= 1.0740`, same artifact / train / eval caps, zero scorer deviation from the NS0 canonical path. |
| `G3 main swing` | `NV-002b` must hit seed-42 `<= 1.0580` before any compound lane is allowed. |
| `G4 secondary lane` | `NV-003`, `NV-004`, `NV-005`, `H88`, `PRC-003`, and `PRC-004` survive only if the isolated seed-42 result improves the parent retained score by at least `0.0020` BPB while keeping artifact, train, and eval inside budget. |
| `G5 compound gate` | Any compound candidate must reach seed-42 `<= 1.0520` before it earns a three-seed run. |
| `G6 final submission` | Three-seed mean `< 1.030`, canonical scorer parity preserved, artifact `<= 16,000,000`, train `<= 600s`, eval `<= 600s`. |

The numerical gates are strict on purpose. If `NV-002b` cannot cross `1.0580`, the repo evidence does not support a scorer-clean path below `1.03` without reopening dead lanes.

## Risks

- `NV-002b` is the only lane with enough expected magnitude to carry the plan, but its support is indirect because the public CLA evidence arrived inside scorer-poisoned GDN PRs.
- `NV-004` fails immediately if the STE grid diverges from GPTQ or if base-updating `PRC-005` is mixed with it without freezing the base.
- `NV-003` can still erase long-range context if the window is too tight even though it is not in the `BQ1` kill family.
- `PRC-003` is small but fragile: keep it fp16, do not quantize the sigmoid gate, and do not assume it compounds cleanly without measurement.
- `H98` plus `PRC-008` is only safe after sigma retuning, and CLA sharing groups must keep a consistent rotation basis.

## Submission Path

The retained submission path is:

1. Establish the local trust base with `PRC-001` + `PRC-002` (`PR #1716` repro).
2. Import the legal TTT ceiling with `PRC-005` + `PRC-006` (`PR #1700` repro).
3. Promote `NV-002b` as the main sub-1.03 swing only if it clears `G3`.
4. Add at most one retained secondary family at a time from `NV-003`, `NV-004`, or `NV-005/H88/PRC-003/PRC-004`.
5. Finish with `H98` + `PRC-008` only after their interaction gate is satisfied.
6. Package the final artifact under the canonical scorer, bit-shuffle + Brotli-11 byte budget, and the documented train / eval limits.

Anything that depends on reopening `H85`, `H96`, `H89`, `H91`, or `BQ1` is out of scope for this campaign. The sub-1.03 claim stands or falls on the canonical softmax reinvestment spine above.
