# PR Brief — Lock Benchmark to 1.03385760 and Carry Forward TMA G3

## Suggested Title

`orchestrator: lock 1.03385760 benchmark and promote TMA G3 as passed systems gate`

## Why

The active V7 planning docs were still anchored to the older `NS0` softmax control line at
`1.07914847 BPB`, even though the strongest verified local result in this repo is now the
Opensens reproduction of `K_KVShare_Wider FLA` at `1.03385760 BPB`.

At the same time, `TMA G3` is no longer hypothetical. The systems gate already passed on the
control line with:

- baseline mean step time: `415.3298 ms`
- TMA mean step time: `374.5369 ms`
- delta: `-9.8218%`

So the correct planning posture is:

1. beat `1.03385760`, not `1.07914847`
2. keep `NS0` and `NS1` as controls, not as the mainline target
3. treat `TMA G3` as a donor to transfer onto the benchmark family before recurrence isolates

## Evidence

- best verified benchmark:
  [parameter-golf/records/track_10min_16mb/2026-04-17_KKVShareWider_FLA_Opensens/README.md](/Users/applefamily/Desktop/Business/Opensens/03%20-%20R%26D%20Projects/Parameter%20golf/parameter-golf/records/track_10min_16mb/2026-04-17_KKVShareWider_FLA_Opensens/README.md:1)
- machine-readable benchmark:
  [submission.json](/Users/applefamily/Desktop/Business/Opensens/03%20-%20R%26D%20Projects/Parameter%20golf/parameter-golf/records/track_10min_16mb/2026-04-17_KKVShareWider_FLA_Opensens/submission.json:1)
- passed systems gate:
  [orchestrator/runs/RUN-20260417-091551-loop10-sub105-research/synthesis/TMA-G3-RESULT.md](/Users/applefamily/Desktop/Business/Opensens/03%20-%20R%26D%20Projects/Parameter%20golf/orchestrator/runs/RUN-20260417-091551-loop10-sub105-research/synthesis/TMA-G3-RESULT.md:1)

## What Changed

- [orchestrator/CAMPAIGN-SYNTHESIS-V7.md](/Users/applefamily/Desktop/Business/Opensens/03%20-%20R%26D%20Projects/Parameter%20golf/orchestrator/CAMPAIGN-SYNTHESIS-V7.md:18)
  now locks `1.03385760` as the benchmark and reframes `NS0` as historical control
- [orchestrator/parameter-golf-sub100-v7.md](/Users/applefamily/Desktop/Business/Opensens/03%20-%20R%26D%20Projects/Parameter%20golf/orchestrator/parameter-golf-sub100-v7.md:18)
  now routes Phase 0 and Phase 1 through the benchmark family first, then control-line checks
- [orchestrator/architecture-map.md](/Users/applefamily/Desktop/Business/Opensens/03%20-%20R%26D%20Projects/Parameter%20golf/orchestrator/architecture-map.md:3)
  now marks `K_KVShare_Wider` as `BENCHMARK` and `TMA Megakernel` as `PROMOTED (G3 passed)`

## Review Focus

- Does the repo now consistently distinguish:
  - benchmark line = `1.03385760`
  - control lines = `NS0`, `NS1`
  - systems donor = `TMA G3`
- Is the next experimental path correctly ordered as:
  1. exact-payload audit on benchmark family
  2. `TMA` transfer onto benchmark family
  3. one isolated recurrence or sharing delta

## Next Work After Merge

1. Audit `brotli-11` vs `rANS` vs block-adaptive Huffman on the exact `1.03385760` artifact family.
2. Port `TMA` onto the `K_KVShare_Wider` benchmark stack and confirm throughput behavior there.
3. If the transfer is clean, run one isolated follow-up:
   - `Loop45`, or
   - `DeltaShare`

## Suggested PR Body

This updates the active V7 planning docs to reflect the repo's current evidence.

- Locks the live benchmark to the verified `K_KVShare_Wider FLA` Opensens reproduction at `1.03385760 BPB`
- Reframes `NS0` and `NS1` as historical control lines rather than the mainline target
- Promotes `TMA G3` from a pending gate to a passed systems donor based on the recorded `-9.8218%` step-time improvement
- Reorders the next plan around benchmark-family payload audit, benchmark-family TMA transfer, and then one isolated architectural delta

This is a planning/docs correction. It does not claim new model results.
