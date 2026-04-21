# Campaign Synthesis V7

**Date:** 2026-04-20  
**Scope:** Parameter Golf sub-`1.03` / sub-`1.00` planning refresh around the current best verified benchmark  
**Detailed plan:** [parameter-golf-sub100-v7.md](./parameter-golf-sub100-v7.md)

## Summary

V7 changes the campaign in one important way: it stops treating compression, systems, and architecture as separate queues.

For Parameter Golf, artifact bytes are architectural budget. That means:

1. lossless compression headroom must be measured before more architecture spend,
2. deeper recurrence must be unlocked by throughput-neutral kernels before it is trusted,
3. cross-layer sharing should be revisited with low-rank compensation rather than only hard sharing,
4. pre-quant TTT remains legal-gated, but it is now strong enough to justify active preparation.

## Locked References

- **Current benchmark:** `K_KVShare_Wider FLA (Opensens reproduction)` at `1.03385760 BPB` (3-seed mean, no TTT)
- **Historical canonical control:** `NS0` at `1.07914847 BPB`
- **Passed systems gate:** `TMA G3` with `415.3298 ms -> 374.5369 ms` (`-9.8218%`)

Operational consequence:

- the benchmark to beat is now `1.03385760`, not `NS0`
- `NS0` and `NS1` stay useful as canonical controls and transfer surfaces
- `TMA` is no longer a hypothetical gate in V7; it already passed on the softmax control line

## What Changed

### 1. Compression moves to the front

`ANS` / entropy-coded packaging is now a Phase 0 requirement, not a late packaging nicety.

Reason:

- PR `#1510` suggests lossless artifact savings large enough to unlock otherwise-blocked variants.
- The tensor-compression paper supports byte-grouping plus entropy coding as a strong general direction, even if its checkpoint-delta framing is not directly transferable.

Operational effect:

- audit `brotli-11` vs `rANS` vs block-adaptive Huffman on the exact mixed-quant payload
- only trust gains after wrapper/code overhead is included

### 2. TMA is now a passed recurrence gate, not a pending one

`Loop45` is no longer treated as a naked architecture bet.

Reason:

- PR `#1555` changes the recurrence question into a systems question
- PR `#1736` shows a strong number, but its loop contribution is still confounded

Operational effect:

- no deep-loop escalation on a new stack without confirming the same throughput story there
- the softmax control line already passed `G3`
- next work is to carry the systems lesson onto the current `1.03385760` benchmark family, then isolate `Loop45` or another recurrence delta cleanly

### 3. New structural lane: DeltaShare

V7 adds a new hypothesis family:

- cross-layer sharing in middle blocks
- tiny low-rank deltas to restore flexibility
- keep edge/specialized blocks unique

Reason:

- DeltaLLM shows the right structural lesson even though its full teacher-distillation recipe is too heavy to import directly

Operational effect:

- create a Parameter Golf-native sharing lane instead of treating hard sharing as all-or-nothing

### 4. PRC-009 moves from passive watchlist to active prep

Reason:

- there are now two independent public signals pointing to pre-quant TTT

Operational effect:

- legal gate remains
- implementation prep, compliance review, and budget math move earlier

## V7 Queue

1. Lock `1.03385760` as the benchmark and compare every fresh lane against it.
2. Keep `NS0` and `NS1` alive as canonical control lines, not as the headline target.
3. Run exact-payload `ANS` packaging audit on the current benchmark artifact family first.
4. Reuse the passed `TMA G3` result as a systems donor and port that throughput gain onto the benchmark family.
5. Only after that, isolate `Loop45` or another recurrence delta on a TMA-safe path.
6. Run `NV-015 DeltaShare` as the new v7-specific architecture lane.
7. Keep `NV-002b` alive as a bridge lane, not the automatic main bet.
8. Keep `PRC-009` warm and ready while legal status is unresolved.

## Net Effect

V7 is stricter than v6.

It narrows the main campaign to:

- one locked benchmark family at `1.03385760`,
- canonical softmax controls retained for hygiene and transfer checks,
- earlier compression verification,
- systems-backed deep recurrence only,
- one new sharing-aware architecture lane,
- legal-conditional pre-quant TTT kept ready but not assumed legal.

That is the cleanest update consistent with the current PR scan, the compression literature, the passed `TMA G3` result, and the repo’s existing evidence hygiene rules.
