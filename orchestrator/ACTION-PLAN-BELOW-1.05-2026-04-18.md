# Action Plan To Reach Below 1.05 BPB

**Date:** 2026-04-18
**Workspace:** `/Users/applefamily/Desktop/Business/Opensens/03 - R&D Projects/Parameter golf`
**Ground truth baseline:** `1.07882 BPB` from PR `#1716`
**Target:** `< 1.050 BPB` under the canonical scorer

## Executive Decision

Do **not** run V4 as written.

V4 is useful as a hypothesis bank, but it still over-credits unisolated deltas and hides too much of the path inside speculative compounds.

The only solid path to `< 1.050` from the current evidence is:

1. lock onto the `#1716` canonical softmax stack as the new base
2. import the strongest clean eval-time delta from `#1700`
3. add only tiny recurrence-specialization changes first
4. hold one real architecture swing in reserve: **CLA on softmax**

## What Is Actually Solid

### Proven enough to build on

- **PR #1716**
  - canonical-valid
  - legal
  - measured at `1.07882`
  - gives us:
    - BigramHash `d=32`
    - Path A v3 passthrough quantization
    - strict artifact-size discipline

- **PR #1700**
  - canonical softmax
  - legal
  - gives us a distinct eval mechanism:
    - multi-phase global SGD
    - phased LoRA TTT

- **H88**
  - tiny parameter cost
  - clean to isolate
  - fits the exact SOTA family

### Not solid enough to lead

- **H96 / GDN**
  - blocked by scorer-trust issues
- **Casefold family (`#1670`, `#1693`)**
  - legality unresolved in issue `#1604`
- **H87 first**
  - artifact overhead too large for a size-tight winning line
- **CLA projected gains**
  - high upside, but still not measured on canonical softmax in this workspace

## Core Analysis of V4

### What V4 gets right

- `#1716` is the live clean baseline, not `1.0810`
- the GDN lane should not lead
- Path A v3, Bigram32, and `#1700`-style TTT are the best public softmax takeaways
- H87 storage cost is a real problem

### What V4 gets wrong

- it still assigns large expected deltas to ideas that are not isolated
- it still budgets as if several unproven compounds will all survive contact with reality
- it promotes CLA to top rank too early, despite no local canonical softmax measurement
- it spreads effort across too many MLX probes before securing the one line with the strongest public evidence

## The Actual Below-1.05 Plan

## Phase 0: Rebase The Campaign

**Objective:** stop using `1.0810` as the operational bar.

**Action**
- treat PR `#1716` at `1.07882` as the active legal SOTA reference
- keep canonical scorer discipline as non-negotiable

**Deliverable**
- queue and status updated to use `1.07882`

## Phase 1: Reproduce The Clean Base

**Experiment:** `NS0`

**Definition**
- exact `#1716`-style canonical stack only
- BigramHash `d=32`
- Path A v3 passthrough quantization
- no new recurrence changes
- no architecture change

**Purpose**
- establish a trusted local reproduction anchor
- eliminate implementation drift before stacking anything

**Gate**
- seed `42 <= 1.0800`
- artifact `<= 16,000,000`
- train `<= 600s`
- eval `<= 600s`

**If fail**
- stop all downstream work until the reproduction mismatch is understood

## Phase 2: Import The Strongest Clean Delta

**Experiment:** `NS1`

**Definition**
- `NS0` base
- replace plain score-first TTT with `#1700`-style multi-phase global SGD / phased TTT

**Why first**
- this is the best clean public delta source in the PR set
- it is larger and better evidenced than any tiny gate or ALiBi-style tweak

**Gate**
- seed `42 <= 1.0740`
- same byte and wallclock constraints

**Interpretation**
- if `NS1` cannot clearly beat `1.07882`, the public `#1700` gain is not transferring cleanly to the `#1716` stack
- in that case, do not keep layering eval complexity

## Phase 3: Add The Smallest Structural Delta

**Experiment:** `NS2`

**Definition**
- `NS1` winner
- add `H88` per-loop LN + loop embedding
- no LoRA
- no CLA yet

**Why**
- H88 is the cheapest recurrence-specialization probe
- unlike H87, it does not threaten the artifact budget

**Gate**
- seed `42 <= 1.0690`
- negligible throughput regression

**Interpretation**
- if H88 does not help on top of `NS1`, recurrence differentiation is weaker than hoped on this line

## Phase 4: Small Public Add-ons Only

**Experiment:** `NS3`

**Definition**
- `NS2` winner
- add only the tiny legal public extras from the PR set:
  - AttnOutGate
  - SmearGate

**Why**
- these are near-free
- they are already shown in public softmax records
- they are safer than H87 and cheaper than CLA

**Gate**
- seed `42 <= 1.0660`
- artifact headroom remains comfortable

**Interpretation**
- if NS1 + H88 + tiny gates still sits above `1.066`, the incremental-softmax route is probably too weak to close to `< 1.050` by itself

## Phase 5: The Only Real Architecture Swing

**Experiment:** `NS4`

**Definition**
- canonical softmax stack
- CLA / K_KVShare_Wider on softmax only
- retain Path A v3 discipline
- retain the best eval path from earlier phases

**Why this is the swing**
- among remaining ideas, CLA is the only one with enough theoretical headroom to matter
- unlike GDN, CLA itself is not tied to the scorer-bugged family

**Important constraint**
- do **not** run CLA first
- only run CLA after `NS1` and at least one tiny structural win or clean transfer failure

**Gate**
- seed `42 <= 1.0580`
- no scorer deviation
- train/eval/artifact budgets pass

**Interpretation**
- if CLA-softmax cannot land below roughly `1.058` on seed `42`, the `< 1.050` path is probably not alive in this campaign window

## Phase 6: Final Stack For The Actual Record Attempt

Only if earlier phases validate individually.

**Endgame stack**
- `#1716` canonical base
- Path A v3 passthrough quantization
- BigramHash `d=32`
- `#1700` multi-phase global SGD / phased TTT
- `H88` per-loop LN / loop embeddings
- AttnOutGate + SmearGate
- optional CLA-softmax if its isolated run validates

**Not included by default**
- H87
- casefold
- GDN
- AdamW TTT
- Late-SAM

## Why This Can Actually Reach Below 1.05

By itself, no single validated clean delta here gets us from `1.07882` to `< 1.050`.

So the plan must separate:

- **incremental line**:
  - `NS0 -> NS1 -> NS2 -> NS3`
- **architecture line**:
  - `NS4` CLA-softmax

The endgame is:

- if the incremental line reaches about `1.064-1.068`,
- then CLA-softmax becomes the step that can plausibly finish the job

Without CLA or another architecture-scale move, the current public softmax ideas are unlikely to close the full `0.029` gap cleanly.

## Ranked Queue

1. `NS0` — reproduce `#1716`
2. `NS1` — `#1716` base + `#1700` multi-phase/phased TTT
3. `NS2` — `NS1` + `H88`
4. `NS3` — `NS2` + AttnOutGate + SmearGate
5. `NS4` — CLA-softmax on the winning canonical stack
6. final 3-seed validation of the best stack under canonical scorer

## Hard Stop Rules

- if `NS0` does not reproduce, stop and fix calibration first
- if `NS1` does not beat `1.07882`, stop treating multi-phase TTT as the main unlock
- if `NS2` and `NS3` together cannot get below about `1.066`, stop pretending incremental softmax alone will hit `< 1.050`
- if `NS4` seed `42` cannot get below `1.058`, stop the current campaign and reassess

## What To Run Next

**Immediate next run**
- `NS0` if we do not yet have a local `#1716` reproduction
- otherwise `NS1`

**Immediate next document to add**
- a concrete `NS0` / `NS1` RunPod handoff plan

## Bottom Line

The solid plan is not:
- “probe 10 ideas and hope one compounds into 1.03”

The solid plan is:
- anchor on `#1716`
- import `#1700`
- add `H88`
- test tiny legal gates
- reserve CLA-softmax as the only architecture move with enough remaining headroom to plausibly finish below `1.05`
