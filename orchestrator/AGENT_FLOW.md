# Agent Flow Integration

This scaffold now supports `Agent Flow` as a visualization layer for the Claude Code orchestration loop.

Repository referenced by the integration:

- `https://github.com/patoles/agent-flow`

## What This Integration Does

This repo does not embed the Agent Flow extension itself. Instead, it adds a bridge that emits an Agent Flow-compatible JSONL stream from the orchestrator run directory.

That gives you two useful modes:

1. native Agent Flow session tracking for real Claude Code sessions in VS Code
2. orchestrator-level replay and live updates from this scaffold through a shared JSONL event log

The bridge is especially useful because this scaffold is file-first. It can visualize:

- phase transitions
- queue depth changes
- AL / RL / M / CAR dispatch and completion
- artifact creation in `outbox/`, `synthesis/`, `test-plans/`, and selected `test-results/` paths
- semantic CAR progress messages from `remote-mlx-monitor.json`, including remote phase changes, validation milestones, and final `val_bpb`

## Install Agent Flow

Follow the upstream extension instructions from the Agent Flow README. The key usage path is:

1. install the VS Code extension
2. open the Command Palette
3. run `Agent Flow: Open Agent Flow`

If Claude Code hooks are configured in your environment, Agent Flow can auto-detect live Claude sessions directly.

## Run The Orchestrator Bridge

From the project root:

```bash
RUN_ID="RUN-20260326-042326-scaffold-smoke"
python3 orchestrator/scripts/agent_flow_bridge.py watch --run "$RUN_ID"
```

That command prints the JSONL file path and then keeps appending orchestration events as the run evolves.

Default output path:

- `orchestrator/runs/$RUN_ID/logs/agent-flow-events.jsonl`

You can print that path from the control plane with:

```bash
python3 orchestrator/scripts/control_plane.py agent-flow-path --run "$RUN_ID"
```

And you can print a ready-to-paste VS Code settings snippet with:

```bash
python3 orchestrator/scripts/control_plane.py agent-flow-path --run "$RUN_ID" --json
```

Or let the control plane start the bridge for you:

```bash
python3 orchestrator/scripts/control_plane.py start-agent-flow --run "$RUN_ID"
```

Check or stop it later with:

```bash
python3 orchestrator/scripts/control_plane.py agent-flow-status --run "$RUN_ID"
python3 orchestrator/scripts/control_plane.py stop-agent-flow --run "$RUN_ID"
```

If you only want a one-time replay snapshot of the current run state:

```bash
python3 orchestrator/scripts/agent_flow_bridge.py watch --run "$RUN_ID" --once
```

If you want to clear the generated log and state and start fresh:

```bash
python3 orchestrator/scripts/agent_flow_bridge.py reset --run "$RUN_ID"
```

If you want to keep the current log but remove accidental exact duplicate events:

```bash
python3 orchestrator/scripts/agent_flow_bridge.py dedupe --run "$RUN_ID"
```

## Point Agent Flow At The JSONL File

In VS Code settings, set:

- `agentVisualizer.eventLogPath`

to the emitted file path, for example:

- `/Users/applefamily/Desktop/Business/Opensens/03 - R&D Projects/Parameter golf/orchestrator/runs/RUN-20260326-042326-scaffold-smoke/logs/agent-flow-events.jsonl`

After `seed-briefs`, the same value is also written to:

- `orchestrator/runs/$RUN_ID/logs/agent-flow-settings.json`

Then open Agent Flow and choose the JSONL log mode if needed.

## Recommended Workflow

1. start the bridge watcher
2. open the Agent Flow panel in VS Code
3. launch `AL`, `RL`, and `M` Claude Code sessions
4. let them write artifacts into the run outboxes
5. synthesize into CAR experiments
6. keep the bridge watcher running so the orchestration graph continues to update

## Practical Notes

- This bridge does not replace native Claude hook streaming. It complements it.
- The JSONL stream is orchestration-focused, not token-accurate.
- If Agent Flow is already auto-detecting your live Claude sessions, keep using that. The bridge is still useful for showing run-level coordination and artifact flow.
