#!/bin/bash
# CAR (Controlled Autoresearch Runner) for Mac Mini M4 MLX
# Runs A/B experiments: baseline → change → measure delta
#
# Usage: bash car_mlx_runner.sh
#
# Each experiment runs for STEPS iterations, measures val_bpb, compares to baseline.
# Results logged to ~/parameter-golf/car_results.json

set -euo pipefail

cd ~/parameter-golf
PYTHON=".venv/bin/python3"
SCRIPT="train_gpt_mlx.py"
RESULTS="car_results.json"
STEPS="${CAR_STEPS:-1500}"
SEQ_LEN="${CAR_SEQ_LEN:-1024}"

# Common env for all runs
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export TRAIN_SEQ_LEN="$SEQ_LEN"
export ITERATIONS="$STEPS"
export VAL_LOSS_EVERY="$STEPS"          # validate only at end
export TRAIN_LOG_EVERY="100"
export TRAIN_BATCH_TOKENS="524288"
export GRAD_ACCUM_STEPS="8"
export MAX_WALLCLOCK_SECONDS="9999"     # don't time-cap, use step count
export MLX_EAGER_EVAL="1"              # keep memory low on 16GB
export WARMDOWN_ITERS="300"

echo "=================================================="
echo "CAR MLX Runner — Mac Mini M4 16GB"
echo "Steps: $STEPS | Seq: $SEQ_LEN"
echo "=================================================="

# Initialize results
echo '{"experiments":[]}' > "$RESULTS"

run_experiment() {
    local name="$1"
    local desc="$2"
    shift 2
    # remaining args are env overrides

    echo ""
    echo "── EXP: $name ──────────────────────────────"
    echo "  $desc"

    # Set experiment-specific env vars
    for envvar in "$@"; do
        export "$envvar"
        echo "  ENV: $envvar"
    done

    local seed=1337
    export SEED="$seed"

    # Run training
    local logfile="/tmp/car_${name}.log"
    local t0=$(date +%s)

    $PYTHON "$SCRIPT" > "$logfile" 2>&1 || true

    local t1=$(date +%s)
    local elapsed=$((t1 - t0))

    # Extract val_bpb from log
    local val_bpb=$(grep -o 'val_bpb:[0-9.]*' "$logfile" | tail -1 | cut -d: -f2)
    local val_loss=$(grep -o 'val_loss:[0-9.]*' "$logfile" | tail -1 | cut -d: -f2)
    local steps_done=$(grep -o 'step:[0-9]*/[0-9]*' "$logfile" | tail -1 | cut -d: -f2 | cut -d/ -f1)
    local step_avg=$(grep -o 'step_avg:[0-9.]*' "$logfile" | tail -1 | cut -d: -f2)

    echo "  val_bpb=$val_bpb  val_loss=$val_loss  steps=$steps_done  avg=${step_avg}ms  time=${elapsed}s"

    # Append to results JSON
    $PYTHON -c "
import json, sys
with open('$RESULTS') as f:
    data = json.load(f)
data['experiments'].append({
    'name': '$name',
    'description': '$desc',
    'val_bpb': float('${val_bpb:-0}') if '${val_bpb:-0}' != '' else None,
    'val_loss': float('${val_loss:-0}') if '${val_loss:-0}' != '' else None,
    'steps': int('${steps_done:-0}') if '${steps_done:-0}' != '' else None,
    'step_avg_ms': float('${step_avg:-0}') if '${step_avg:-0}' != '' else None,
    'elapsed_s': $elapsed,
    'seed': $seed,
})
with open('$RESULTS', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || echo "  WARNING: could not save result"

    # Unset experiment-specific env vars
    for envvar in "$@"; do
        local key=$(echo "$envvar" | cut -d= -f1)
        unset "$key" 2>/dev/null || true
    done
}

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Baseline (9L/512d/MLP2x, relu²)
# ══════════════════════════════════════════════════════════════════════
run_experiment "baseline_9L_512d_mlp2x" \
    "Baseline: 9L/512d/MLP2x, relu^2, warmdown=300" \
    "NUM_LAYERS=9" "MODEL_DIM=512" "MLP_MULT=2"

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: LeakyReLU(0.5)² — requires code change
# We patch the MLX script in-place, run, then revert
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "── Patching MLP for LeakyReLU(0.5)² ──"
cp "$SCRIPT" "${SCRIPT}.bak"

# Replace relu² with leaky_relu(0.5)²
$PYTHON -c "
import re
with open('$SCRIPT') as f:
    code = f.read()
# The MLP forward: x = nn.relu(self.fc(x)); return self.proj(x * x)
# Change to: x = nn.leaky_relu(self.fc(x), negative_slope=0.5); return self.proj(x * x)
code = code.replace(
    'x = nn.relu(self.fc(x))',
    'x = nn.leaky_relu(self.fc(x), negative_slope=0.5)'
)
with open('$SCRIPT', 'w') as f:
    f.write(code)
print('Patched: relu -> leaky_relu(0.5)')
"

run_experiment "leaky_relu_05_sq" \
    "LeakyReLU(0.5)^2: 9L/512d/MLP2x" \
    "NUM_LAYERS=9" "MODEL_DIM=512" "MLP_MULT=2"

# Revert
cp "${SCRIPT}.bak" "$SCRIPT"
echo "Reverted MLP patch"

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: 11 Layers (keep MLP2x to isolate effect)
# ══════════════════════════════════════════════════════════════════════
run_experiment "11L_512d_mlp2x" \
    "11 layers: 11L/512d/MLP2x, relu^2" \
    "NUM_LAYERS=11" "MODEL_DIM=512" "MLP_MULT=2"

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: MLP 3x expansion (keep 9L to isolate effect)
# ══════════════════════════════════════════════════════════════════════
run_experiment "9L_512d_mlp3x" \
    "MLP 3x: 9L/512d/MLP3x, relu^2" \
    "NUM_LAYERS=9" "MODEL_DIM=512" "MLP_MULT=3"

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: 11L + MLP3x (full Stack A architecture)
# ══════════════════════════════════════════════════════════════════════
run_experiment "11L_512d_mlp3x" \
    "Stack A arch: 11L/512d/MLP3x, relu^2" \
    "NUM_LAYERS=11" "MODEL_DIM=512" "MLP_MULT=3"

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: 11L + MLP3x + LeakyReLU(0.5)² (best combo)
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "── Patching MLP for LeakyReLU(0.5)² ──"
cp "$SCRIPT" "${SCRIPT}.bak"
$PYTHON -c "
with open('$SCRIPT') as f:
    code = f.read()
code = code.replace(
    'x = nn.relu(self.fc(x))',
    'x = nn.leaky_relu(self.fc(x), negative_slope=0.5)'
)
with open('$SCRIPT', 'w') as f:
    f.write(code)
print('Patched: relu -> leaky_relu(0.5)')
"

run_experiment "11L_512d_mlp3x_leaky05" \
    "Best combo: 11L/512d/MLP3x + LeakyReLU(0.5)^2" \
    "NUM_LAYERS=11" "MODEL_DIM=512" "MLP_MULT=3"

# Revert
cp "${SCRIPT}.bak" "$SCRIPT"
rm -f "${SCRIPT}.bak"
echo "Reverted MLP patch"

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: Warmdown 500 (longer warmdown to compare)
# ══════════════════════════════════════════════════════════════════════
run_experiment "11L_mlp3x_warmdown500" \
    "Longer warmdown: 11L/512d/MLP3x, warmdown=500" \
    "NUM_LAYERS=11" "MODEL_DIM=512" "MLP_MULT=3" "WARMDOWN_ITERS=500"

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=================================================="
echo "CAR RESULTS SUMMARY"
echo "=================================================="
$PYTHON -c "
import json
with open('$RESULTS') as f:
    data = json.load(f)

exps = data['experiments']
# Sort by val_bpb (best first)
valid = [e for e in exps if e.get('val_bpb')]
valid.sort(key=lambda e: e['val_bpb'])

baseline = next((e for e in exps if 'baseline' in e['name']), None)
base_bpb = baseline['val_bpb'] if baseline and baseline.get('val_bpb') else None

print(f\"{'Rank':>4s} {'Name':36s} {'BPB':>8s} {'Delta':>8s} {'Steps':>6s} {'ms/step':>8s}\")
print('-' * 78)
for i, e in enumerate(valid, 1):
    delta = f\"{e['val_bpb'] - base_bpb:+.4f}\" if base_bpb else 'N/A'
    steps = str(e.get('steps', '?'))
    ms = f\"{e.get('step_avg_ms', 0):.1f}\"
    print(f\"{i:4d} {e['name']:36s} {e['val_bpb']:8.4f} {delta:>8s} {steps:>6s} {ms:>8s}\")

if valid:
    best = valid[0]
    print(f\"\nBest: {best['name']} @ {best['val_bpb']:.4f} BPB\")
    if base_bpb:
        print(f\"Improvement over baseline: {base_bpb - best['val_bpb']:+.4f} BPB\")
"

echo ""
echo "Full results: ~/parameter-golf/$RESULTS"
