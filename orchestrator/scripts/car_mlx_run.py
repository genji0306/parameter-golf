#!/usr/bin/env python3
"""
CAR MLX Runner — runs A/B experiments on Mac Mini M4
Execute with: ~/parameter-golf/.venv/bin/python3 car_mlx_run.py
"""
import json
import os
import subprocess
import sys
import time

VENV_PYTHON = os.path.expanduser("~/parameter-golf/.venv/bin/python3")
SCRIPT = os.path.expanduser("~/parameter-golf/train_gpt_mlx.py")
RESULTS_PATH = os.path.expanduser("~/parameter-golf/car_results.json")

BASE_ENV = {
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "TRAIN_SEQ_LEN": "1024",
    "ITERATIONS": "500",
    "VAL_LOSS_EVERY": "0",
    "VAL_BATCH_SIZE": "524288",
    "TRAIN_LOG_EVERY": "100",
    "TRAIN_BATCH_TOKENS": "8192",
    "GRAD_ACCUM_STEPS": "1",
    "MAX_WALLCLOCK_SECONDS": "3600",
    "MLX_EAGER_EVAL": "1",
    "MLX_MAX_MICROBATCH_TOKENS": "4096",
    "WARMDOWN_ITERS": "100",
    "WARMUP_STEPS": "10",
    "SEED": "1337",
}

EXPERIMENTS = [
    {
        "name": "baseline_9L_mlp2x",
        "desc": "Baseline: 9L/512d/MLP2x, relu^2",
        "env": {"NUM_LAYERS": "9", "MODEL_DIM": "512", "MLP_MULT": "2"},
        "patch": None,
    },
    {
        "name": "leaky_relu_05",
        "desc": "LeakyReLU(0.5)^2: 9L/512d/MLP2x",
        "env": {"NUM_LAYERS": "9", "MODEL_DIM": "512", "MLP_MULT": "2"},
        "patch": ("        x = nn.relu(self.fc(x))", "        x = nn.leaky_relu(self.fc(x), negative_slope=0.5)"),
    },
    {
        "name": "11L_mlp2x",
        "desc": "11 layers: 11L/512d/MLP2x",
        "env": {"NUM_LAYERS": "11", "MODEL_DIM": "512", "MLP_MULT": "2"},
        "patch": None,
    },
    {
        "name": "9L_mlp3x",
        "desc": "MLP 3x: 9L/512d/MLP3x",
        "env": {"NUM_LAYERS": "9", "MODEL_DIM": "512", "MLP_MULT": "3"},
        "patch": None,
    },
    {
        "name": "11L_mlp3x",
        "desc": "Stack A: 11L/512d/MLP3x",
        "env": {"NUM_LAYERS": "11", "MODEL_DIM": "512", "MLP_MULT": "3"},
        "patch": None,
    },
    {
        "name": "11L_mlp3x_leaky05",
        "desc": "Best combo: 11L/512d/MLP3x + LeakyReLU(0.5)^2",
        "env": {"NUM_LAYERS": "11", "MODEL_DIM": "512", "MLP_MULT": "3"},
        "patch": ("        x = nn.relu(self.fc(x))", "        x = nn.leaky_relu(self.fc(x), negative_slope=0.5)"),
    },
]


def run_one(exp: dict) -> dict:
    name = exp["name"]
    print(f"\n{'='*60}")
    print(f"EXP: {name} — {exp['desc']}")
    print(f"{'='*60}")

    # Read original script
    with open(SCRIPT) as f:
        original = f.read()

    # Apply patch if needed
    if exp["patch"]:
        old, new = exp["patch"]
        patched = original.replace(old, new)
        if old not in original:
            print(f"  WARNING: patch target not found: {old}")
        else:
            with open(SCRIPT, "w") as f:
                f.write(patched)
            print(f"  Patched: {old[:40]}... -> {new[:40]}...")

    # Build environment
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(exp["env"])

    for k, v in exp["env"].items():
        print(f"  {k}={v}")

    # Run
    t0 = time.time()
    try:
        result = subprocess.run(
            [VENV_PYTHON, "-u", SCRIPT],
            cwd=os.path.expanduser("~/parameter-golf"),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        stdout = result.stdout
        stderr = result.stderr
        output = stdout + "\n" + stderr
    except subprocess.TimeoutExpired:
        output = "TIMEOUT after 600s"
        print(f"  TIMEOUT!")
    except Exception as e:
        output = f"ERROR: {e}"
        print(f"  ERROR: {e}")

    elapsed = time.time() - t0

    # Restore original script
    if exp["patch"]:
        with open(SCRIPT, "w") as f:
            f.write(original)
        print(f"  Reverted patch")

    # Parse results
    val_bpb = None
    val_loss = None
    steps = None
    step_avg = None

    for line in output.split("\n"):
        if "val_bpb:" in line:
            try:
                val_bpb = float(line.split("val_bpb:")[1].split()[0].strip())
            except:
                pass
        if "val_loss:" in line:
            try:
                val_loss = float(line.split("val_loss:")[1].split()[0].strip())
            except:
                pass
        if "step_avg:" in line:
            try:
                step_avg = float(line.split("step_avg:")[1].split("ms")[0].strip())
            except:
                pass
        if "step:" in line and "/" in line:
            try:
                steps = int(line.split("step:")[1].split("/")[0].strip())
            except:
                pass

    print(f"  val_bpb={val_bpb}  steps={steps}  ms/step={step_avg}  time={elapsed:.0f}s")

    if val_bpb is None:
        # Print last 10 lines for debug
        lines = output.strip().split("\n")
        print(f"  OUTPUT (last 10 lines):")
        for l in lines[-10:]:
            print(f"    {l}")

    return {
        "name": name,
        "description": exp["desc"],
        "val_bpb": val_bpb,
        "val_loss": val_loss,
        "steps": steps,
        "step_avg_ms": step_avg,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    print("=" * 60)
    print("CAR MLX Runner — Mac Mini M4 16GB")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"Steps per run: {BASE_ENV['ITERATIONS']}")
    print("=" * 60)

    results = []
    for exp in EXPERIMENTS:
        r = run_one(exp)
        results.append(r)

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump({"experiments": results}, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    valid = [r for r in results if r.get("val_bpb")]
    valid.sort(key=lambda r: r["val_bpb"])

    baseline = next((r for r in results if "baseline" in r["name"]), None)
    base_bpb = baseline["val_bpb"] if baseline and baseline.get("val_bpb") else None

    print(f"{'Rank':>4s} {'Name':36s} {'BPB':>8s} {'Delta':>8s} {'ms/step':>8s}")
    print("-" * 70)
    for i, r in enumerate(valid, 1):
        delta = f"{r['val_bpb'] - base_bpb:+.4f}" if base_bpb else "N/A"
        ms = f"{r.get('step_avg_ms', 0):.1f}" if r.get("step_avg_ms") else "?"
        print(f"{i:4d} {r['name']:36s} {r['val_bpb']:8.4f} {delta:>8s} {ms:>8s}")

    if valid and base_bpb:
        best = valid[0]
        print(f"\nBest: {best['name']} @ {best['val_bpb']:.4f} BPB")
        print(f"Improvement over baseline: {base_bpb - best['val_bpb']:+.4f} BPB")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
