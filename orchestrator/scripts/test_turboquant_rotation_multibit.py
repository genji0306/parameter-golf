#!/usr/bin/env python3
"""
EXP-H38-B: TurboQuant Rotation at int6/int5 Export Precision
=============================================================
Tests whether Hadamard rotation helps at lower bitwidths where quantization
error is much larger. The top solution trains with STE QAT at int6
(clip_range=31) but exports at int8. If we export at actual int6 or int5,
rotation may recover significant MSE.

Also tests on "QAT-shaped" weights: weights that have been trained with int6
STE quantization, making them already clustered near int6 grid points.

Run:  python3 test_turboquant_rotation_multibit.py
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# ── Architecture constants ───────────────────────────────────────────────────
D_MODEL = 512
MLP_DIM = 1536
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
NUM_LAYERS = 11
VOCAB_SIZE = 50304
CLIP_Q = 0.9999984

# ── Quantization configs ────────────────────────────────────────────────────
CONFIGS = {
    "int8":  {"clip_range": 127, "bits": 8},
    "int7":  {"clip_range": 63,  "bits": 7},
    "int6":  {"clip_range": 31,  "bits": 6},
    "int5":  {"clip_range": 15,  "bits": 5},
    "int4":  {"clip_range": 7,   "bits": 4},
}

# ── Walsh-Hadamard Transform ────────────────────────────────────────────────

def walsh_hadamard_transform(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    h = 1
    while h < d:
        x = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = np.stack([a + b, a - b], axis=-2)
        x = x.reshape(*x.shape[:-3], d)
        h <<= 1
    return x


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _rotation_signs(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 2.0 * rng.integers(0, 2, size=d).astype(np.float64) - 1.0


def rotate(t: np.ndarray, seed: int) -> np.ndarray:
    d = t.shape[-1]
    signs = _rotation_signs(d, seed)
    return walsh_hadamard_transform(t * signs) / np.sqrt(d)


def inv_rotate(t: np.ndarray, seed: int) -> np.ndarray:
    d = t.shape[-1]
    signs = _rotation_signs(d, seed)
    return signs * walsh_hadamard_transform(t) / np.sqrt(d)


# ── Parameterized Per-Row Quantization ───────────────────────────────────────

def quant_per_row(t: np.ndarray, clip_range: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize to [-clip_range, clip_range] per row. Returns (dequantized, scale)."""
    clip_abs = np.percentile(np.abs(t), CLIP_Q * 100, axis=1)
    clipped = np.clip(t, -clip_abs[:, None], clip_abs[:, None])
    scale = np.maximum(clip_abs / clip_range, 1.0 / clip_range)
    q = np.clip(np.round(clipped / scale[:, None]), -clip_range, clip_range)
    dequant = q * scale[:, None]
    return dequant, scale


def quant_with_rotation(W: np.ndarray, clip_range: int, seed: int) -> Tuple[np.ndarray, bool]:
    """MSE-adaptive: try rotation, keep if better. Returns (dequantized, used_rotation)."""
    dq_base, _ = quant_per_row(W, clip_range)
    mse_base = np.mean((W - dq_base) ** 2)

    if not _is_pow2(W.shape[-1]):
        return dq_base, False

    W_rot = rotate(W, seed)
    dq_rot, _ = quant_per_row(W_rot, clip_range)
    W_hat = inv_rotate(dq_rot, seed)
    mse_rot = np.mean((W - W_hat) ** 2)

    if mse_rot < mse_base:
        return W_hat, True
    return dq_base, False


# ── Weight Generators ────────────────────────────────────────────────────────

def generate_gaussian_weights(rows: int, cols: int, rng: np.random.Generator,
                              outlier_col_frac: float = 1/64,
                              outlier_row_frac: float = 1/32) -> np.ndarray:
    """Gaussian weights with outlier rows and columns (realistic transformer)."""
    W = rng.normal(0, 0.02, (rows, cols))
    n_out_cols = max(1, int(cols * outlier_col_frac))
    n_out_rows = max(1, int(rows * outlier_row_frac))
    out_cols = rng.choice(cols, n_out_cols, replace=False)
    out_rows = rng.choice(rows, n_out_rows, replace=False)
    W[:, out_cols] *= rng.uniform(3.0, 8.0, size=n_out_cols)
    W[out_rows, :] *= rng.uniform(2.0, 5.0, size=(n_out_rows, 1))
    return W


def generate_qat_shaped_weights(rows: int, cols: int, rng: np.random.Generator,
                                qat_clip_range: int = 31) -> np.ndarray:
    """Simulate weights after int6 STE QAT training.

    These cluster near int6 grid points but with small fp32 residual noise
    (the STE gradient approximation leaves weights near but not exactly on grid).
    """
    # Start from Gaussian
    W = rng.normal(0, 0.02, (rows, cols))
    # Add outlier structure
    out_cols = rng.choice(cols, max(1, cols // 64), replace=False)
    W[:, out_cols] *= rng.uniform(3.0, 6.0, size=len(out_cols))
    # Simulate QAT: snap to grid + small noise (STE residual)
    row_max = np.abs(W).max(axis=1)
    scale = np.maximum(row_max / qat_clip_range, 1.0 / qat_clip_range)
    W_quantized = np.clip(np.round(W / scale[:, None]), -qat_clip_range, qat_clip_range) * scale[:, None]
    # Add small STE residual noise (weights drift slightly off grid between QAT updates)
    ste_noise_scale = 0.1 * scale  # 10% of one quantization step
    W_qat = W_quantized + rng.normal(0, 1, W.shape) * ste_noise_scale[:, None] / qat_clip_range
    return W_qat


# ── Benchmark ────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.ravel(), b.ravel()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(np.dot(a_flat, b_flat) / denom) if denom > 0 else 1.0


def benchmark_config(
    config_name: str,
    clip_range: int,
    weights: Dict[str, np.ndarray],
    label: str,
) -> dict:
    """Benchmark one quantization config across all weight tensors."""
    total_mse_base = 0.0
    total_mse_best = 0.0
    total_params = 0
    n_rotated = 0
    n_total = 0
    t_base = 0.0
    t_rot = 0.0

    per_tensor = []

    for name, W in sorted(weights.items()):
        seed = 42 ^ (hash(name) & 0xFFFFFFFF)
        n = W.size
        n_total += 1
        total_params += n

        # Baseline
        t0 = time.perf_counter()
        dq_base, _ = quant_per_row(W, clip_range)
        t_base += time.perf_counter() - t0
        mse_base = float(np.mean((W - dq_base) ** 2))

        # Rotated
        t0 = time.perf_counter()
        dq_best, used = quant_with_rotation(W, clip_range, seed)
        t_rot += time.perf_counter() - t0
        mse_best = float(np.mean((W - dq_best) ** 2))

        if used:
            n_rotated += 1

        total_mse_base += mse_base * n
        total_mse_best += mse_best * n

        imp = (1 - mse_best / mse_base) * 100 if mse_base > 0 else 0
        per_tensor.append({
            "name": name,
            "shape": list(W.shape),
            "mse_base": mse_base,
            "mse_best": mse_best,
            "improvement_pct": imp,
            "rotated": used,
        })

    wmse_base = total_mse_base / total_params
    wmse_best = total_mse_best / total_params
    overall_imp = (1 - wmse_best / wmse_base) * 100 if wmse_base > 0 else 0

    return {
        "config": config_name,
        "clip_range": clip_range,
        "weight_type": label,
        "total_tensors": n_total,
        "total_params": total_params,
        "tensors_rotated": n_rotated,
        "weighted_mse_base": wmse_base,
        "weighted_mse_best": wmse_best,
        "overall_improvement_pct": overall_imp,
        "time_base_ms": t_base * 1000,
        "time_rot_ms": t_rot * 1000,
        "per_tensor": per_tensor,
    }


def generate_model_weights(rng: np.random.Generator, qat: bool = False) -> Dict[str, np.ndarray]:
    """Generate all weight tensors for 11L/512d model."""
    gen = generate_qat_shaped_weights if qat else generate_gaussian_weights
    weights = {}
    for i in range(NUM_LAYERS):
        weights[f"blocks.{i}.attn.q_proj"] = gen(D_MODEL, D_MODEL, rng)
        weights[f"blocks.{i}.attn.k_proj"] = gen(D_MODEL, HEAD_DIM * NUM_KV_HEADS, rng)
        weights[f"blocks.{i}.attn.v_proj"] = gen(D_MODEL, HEAD_DIM * NUM_KV_HEADS, rng)
        weights[f"blocks.{i}.attn.o_proj"] = gen(D_MODEL, D_MODEL, rng)
        weights[f"blocks.{i}.mlp.up"]      = gen(D_MODEL, MLP_DIM, rng)
        weights[f"blocks.{i}.mlp.down"]    = gen(MLP_DIM, D_MODEL, rng)
    weights["embed"]   = gen(VOCAB_SIZE, D_MODEL, rng)
    weights["lm_head"] = gen(VOCAB_SIZE, D_MODEL, rng)
    return weights


def estimate_artifact_size(weights: Dict[str, np.ndarray], bits: int) -> float:
    """Rough artifact size in MB at given bitwidth (ignoring compression)."""
    total_params = sum(w.size for w in weights.values())
    # bits per param + 16-bit scale per row
    total_rows = sum(w.shape[0] for w in weights.values())
    return (total_params * bits + total_rows * 16) / 8 / 1e6


def main():
    print("=" * 80)
    print("EXP-H38-B: TurboQuant Rotation at int6/int5 Export Precision")
    print("=" * 80)
    print(f"Architecture: {NUM_LAYERS}L / {D_MODEL}d / MLP {MLP_DIM}")
    print()

    rng = np.random.default_rng(42)

    all_results = []

    for weight_label, qat in [("gaussian_outliers", False), ("qat_shaped_int6", True)]:
        print(f"\n{'='*80}")
        print(f"Weight type: {weight_label}")
        print(f"{'='*80}")

        weights = generate_model_weights(rng, qat=qat)
        total_params = sum(w.size for w in weights.values())
        print(f"Total params: {total_params:,}")

        for bw_name in ["int8", "int7", "int6", "int5", "int4"]:
            cfg = CONFIGS[bw_name]
            est_mb = estimate_artifact_size(weights, cfg["bits"])

            result = benchmark_config(bw_name, cfg["clip_range"], weights, weight_label)
            all_results.append(result)

            # Summary line
            n_rot = result["tensors_rotated"]
            n_tot = result["total_tensors"]
            imp = result["overall_improvement_pct"]
            mse_b = result["weighted_mse_base"]
            mse_r = result["weighted_mse_best"]

            # Top 5 most improved tensors
            top5 = sorted(result["per_tensor"], key=lambda x: -x["improvement_pct"])[:5]

            print(f"\n  {bw_name} (clip_range={cfg['clip_range']}, ~{est_mb:.1f}MB raw):")
            print(f"    Weighted MSE:  base={mse_b:.10f}  best={mse_r:.10f}  delta={imp:+.2f}%")
            print(f"    Rotated: {n_rot}/{n_tot} tensors")
            print(f"    Time:  base={result['time_base_ms']:.0f}ms  rot={result['time_rot_ms']:.0f}ms")
            if top5 and top5[0]["improvement_pct"] > 0.01:
                print(f"    Top improved:")
                for t in top5:
                    if t["improvement_pct"] > 0.01:
                        tag = "ROT" if t["rotated"] else "---"
                        print(f"      {t['name']:40s} {t['improvement_pct']:+.2f}% [{tag}]")

    # ── Comparison matrix ────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("COMPARISON MATRIX: Rotation Improvement by Bitwidth x Weight Type")
    print(f"{'='*80}")
    print(f"{'Bitwidth':>10s} {'Gaussian+Outliers':>20s} {'QAT-Shaped (int6)':>20s}")
    print("-" * 55)

    for bw_name in ["int8", "int7", "int6", "int5", "int4"]:
        vals = []
        for wt in ["gaussian_outliers", "qat_shaped_int6"]:
            r = [x for x in all_results if x["config"] == bw_name and x["weight_type"] == wt]
            if r:
                vals.append(f"{r[0]['overall_improvement_pct']:+.2f}%")
            else:
                vals.append("N/A")
        print(f"{bw_name:>10s} {vals[0]:>20s} {vals[1]:>20s}")

    # ── Artifact size budget analysis ────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("ARTIFACT SIZE BUDGET: Params vs Bitwidth (16MB limit)")
    print(f"{'='*80}")
    weights_sample = generate_model_weights(rng, qat=False)
    base_params = sum(w.size for w in weights_sample.values())
    print(f"Current model: {base_params:,} params")
    for bw_name, cfg in CONFIGS.items():
        est_mb = estimate_artifact_size(weights_sample, cfg["bits"])
        headroom = 16.0 - est_mb
        extra_params = int(headroom * 1e6 * 8 / cfg["bits"]) if headroom > 0 else 0
        fit = "FITS" if est_mb <= 16.0 else "OVER"
        print(f"  {bw_name}: ~{est_mb:.1f}MB raw ({fit}), headroom: {headroom:+.1f}MB = ~{extra_params:,} extra params")

    # ── Export JSON ──────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "EXP-H38-B-results.json")
    # Strip per_tensor for compactness
    export = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != "per_tensor"}
        r_copy["top5_improved"] = sorted(r["per_tensor"], key=lambda x: -x["improvement_pct"])[:5]
        export.append(r_copy)
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Overall verdict
    int6_gauss = [x for x in all_results if x["config"] == "int6" and x["weight_type"] == "gaussian_outliers"]
    int6_qat = [x for x in all_results if x["config"] == "int6" and x["weight_type"] == "qat_shaped_int6"]
    int5_gauss = [x for x in all_results if x["config"] == "int5" and x["weight_type"] == "gaussian_outliers"]

    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    if int6_gauss and int6_gauss[0]["overall_improvement_pct"] > 1.0:
        print(f"POSITIVE: Rotation helps at int6 ({int6_gauss[0]['overall_improvement_pct']:+.2f}%)")
        sys.exit(0)
    elif int5_gauss and int5_gauss[0]["overall_improvement_pct"] > 1.0:
        print(f"MARGINAL: Rotation helps at int5 ({int5_gauss[0]['overall_improvement_pct']:+.2f}%) but not int6")
        sys.exit(0)
    else:
        print("NEGATIVE: Rotation provides negligible benefit even at int6/int5")
        sys.exit(1)


if __name__ == "__main__":
    main()
