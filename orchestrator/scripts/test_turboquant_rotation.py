#!/usr/bin/env python3
"""
EXP-H38-A: TurboQuant-Inspired Hadamard Rotation Before Weight Quantization
=============================================================================
Standalone benchmark — requires only numpy. Tests the rotation-quantization
technique on realistic transformer weight distributions matching the Parameter
Golf Stack A architecture (11L, 512d, MLP 3x = 1536).

Run:  python3 test_turboquant_rotation.py
"""

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ── Architecture constants (Stack A rank 1) ─────────────────────────────────
D_MODEL = 512
MLP_DIM = 1536     # 3x expansion
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS  # 64
NUM_LAYERS = 11
VOCAB_SIZE = 50304

INT8_CLIP_Q = 0.9999984  # 99.99984th percentile

# ── Walsh-Hadamard Transform ────────────────────────────────────────────────

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def walsh_hadamard_transform(x: np.ndarray) -> np.ndarray:
    """Butterfly WHT on last dimension. Expects power-of-2."""
    d = x.shape[-1]
    h = 1
    while h < d:
        x = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = np.stack([a + b, a - b], axis=-2)
        x = x.reshape(*x.shape[:-3], d)
        h <<= 1
    return x


def _rotation_signs(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 2.0 * rng.integers(0, 2, size=d).astype(np.float64) - 1.0


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def can_rotate(t: np.ndarray) -> bool:
    """Only rotate power-of-2 last dims (exact WHT round-trip)."""
    return t.ndim >= 2 and _is_pow2(t.shape[-1])


def rotate_for_quant(t: np.ndarray, seed: int) -> np.ndarray:
    d = t.shape[-1]
    assert _is_pow2(d), f"rotate_for_quant requires power-of-2 dim, got {d}"
    signs = _rotation_signs(d, seed)
    return walsh_hadamard_transform(t * signs) / np.sqrt(d)


def inverse_rotate_for_quant(t: np.ndarray, seed: int) -> np.ndarray:
    d = t.shape[-1]
    assert _is_pow2(d), f"inverse_rotate_for_quant requires power-of-2 dim, got {d}"
    signs = _rotation_signs(d, seed)
    return signs * walsh_hadamard_transform(t) / np.sqrt(d)


# ── Int8 Per-Row Quantization ────────────────────────────────────────────────

def quant_int8_per_row(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (quantized_int8, scales_fp16, dequantized_fp64)."""
    clip_abs = np.percentile(np.abs(t), INT8_CLIP_Q * 100, axis=1)
    clipped = np.clip(t, -clip_abs[:, None], clip_abs[:, None])
    scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0)
    q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8)
    dequant = q.astype(np.float64) * scale[:, None]
    return q, scale.astype(np.float16), dequant


# ── MSE-Adaptive Rotation Quantization ───────────────────────────────────────

def quantize_with_rotation(W: np.ndarray, seed: int) -> Tuple[np.ndarray, bool, float]:
    """Returns (dequantized_W, used_rotation, mse)."""
    # Baseline
    _, _, dq_base = quant_int8_per_row(W)
    mse_base = np.mean((W - dq_base) ** 2)

    # Only try rotation for power-of-2 last dim
    if not can_rotate(W):
        return dq_base, False, mse_base

    # Rotated
    W_rot = rotate_for_quant(W, seed)
    _, _, dq_rot = quant_int8_per_row(W_rot)
    W_hat = inverse_rotate_for_quant(dq_rot, seed)
    mse_rot = np.mean((W - W_hat) ** 2)

    if mse_rot < mse_base:
        return W_hat, True, mse_rot
    return dq_base, False, mse_base


# ── Realistic Weight Generator ───────────────────────────────────────────────

def generate_layer_weights(layer_idx: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Generate weight tensors mimicking a trained transformer layer."""
    # Real transformer weights are roughly Gaussian with some structure:
    # - Attention Q/K/V projections: (d_model, head_dim * n_heads)
    # - MLP up/down: (d_model, mlp_dim) — often have outlier columns
    # - Earlier layers tend to have larger magnitude ranges

    scale_factor = 1.0 + 0.3 * (NUM_LAYERS - layer_idx) / NUM_LAYERS  # earlier layers slightly larger

    weights = {}

    # Attention weights
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        n_heads = NUM_KV_HEADS if name in ("k_proj", "v_proj") else NUM_HEADS
        W = rng.normal(0, scale_factor * 0.02, (D_MODEL, HEAD_DIM * n_heads))
        # Add outlier columns (common in real transformers)
        outlier_cols = rng.choice(HEAD_DIM * n_heads, size=max(1, HEAD_DIM * n_heads // 64), replace=False)
        W[:, outlier_cols] *= rng.uniform(3.0, 8.0, size=len(outlier_cols))
        weights[f"blocks.{layer_idx}.attn.{name}.weight"] = W

    # MLP weights (3x expansion)
    for name, shape in [("up", (D_MODEL, MLP_DIM)), ("down", (MLP_DIM, D_MODEL))]:
        W = rng.normal(0, scale_factor * 0.02, shape)
        # MLP up tends to have more outlier structure
        if name == "up":
            outlier_rows = rng.choice(D_MODEL, size=D_MODEL // 32, replace=False)
            W[outlier_rows, :] *= rng.uniform(2.0, 5.0, size=(len(outlier_rows), 1))
        outlier_cols = rng.choice(shape[1], size=shape[1] // 64, replace=False)
        W[:, outlier_cols] *= rng.uniform(3.0, 6.0, size=len(outlier_cols))
        weights[f"blocks.{layer_idx}.mlp.{name}.weight"] = W

    return weights


def generate_full_model_weights(seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate all 2D weight tensors for an 11L/512d/MLP3x model."""
    rng = np.random.default_rng(seed)
    all_weights = {}

    # Embedding (typically passthrough in int8 export, but test anyway)
    all_weights["embed.weight"] = rng.normal(0, 0.02, (VOCAB_SIZE, D_MODEL))

    # Transformer layers
    for layer_idx in range(NUM_LAYERS):
        all_weights.update(generate_layer_weights(layer_idx, rng))

    # LM head (often tied with embed)
    all_weights["lm_head.weight"] = rng.normal(0, 0.02, (VOCAB_SIZE, D_MODEL))

    return all_weights


# ── Benchmarks ───────────────────────────────────────────────────────────────

@dataclass
class TensorResult:
    name: str
    shape: Tuple[int, ...]
    mse_baseline: float
    mse_rotated: float
    used_rotation: bool
    improvement_pct: float
    cosine_baseline: float
    cosine_rotated: float
    time_baseline_ms: float
    time_rotated_ms: float


@dataclass
class BenchmarkReport:
    tensors: List[TensorResult] = field(default_factory=list)
    total_baseline_mse: float = 0.0
    total_best_mse: float = 0.0
    overall_improvement_pct: float = 0.0
    tensors_improved: int = 0
    tensors_neutral: int = 0
    total_tensors: int = 0
    total_params: int = 0
    time_baseline_total_ms: float = 0.0
    time_rotated_total_ms: float = 0.0
    time_overhead_pct: float = 0.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.ravel(), b.ravel()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 1.0
    return float(np.dot(a_flat, b_flat) / denom)


def benchmark_tensor(name: str, W: np.ndarray, seed: int) -> TensorResult:
    """Benchmark baseline vs rotated quantization for a single tensor."""
    # Baseline
    t0 = time.perf_counter()
    _, _, dq_base = quant_int8_per_row(W)
    t_base = (time.perf_counter() - t0) * 1000

    mse_base = float(np.mean((W - dq_base) ** 2))
    cos_base = cosine_sim(W, dq_base)

    # Rotated
    t0 = time.perf_counter()
    dq_rot, used_rotation, mse_rot = quantize_with_rotation(W, seed)
    t_rot = (time.perf_counter() - t0) * 1000

    cos_rot = cosine_sim(W, dq_rot)

    improvement = (1 - mse_rot / mse_base) * 100 if mse_base > 0 else 0.0

    return TensorResult(
        name=name,
        shape=W.shape,
        mse_baseline=mse_base,
        mse_rotated=mse_rot,
        used_rotation=used_rotation,
        improvement_pct=improvement,
        cosine_baseline=cos_base,
        cosine_rotated=cos_rot,
        time_baseline_ms=t_base,
        time_rotated_ms=t_rot,
    )


def run_full_benchmark() -> BenchmarkReport:
    print("=" * 72)
    print("EXP-H38-A: TurboQuant Hadamard Rotation Quantization Benchmark")
    print("=" * 72)
    print(f"Architecture: {NUM_LAYERS}L / {D_MODEL}d / MLP {MLP_DIM} / {NUM_HEADS}H / {NUM_KV_HEADS}KV")
    print(f"Quantization: int8 per-row, clip {INT8_CLIP_Q*100:.5f}th percentile")
    print(f"Rotation: Walsh-Hadamard + random signs, MSE-adaptive per-tensor")
    print()

    print("Generating realistic model weights... ", end="", flush=True)
    t0 = time.perf_counter()
    weights = generate_full_model_weights(seed=42)
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    total_params = sum(w.size for w in weights.values())
    print(f"Total tensors: {len(weights)}, Total params: {total_params:,}")
    print()

    report = BenchmarkReport()
    report.total_tensors = len(weights)
    report.total_params = total_params

    # Header
    print(f"{'Tensor':<50s} {'Shape':>12s} {'Base MSE':>12s} {'Rot MSE':>12s} {'Delta':>8s} {'Rot?':>5s} {'ms_b':>6s} {'ms_r':>6s}")
    print("-" * 115)

    for name, W in sorted(weights.items()):
        seed = 42 ^ (hash(name) & 0xFFFFFFFF)
        result = benchmark_tensor(name, W, seed)
        report.tensors.append(result)

        # Accumulate
        report.total_baseline_mse += result.mse_baseline * W.size
        report.total_best_mse += result.mse_rotated * W.size
        report.time_baseline_total_ms += result.time_baseline_ms
        report.time_rotated_total_ms += result.time_rotated_ms
        if result.used_rotation:
            report.tensors_improved += 1
        else:
            report.tensors_neutral += 1

        tag = "ROT" if result.used_rotation else "---"
        shape_str = f"{W.shape[0]}x{W.shape[1]}"
        print(
            f"{name:<50s} {shape_str:>12s} {result.mse_baseline:>12.8f} {result.mse_rotated:>12.8f} "
            f"{result.improvement_pct:>+7.1f}% {tag:>5s} {result.time_baseline_ms:>5.1f} {result.time_rotated_ms:>5.1f}"
        )

    # Weighted average MSE
    report.total_baseline_mse /= total_params
    report.total_best_mse /= total_params
    report.overall_improvement_pct = (1 - report.total_best_mse / report.total_baseline_mse) * 100
    report.time_overhead_pct = (
        (report.time_rotated_total_ms / report.time_baseline_total_ms - 1) * 100
        if report.time_baseline_total_ms > 0
        else 0
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Tensors rotated:    {report.tensors_improved}/{report.total_tensors}")
    print(f"Tensors fallback:   {report.tensors_neutral}/{report.total_tensors}")
    print(f"Weighted avg MSE (baseline): {report.total_baseline_mse:.10f}")
    print(f"Weighted avg MSE (best):     {report.total_best_mse:.10f}")
    print(f"Overall MSE improvement:     {report.overall_improvement_pct:+.2f}%")
    print(f"Export time (baseline):      {report.time_baseline_total_ms:.0f}ms")
    print(f"Export time (w/ rotation):   {report.time_rotated_total_ms:.0f}ms")
    print(f"Time overhead:               {report.time_overhead_pct:+.1f}%")
    print()

    # Per-layer breakdown
    print("Per-Layer MSE Improvement:")
    for layer_idx in range(NUM_LAYERS):
        layer_tensors = [t for t in report.tensors if f"blocks.{layer_idx}." in t.name]
        if not layer_tensors:
            continue
        avg_imp = np.mean([t.improvement_pct for t in layer_tensors])
        rotated = sum(1 for t in layer_tensors if t.used_rotation)
        print(f"  Layer {layer_idx:2d}: avg improvement {avg_imp:+.1f}%, rotated {rotated}/{len(layer_tensors)} tensors")

    # By tensor type
    print()
    print("By Tensor Type:")
    for pattern, label in [
        ("q_proj", "Q proj"), ("k_proj", "K proj"), ("v_proj", "V proj"), ("o_proj", "O proj"),
        ("mlp.up", "MLP up"), ("mlp.down", "MLP down"),
        ("embed", "Embedding"), ("lm_head", "LM head"),
    ]:
        matched = [t for t in report.tensors if pattern in t.name]
        if not matched:
            continue
        avg_imp = np.mean([t.improvement_pct for t in matched])
        rotated = sum(1 for t in matched if t.used_rotation)
        avg_cos_base = np.mean([t.cosine_baseline for t in matched])
        avg_cos_rot = np.mean([t.cosine_rotated for t in matched])
        print(
            f"  {label:10s}: avg improvement {avg_imp:+.1f}%, "
            f"rotated {rotated}/{len(matched)}, "
            f"cosine {avg_cos_base:.8f} -> {avg_cos_rot:.8f}"
        )

    return report


def export_report_json(report: BenchmarkReport, path: str):
    """Export machine-readable report."""
    data = {
        "experiment_id": "EXP-H38-A",
        "hypothesis_id": "H38",
        "title": "TurboQuant Hadamard Rotation Before Weight Quantization",
        "architecture": f"{NUM_LAYERS}L/{D_MODEL}d/MLP{MLP_DIM}",
        "platform": f"numpy {np.__version__}, {os.uname().machine}",
        "summary": {
            "total_tensors": report.total_tensors,
            "total_params": report.total_params,
            "tensors_rotated": report.tensors_improved,
            "tensors_fallback": report.tensors_neutral,
            "weighted_mse_baseline": report.total_baseline_mse,
            "weighted_mse_best": report.total_best_mse,
            "overall_mse_improvement_pct": report.overall_improvement_pct,
            "export_time_baseline_ms": report.time_baseline_total_ms,
            "export_time_rotated_ms": report.time_rotated_total_ms,
            "time_overhead_pct": report.time_overhead_pct,
        },
        "per_tensor": [
            {
                "name": t.name,
                "shape": list(t.shape),
                "mse_baseline": t.mse_baseline,
                "mse_rotated": t.mse_rotated,
                "used_rotation": t.used_rotation,
                "improvement_pct": t.improvement_pct,
                "cosine_baseline": t.cosine_baseline,
                "cosine_rotated": t.cosine_rotated,
            }
            for t in report.tensors
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Report saved to {path}")


# ── Round-Trip Fidelity Test ─────────────────────────────────────────────────

def test_roundtrip_fidelity():
    """Verify rotation is perfectly invertible (numerical precision)."""
    print("\n" + "=" * 72)
    print("ROUND-TRIP FIDELITY TEST")
    print("=" * 72)

    dims = [64, 128, 256, 512, 1024, 2048]  # power-of-2 only (WHT requirement)
    print(f"{'Dim':>6s} {'Max Error':>12s} {'Mean Error':>12s} {'Status':>8s}")
    print("-" * 45)

    all_pass = True
    for d in dims:
        W = np.random.randn(256, d)
        W_rot = rotate_for_quant(W, seed=42)
        W_back = inverse_rotate_for_quant(W_rot, seed=42)
        max_err = np.max(np.abs(W - W_back))
        mean_err = np.mean(np.abs(W - W_back))
        ok = max_err < 1e-10
        all_pass = all_pass and ok
        print(f"{d:>6d} {max_err:>12.2e} {mean_err:>12.2e} {'PASS' if ok else 'FAIL':>8s}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return all_pass


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Round-trip test first
    rt_ok = test_roundtrip_fidelity()
    if not rt_ok:
        print("ABORT: Round-trip fidelity test failed!")
        sys.exit(1)

    print()

    # Full model benchmark
    report = run_full_benchmark()

    # Export JSON report
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "EXP-H38-A-results.json")
    export_report_json(report, json_path)

    # Exit code: 0 if improvement > 0
    sys.exit(0 if report.overall_improvement_pct > 0 else 1)
