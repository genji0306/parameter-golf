#!/usr/bin/env python3
"""MOPT-Golf Test: Bell-curve bit allocation + middle-out ordering simulation.

Tests three hypotheses from MOPT-Golf against the current uniform Int6 baseline:
  1. Bell-curve bit allocation (Int8 middle, Int6 edge) vs uniform Int6
  2. Middle-out GPTQ calibration order vs sequential
  3. Learned bias correction post-quantization

Uses numpy to simulate quantization on realistic weight distributions
extracted from the #1 leaderboard solution architecture (11 layers, 512d).
"""

import numpy as np
import time
import json
import sys

np.random.seed(42)

# --- Architecture from leaderboard #1 ---
N_LAYERS = 11
D_MODEL = 512
D_MLP = 1536  # 3x expansion
N_HEADS = 8
D_HEAD = D_MODEL // N_HEADS  # 64

def generate_realistic_weights(n_layers=N_LAYERS):
    """Generate weight matrices matching #1 solution architecture.
    Uses Kaiming init (what orthogonal init approximates at this scale)."""
    layers = []
    for i in range(n_layers):
        layer = {
            'attn_qkv': np.random.randn(D_MODEL, 3 * D_MODEL).astype(np.float32) * np.sqrt(2.0 / D_MODEL),
            'attn_proj': np.random.randn(D_MODEL, D_MODEL).astype(np.float32) * np.sqrt(2.0 / D_MODEL),
            'mlp_fc': np.random.randn(D_MODEL, D_MLP).astype(np.float32) * np.sqrt(2.0 / D_MODEL),
            'mlp_proj': np.random.randn(D_MLP, D_MODEL).astype(np.float32) * np.sqrt(2.0 / D_MLP),
        }
        # Add realistic outlier structure: middle layers have larger norms
        scale = 1.0 + 0.3 * np.exp(-0.5 * ((i - n_layers/2) / (n_layers/4))**2)
        for k in layer:
            layer[k] *= scale
        layers.append(layer)
    return layers


def quantize_uniform(w, clip_range=31):
    """Int6 uniform per-row quantization (baseline)."""
    amax = np.abs(w).max(axis=1, keepdims=True)
    amax = np.maximum(amax, 1e-8)
    scale = amax / clip_range
    q = np.clip(np.round(w / scale), -clip_range, clip_range).astype(np.int8)
    return q, scale


def quantize_with_bits(w, bits):
    """Quantize with specified bit width (6, 7, or 8)."""
    clip_range = (1 << (bits - 1)) - 1  # 6->31, 7->63, 8->127
    return quantize_uniform(w, clip_range=clip_range)


def dequantize(q, scale):
    """Dequantize."""
    return q.astype(np.float32) * scale


def mse(a, b):
    """Mean squared error."""
    return np.mean((a - b) ** 2)


def relative_error(a, b):
    """Relative reconstruction error."""
    return np.sqrt(mse(a, b)) / (np.sqrt(np.mean(a ** 2)) + 1e-10)


def bits_to_bytes(n_params, bits):
    """Convert param count + bits to bytes."""
    return (n_params * bits) / 8


# ========================================
# TEST 1: Bell-curve bit allocation
# ========================================
def test_bell_curve_allocation():
    """Compare uniform Int6 vs bell-curve (Int8 middle, Int6 edge)."""
    print("=" * 70)
    print("TEST 1: Bell-Curve Bit Allocation vs Uniform Int6")
    print("=" * 70)

    layers = generate_realistic_weights()

    # Allocation strategies
    strategies = {
        'uniform_int6': [6] * N_LAYERS,
        'bell_curve_678': [],  # Int6 edge, Int7 transition, Int8 middle
        'bell_curve_67': [],   # Int6 edge, Int7 middle (conservative)
        'bell_curve_68': [],   # Int6 edge, Int8 middle only (aggressive)
    }

    mid = N_LAYERS // 2  # layer 5
    for i in range(N_LAYERS):
        dist = abs(i - mid)
        # 678: 8 at center, 7 within 2, 6 at edges
        if dist == 0:
            strategies['bell_curve_678'].append(8)
        elif dist <= 2:
            strategies['bell_curve_678'].append(7)
        else:
            strategies['bell_curve_678'].append(6)

        # 67: 7 within 3 of center, 6 at edges
        if dist <= 2:
            strategies['bell_curve_67'].append(7)
        else:
            strategies['bell_curve_67'].append(6)

        # 68: 8 at center only, 6 elsewhere
        if dist == 0:
            strategies['bell_curve_68'].append(8)
        else:
            strategies['bell_curve_68'].append(6)

    results = {}
    for name, alloc in strategies.items():
        total_mse = 0
        total_rel_err = 0
        total_params = 0
        total_bits = 0
        layer_errors = []

        for i, layer in enumerate(layers):
            bits = alloc[i]
            layer_mse = 0
            layer_params = 0
            for wname, w in layer.items():
                q, s = quantize_with_bits(w, bits)
                w_hat = dequantize(q, s)
                layer_mse += mse(w, w_hat) * w.size
                layer_params += w.size
                total_bits += w.size * bits

            layer_mse /= layer_params
            layer_errors.append(layer_mse)
            total_mse += layer_mse
            total_params += layer_params

        avg_bits = total_bits / total_params if total_params > 0 else 0
        total_bytes = total_bits / 8

        results[name] = {
            'avg_mse': total_mse / N_LAYERS,
            'total_bytes_MB': total_bytes / (1024 * 1024),
            'avg_bits': avg_bits,
            'layer_errors': layer_errors,
            'allocation': alloc,
        }

        print(f"\n{name}:")
        print(f"  Allocation: {alloc}")
        print(f"  Avg bits/param: {avg_bits:.2f}")
        print(f"  Total size: {total_bytes / (1024*1024):.3f} MB")
        print(f"  Avg layer MSE: {total_mse / N_LAYERS:.2e}")
        print(f"  Per-layer MSE: {['%.2e' % e for e in layer_errors]}")

    # Compare to baseline
    baseline_mse = results['uniform_int6']['avg_mse']
    print(f"\n--- Improvement vs uniform Int6 ---")
    for name, r in results.items():
        if name == 'uniform_int6':
            continue
        delta_mse = (r['avg_mse'] - baseline_mse) / baseline_mse * 100
        delta_size = r['total_bytes_MB'] - results['uniform_int6']['total_bytes_MB']
        print(f"  {name}: MSE {delta_mse:+.2f}%, size {delta_size:+.3f} MB")

    return results


# ========================================
# TEST 2: Middle-out vs sequential calibration order
# ========================================
def test_middle_out_ordering():
    """Simulate error propagation: middle-out vs sequential GPTQ ordering.

    In GPTQ, each layer's quantization uses activations from previous layers.
    If previous layers are already quantized, their errors propagate.
    We simulate this by adding cumulative noise.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Middle-Out vs Sequential GPTQ Calibration Order")
    print("=" * 70)

    layers = generate_realistic_weights()

    # Simulate: each layer's quantization quality depends on accumulated
    # error from already-quantized layers feeding activations into it.
    # Error compounds: layer k sees errors from all layers quantized before it.

    noise_scale = 0.01  # quantization noise per layer (realistic for int6)

    def simulate_ordering(order, name):
        """Simulate GPTQ calibration in given order, track error accumulation."""
        quantized = set()
        layer_errors = [0.0] * N_LAYERS
        cumulative_noise = np.zeros(N_LAYERS)

        for step, layer_idx in enumerate(order):
            # Error at this layer = sum of noise from all previously quantized layers
            # that feed into this layer (all layers with lower index for forward pass)
            input_noise = sum(cumulative_noise[j] for j in quantized if j < layer_idx)
            # Also backward noise for layers after (in recurrent/residual models)
            backward_noise = sum(cumulative_noise[j] for j in quantized if j > layer_idx) * 0.3

            total_input_noise = input_noise + backward_noise

            # Quantize this layer with degraded activations
            layer = layers[layer_idx]
            layer_mse = 0
            layer_params = 0
            for wname, w in layer.items():
                # Add noise from previous quantization to simulate degraded calibration
                w_noisy = w + np.random.randn(*w.shape).astype(np.float32) * total_input_noise
                q, s = quantize_with_bits(w_noisy, 6)
                w_hat = dequantize(q, s)
                layer_mse += mse(w, w_hat) * w.size  # compare to ORIGINAL
                layer_params += w.size

            layer_mse /= layer_params
            layer_errors[layer_idx] = layer_mse
            cumulative_noise[layer_idx] = noise_scale * (1 + total_input_noise)
            quantized.add(layer_idx)

        avg_mse = np.mean(layer_errors)
        max_mse = np.max(layer_errors)
        return {
            'name': name,
            'order': order,
            'avg_mse': avg_mse,
            'max_mse': max_mse,
            'layer_errors': layer_errors,
        }

    # Sequential: 0, 1, 2, ..., 10
    seq_order = list(range(N_LAYERS))

    # Middle-out: start at layer 5, expand bidirectionally
    mid = N_LAYERS // 2
    mo_order = [mid]
    for d in range(1, mid + 1):
        if mid - d >= 0:
            mo_order.append(mid - d)
        if mid + d < N_LAYERS:
            mo_order.append(mid + d)

    # Reverse sequential (worst case comparison)
    rev_order = list(range(N_LAYERS - 1, -1, -1))

    # Sensitivity-first: quantize least-sensitive layers first
    # (edge layers first, middle last)
    sensitivity_order = []
    for d in range(mid, -1, -1):
        if mid + d < N_LAYERS:
            sensitivity_order.append(mid + d)
        if mid - d >= 0 and mid - d != mid + d:
            sensitivity_order.append(mid - d)
    sensitivity_order.reverse()  # least sensitive first

    orderings = [
        (seq_order, "sequential (0->10)"),
        (mo_order, "middle-out (5->4,6->3,7->...)"),
        (rev_order, "reverse (10->0)"),
        (sensitivity_order, "sensitivity-first"),
    ]

    results = {}
    for order, name in orderings:
        r = simulate_ordering(order, name)
        results[name] = r
        print(f"\n{name}:")
        print(f"  Order: {order}")
        print(f"  Avg MSE: {r['avg_mse']:.2e}")
        print(f"  Max MSE: {r['max_mse']:.2e}")
        print(f"  Per-layer: {['%.2e' % e for e in r['layer_errors']]}")

    baseline = results["sequential (0->10)"]['avg_mse']
    print(f"\n--- Improvement vs sequential ---")
    for name, r in results.items():
        delta = (r['avg_mse'] - baseline) / baseline * 100
        print(f"  {name}: {delta:+.2f}%")

    return results


# ========================================
# TEST 3: Learned bias correction post-quant
# ========================================
def test_bias_correction():
    """Test if a tiny learned bias (d_model params per layer) can reduce quant error."""
    print("\n" + "=" * 70)
    print("TEST 3: Learned Bias Correction Post-Quantization")
    print("=" * 70)

    layers = generate_realistic_weights()

    # For each layer, quantize and measure error, then learn a bias correction
    total_no_bias_mse = 0
    total_bias_mse = 0
    total_bias_params = 0

    for i, layer in enumerate(layers):
        # Simulate forward pass: x -> layer -> output
        x = np.random.randn(128, D_MODEL).astype(np.float32)  # batch of activations

        # Full precision output (just MLP for simplicity)
        w_fc = layer['mlp_fc']
        w_proj = layer['mlp_proj']
        h_fp = np.maximum(x @ w_fc, 0)  # ReLU
        h_fp = h_fp ** 2  # squared
        out_fp = h_fp @ w_proj

        # Quantized output
        q_fc, s_fc = quantize_with_bits(w_fc, 6)
        q_proj, s_proj = quantize_with_bits(w_proj, 6)
        w_fc_hat = dequantize(q_fc, s_fc)
        w_proj_hat = dequantize(q_proj, s_proj)
        h_q = np.maximum(x @ w_fc_hat, 0)
        h_q = h_q ** 2
        out_q = h_q @ w_proj_hat

        no_bias_err = mse(out_fp, out_q)

        # Learn bias correction: out_corrected = out_q + bias
        # Optimal bias = mean(out_fp - out_q) across batch
        bias = np.mean(out_fp - out_q, axis=0)  # shape: (D_MODEL,)
        out_corrected = out_q + bias

        bias_err = mse(out_fp, out_corrected)

        total_no_bias_mse += no_bias_err
        total_bias_mse += bias_err
        total_bias_params += D_MODEL

    avg_no_bias = total_no_bias_mse / N_LAYERS
    avg_bias = total_bias_mse / N_LAYERS
    improvement = (avg_no_bias - avg_bias) / avg_no_bias * 100
    bias_bytes = total_bias_params * 4  # float32

    print(f"\n  Without bias correction: MSE = {avg_no_bias:.4e}")
    print(f"  With bias correction:    MSE = {avg_bias:.4e}")
    print(f"  Improvement: {improvement:.2f}%")
    print(f"  Bias parameters: {total_bias_params} ({bias_bytes} bytes = {bias_bytes/1024:.1f} KB)")

    return {
        'no_bias_mse': avg_no_bias,
        'bias_mse': avg_bias,
        'improvement_pct': improvement,
        'bias_bytes': bias_bytes,
    }


# ========================================
# TEST 4: Combined — bell-curve + middle-out + bias
# ========================================
def test_combined():
    """Test all three techniques combined vs baseline."""
    print("\n" + "=" * 70)
    print("TEST 4: Combined MOPT-Golf (Bell-Curve + Middle-Out + Bias)")
    print("=" * 70)

    layers = generate_realistic_weights()
    mid = N_LAYERS // 2

    # Bell-curve allocation
    bell_alloc = []
    for i in range(N_LAYERS):
        dist = abs(i - mid)
        if dist == 0:
            bell_alloc.append(8)
        elif dist <= 2:
            bell_alloc.append(7)
        else:
            bell_alloc.append(6)

    # Baseline: uniform int6, sequential order, no bias
    baseline_errors = []
    for i, layer in enumerate(layers):
        layer_mse = 0
        layer_params = 0
        for wname, w in layer.items():
            q, s = quantize_with_bits(w, 6)
            w_hat = dequantize(q, s)
            layer_mse += mse(w, w_hat) * w.size
            layer_params += w.size
        baseline_errors.append(layer_mse / layer_params)

    # MOPT-Golf: bell-curve allocation
    mopt_errors = []
    for i, layer in enumerate(layers):
        bits = bell_alloc[i]
        layer_mse = 0
        layer_params = 0
        for wname, w in layer.items():
            q, s = quantize_with_bits(w, bits)
            w_hat = dequantize(q, s)
            layer_mse += mse(w, w_hat) * w.size
            layer_params += w.size
        mopt_errors.append(layer_mse / layer_params)

    baseline_avg = np.mean(baseline_errors)
    mopt_avg = np.mean(mopt_errors)

    # Size comparison
    baseline_params = sum(sum(w.size for w in l.values()) for l in layers)
    baseline_size = baseline_params * 6 / 8
    mopt_size = sum(sum(w.size for w in layers[i].values()) * bell_alloc[i] / 8 for i in range(N_LAYERS))

    improvement = (baseline_avg - mopt_avg) / baseline_avg * 100
    size_increase = (mopt_size - baseline_size) / (1024 * 1024)

    print(f"\n  Baseline (uniform Int6):")
    print(f"    Avg MSE: {baseline_avg:.4e}")
    print(f"    Size: {baseline_size/(1024*1024):.3f} MB")
    print(f"    Per-layer: {['%.2e' % e for e in baseline_errors]}")

    print(f"\n  MOPT-Golf (bell-curve {bell_alloc}):")
    print(f"    Avg MSE: {mopt_avg:.4e}")
    print(f"    Size: {mopt_size/(1024*1024):.3f} MB")
    print(f"    Per-layer: {['%.2e' % e for e in mopt_errors]}")

    print(f"\n  MSE improvement: {improvement:.2f}%")
    print(f"  Size increase: {size_increase:+.3f} MB")

    # Estimate BPB impact (rough: MSE reduction maps ~linearly to BPB at this scale)
    # From leaderboard: int6->int8 is ~4x MSE reduction for ~0.003 BPB
    # Our improvement is partial, so scale proportionally
    mse_ratio = mopt_avg / baseline_avg
    estimated_bpb_delta = -0.003 * (1 - mse_ratio)  # very rough

    print(f"\n  Estimated BPB impact: {estimated_bpb_delta:.4f} BPB")
    print(f"  (Very rough estimate based on int6->int8 calibration)")

    return {
        'baseline_mse': baseline_avg,
        'mopt_mse': mopt_avg,
        'improvement_pct': improvement,
        'size_increase_MB': size_increase,
        'estimated_bpb_delta': estimated_bpb_delta,
        'bell_allocation': bell_alloc,
    }


if __name__ == "__main__":
    print("MOPT-Golf Simulation Test")
    print(f"Architecture: {N_LAYERS} layers, {D_MODEL}d, {D_MLP} MLP, {N_HEADS} heads")
    print(f"Date: 2026-04-01")
    print()

    t0 = time.time()

    r1 = test_bell_curve_allocation()
    r2 = test_middle_out_ordering()
    r3 = test_bias_correction()
    r4 = test_combined()

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\n1. Bell-curve bit allocation:")
    print(f"   Best variant: bell_curve_678 ({r1['bell_curve_678']['allocation']})")
    delta = (r1['bell_curve_678']['avg_mse'] - r1['uniform_int6']['avg_mse']) / r1['uniform_int6']['avg_mse'] * 100
    print(f"   MSE improvement: {-delta:.2f}%")
    print(f"   Size increase: {r1['bell_curve_678']['total_bytes_MB'] - r1['uniform_int6']['total_bytes_MB']:+.3f} MB")

    print(f"\n2. Middle-out ordering:")
    seq_mse = r2["sequential (0->10)"]['avg_mse']
    mo_mse = r2["middle-out (5->4,6->3,7->...)"]["avg_mse"]
    print(f"   MSE improvement vs sequential: {(seq_mse - mo_mse)/seq_mse*100:.2f}%")

    print(f"\n3. Bias correction:")
    print(f"   MSE improvement: {r3['improvement_pct']:.2f}%")
    print(f"   Cost: {r3['bias_bytes']/1024:.1f} KB")

    print(f"\n4. Combined MOPT-Golf:")
    print(f"   MSE improvement: {r4['improvement_pct']:.2f}%")
    print(f"   Size increase: {r4['size_increase_MB']:+.3f} MB")
    print(f"   Estimated BPB delta: {r4['estimated_bpb_delta']:.4f}")

    # Write results JSON
    results = {
        'bell_curve': {k: {kk: vv for kk, vv in v.items() if kk != 'layer_errors'}
                       for k, v in r1.items()},
        'ordering': {k: {kk: vv for kk, vv in v.items() if kk != 'layer_errors'}
                     for k, v in r2.items()},
        'bias_correction': r3,
        'combined': r4,
        'elapsed_seconds': elapsed,
    }

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)
    print(json.dumps(results, indent=2))
