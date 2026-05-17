import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from reference_gae import gae_reference
from synthetic_buffer import make_buffer


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "gae_test"


def require_binary():
    if not BIN.exists():
        raise SystemExit(f"Missing {BIN}. Build first with: cmake -S . -B build && cmake --build build")


def run_cmd(args):
    proc = subprocess.run(args, check=True, text=True, capture_output=True)
    return json.loads(proc.stdout.strip())


def max_errors(actual, expected):
    diff = np.abs(actual - expected)
    denom = np.maximum(np.abs(expected), 1e-8)
    return float(diff.max()), float((diff / denom).max())


def run_gae_case(N, T, gamma=0.99, lam=0.95, seed=0, custom=None):
    if custom is None:
        rewards, values, dones = make_buffer(N, T, seed=seed)
    else:
        rewards, values, dones = custom

    t0 = time.perf_counter()
    ref_adv, ref_ret = gae_reference(rewards, values, dones, gamma, lam)
    cpu_ms = (time.perf_counter() - t0) * 1000.0

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "gae_in.bin"
        out_path = Path(tmp) / "gae_out.bin"
        with in_path.open("wb") as f:
            rewards.tofile(f)
            values.tofile(f)
            dones.tofile(f)
        info = run_cmd([str(BIN), str(N), str(T), str(gamma), str(lam), str(in_path), str(out_path)])
        raw = np.fromfile(out_path, dtype=np.float32)

    adv_count = N * T
    got_adv = raw[:adv_count].reshape(N, T)
    got_ret = raw[adv_count:].reshape(N, T)
    adv_abs, adv_rel = max_errors(got_adv, ref_adv)
    ret_abs, ret_rel = max_errors(got_ret, ref_ret)
    return {
        "shape": (N, T),
        "max_abs": max(adv_abs, ret_abs),
        "max_rel": max(adv_rel, ret_rel),
        "cpu_ms": cpu_ms,
        "gpu_ms": info["gpu_ms"],
    }


def run_normalize_case(M, seed=0, constant=False):
    rng = np.random.default_rng(seed)
    x = np.full(M, 3.5, dtype=np.float32) if constant else rng.standard_normal(M).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "norm_in.bin"
        out_path = Path(tmp) / "norm_out.bin"
        x.tofile(in_path)
        run_cmd([str(BIN), "normalize", str(M), str(in_path), str(out_path), "1e-8"])
        y = np.fromfile(out_path, dtype=np.float32)
    if constant:
        assert np.max(np.abs(y)) < 1e-5
        return
    assert abs(float(y.mean())) < 5e-5
    assert abs(float(y.std()) - 1.0) < 5e-4


def run_gather_case(M, D, batch, seed=0):
    rng = np.random.default_rng(seed)
    flat = rng.standard_normal((M, D)).astype(np.float32)
    indices = rng.integers(0, M, size=batch, dtype=np.int32)
    expected = flat[indices]
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "gather_in.bin"
        out_path = Path(tmp) / "gather_out.bin"
        with in_path.open("wb") as f:
            flat.tofile(f)
            indices.tofile(f)
        run_cmd([str(BIN), "gather", str(M), str(D), str(batch), str(in_path), str(out_path)])
        got = np.fromfile(out_path, dtype=np.float32).reshape(batch, D)
    np.testing.assert_array_equal(got, expected)


def main():
    require_binary()
    rows = []
    for N, T in [(1024, 16), (1024, 32), (4096, 32), (16384, 32)]:
        rows.append(run_gae_case(N, T, seed=N + T))

    N, T = 1024, 32
    rewards = np.zeros((N, T), dtype=np.float32)
    values = np.zeros((N, T + 1), dtype=np.float32)
    rows.append(run_gae_case(N, T, custom=(rewards, values, np.zeros((N, T), dtype=np.float32))))
    rows.append(run_gae_case(N, T, custom=(rewards, values, np.ones((N, T), dtype=np.float32))))
    rows.append(run_gae_case(N, T, gamma=0.0, lam=0.95, seed=7))
    rows.append(run_gae_case(N, T, gamma=0.99, lam=0.0, seed=8))
    rows.append(run_gae_case(N, T, gamma=0.99, lam=1.0, seed=9))

    for row in rows:
        assert row["max_abs"] < 1e-5, row
        assert row["max_rel"] < 1e-4, row

    run_normalize_case(1 << 20)
    run_normalize_case(4096, constant=True)
    run_gather_case(2048, 17, 257)
    run_gather_case(8192, 64, 1024)

    print("(N_envs, T)    | max_abs_err | max_rel_err | reference ms | gpu ms | speedup")
    for row in rows[:4]:
        speedup = row["cpu_ms"] / max(row["gpu_ms"], 1e-9)
        print(
            f"{row['shape']!s:<14} | {row['max_abs']:.3e}  | {row['max_rel']:.3e}  | "
            f"{row['cpu_ms']:.3f}       | {row['gpu_ms']:.4f} | {speedup:.1f}x"
        )
    print("All correctness tests passed.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stdout)
        sys.stderr.write(exc.stderr)
        raise

