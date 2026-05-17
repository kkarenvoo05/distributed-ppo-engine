import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reference_gae import gae_reference
from synthetic_buffer import make_buffer


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "gae_test"
FIG = ROOT / "report" / "figures" / "bench_gae.pdf"


def run_cli(N, T, iters=100):
    proc = subprocess.run(
        [str(BIN), str(N), str(T), "0.99", "0.95", "--iters", str(iters)],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(proc.stdout.strip())


def main():
    if not BIN.exists():
        raise SystemExit(f"Missing {BIN}. Build first with: cmake -S . -B build && cmake --build build")

    shapes = [(1024, 16), (4096, 16), (16384, 16), (65536, 16),
              (1024, 32), (4096, 32), (16384, 32), (65536, 32)]
    rows = []
    peak = None
    for N, T in shapes:
        run_cli(N, T, iters=5)
        samples = [run_cli(N, T, iters=100)["gpu_ms"] for _ in range(5)]
        info = run_cli(N, T, iters=100)
        peak = info.get("peak_mem_gbps", peak)
        mean_ms = float(np.mean(samples))
        std_ms = float(np.std(samples))
        bytes_moved = N * (5 * T + 1) * 4
        bandwidth = bytes_moved / (mean_ms * 1e-3) / 1e9
        rows.append({"N": N, "T": T, "mean_ms": mean_ms, "std_ms": std_ms, "bandwidth": bandwidth})

    plt.figure(figsize=(7.0, 3.0))
    plt.subplot(1, 2, 1)
    for T in [16, 32]:
        sub = [r for r in rows if r["T"] == T]
        plt.loglog([r["N"] for r in sub], [r["mean_ms"] for r in sub], marker="o", label=f"T={T}")
    plt.xlabel("N_envs")
    plt.ylabel("Runtime (ms)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    for T in [16, 32]:
        sub = [r for r in rows if r["T"] == T]
        plt.plot([r["N"] for r in sub], [r["bandwidth"] for r in sub], marker="o", label=f"T={T}")
    if peak:
        plt.axhline(peak, color="black", linestyle="--", linewidth=1.0, label=f"Peak {peak} GB/s")
    plt.xscale("log")
    plt.xlabel("N_envs")
    plt.ylabel("Achieved bandwidth (GB/s)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG)

    rewards, values, dones = make_buffer(16384, 32, seed=123)
    t0 = time.perf_counter()
    gae_reference(rewards, values, dones, 0.99, 0.95)
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    gpu_row = next(r for r in rows if r["N"] == 16384 and r["T"] == 32)

    print("N_envs,T,mean_ms,std_ms,bandwidth_GBps")
    for r in rows:
        print(f"{r['N']},{r['T']},{r['mean_ms']:.6f},{r['std_ms']:.6f},{r['bandwidth']:.3f}")
    print(f"CPU reference at (16384,32): {cpu_ms:.3f} ms")
    print(f"GPU kernel at (16384,32): {gpu_row['mean_ms']:.6f} ms")
    print(f"Speedup: {cpu_ms / max(gpu_row['mean_ms'], 1e-9):.1f}x")
    print(f"Wrote {FIG}")


if __name__ == "__main__":
    main()
