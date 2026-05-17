# Distributed PPO Engine: Milestone 3 CUDA Kernels

Self-contained CUDA implementation of PPO rollout-buffer preprocessing kernels for CME 213 Milestone 3.

Implemented kernels:

1. Fused TD residual + GAE backward scan.
2. Advantage normalization with hand-written reduction.
3. Basic minibatch index gather.

No simulator, MPI, actor-critic model, or PPO loss is included in this milestone. Inputs are synthetic rollout buffers.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The build expects CUDA Toolkit 12+ and an NVIDIA GPU target. It builds:

- `build/libppo_kernels.so`
- `build/gae_test`
- `build/test_smoke`
- `build/bench_gae`

## Correctness

```bash
python python/test_correctness.py
```

## Benchmark

```bash
python python/bench.py
```

The benchmark writes `report/figures/bench_gae.pdf` and prints a summary table.

