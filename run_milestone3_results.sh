#!/bin/bash
#SBATCH -p gpu-turing
#SBATCH --gres gpu:1
#SBATCH --job-name=ppo-m3
#SBATCH --output=milestone3-%j.out

set -euo pipefail

cd /home/cme213/karenvo/distributed-ppo-engine

rm -f report/correctness.txt report/benchmark.txt report/benchmark.csv

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo
echo "Device sanity check"
./build/gae_test 1 4 0.99 0.95 --iters 10

echo
echo "Checking Python dependencies"
python3 -c "import numpy" || {
  echo
  echo "Missing Python package(s). Run this once, then resubmit:"
  echo "  python3 -m pip install --user numpy"
  exit 1
}

echo
echo "Correctness"
python3 python/test_correctness.py | tee report/correctness.txt

echo
echo "Benchmark"
python3 python/bench.py | tee report/benchmark.txt

echo
echo "Outputs written to:"
echo "  report/correctness.txt"
echo "  report/benchmark.txt"
echo "  report/benchmark.csv"
echo "  report/figures/bench_gae.pdf if matplotlib is available"
