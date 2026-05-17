#include "ppo_kernels.h"

#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    int N = argc > 1 ? std::atoi(argv[1]) : 16384;
    int T = argc > 2 ? std::atoi(argv[2]) : 32;
    int iters = argc > 3 ? std::atoi(argv[3]) : 100;

    std::mt19937 rng(0);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::bernoulli_distribution done_dist(0.05);
    std::vector<float> rewards(static_cast<size_t>(N) * T);
    std::vector<float> values(static_cast<size_t>(N) * (T + 1));
    std::vector<float> dones(static_cast<size_t>(N) * T);
    for (float& x : rewards) x = normal(rng);
    for (float& x : values) x = normal(rng);
    for (float& x : dones) x = done_dist(rng) ? 1.0f : 0.0f;

    GaeResult result = run_gae_host(rewards, values, dones, N, T, 0.99f, 0.95f, iters);
    double bytes = static_cast<double>(N) * (5.0 * T + 1.0) * sizeof(float);
    double gbps = bytes / (result.milliseconds * 1.0e-3) / 1.0e9;
    std::printf("N=%d T=%d gpu_ms=%.6f bandwidth_gbps=%.3f\n", N, T, result.milliseconds, gbps);
    return 0;
}

