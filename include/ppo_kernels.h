#pragma once

#include <cuda_runtime.h>

#include <vector>

void launch_gae_kernel(
    const float* rewards,
    const float* values,
    const float* dones,
    float* advantages,
    float* returns,
    int N_envs,
    int T,
    float gamma,
    float lam,
    cudaStream_t stream = 0);

void launch_normalize_kernel(
    float* advantages,
    int M,
    float eps,
    cudaStream_t stream = 0);

void launch_gather_minibatch(
    const float* flat_buffer,
    const int* indices,
    float* out,
    int batch_size,
    int D,
    cudaStream_t stream = 0);

struct GaeResult {
    std::vector<float> advantages;
    std::vector<float> returns;
    float milliseconds;
};

struct NormalizeResult {
    std::vector<float> advantages;
    float milliseconds;
};

struct GatherResult {
    std::vector<float> out;
    float milliseconds;
};

GaeResult run_gae_host(
    const std::vector<float>& rewards,
    const std::vector<float>& values,
    const std::vector<float>& dones,
    int N_envs,
    int T,
    float gamma,
    float lam,
    int iterations = 1);

NormalizeResult run_normalize_host(
    const std::vector<float>& advantages,
    float eps,
    int iterations = 1);

GatherResult run_gather_host(
    const std::vector<float>& flat_buffer,
    const std::vector<int>& indices,
    int M,
    int D,
    int iterations = 1);

int get_device_peak_memory_bandwidth_gbps();

