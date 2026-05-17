#include "ppo_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

void check_kernel(const char* what) {
    check_cuda(cudaGetLastError(), what);
}

template <typename T>
void cuda_alloc(T** ptr, size_t count, const char* what) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T)), what);
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop, int iterations) {
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    return ms / static_cast<float>(std::max(iterations, 1));
}

}  // namespace

GaeResult run_gae_host(
    const std::vector<float>& rewards,
    const std::vector<float>& values,
    const std::vector<float>& dones,
    int N_envs,
    int T,
    float gamma,
    float lam,
    int iterations) {
    size_t rt_count = static_cast<size_t>(N_envs) * T;
    size_t v_count = static_cast<size_t>(N_envs) * (T + 1);
    if (rewards.size() != rt_count || dones.size() != rt_count || values.size() != v_count) {
        throw std::invalid_argument("run_gae_host received buffers with unexpected sizes");
    }

    float *d_rewards = nullptr, *d_values = nullptr, *d_dones = nullptr;
    float *d_adv = nullptr, *d_ret = nullptr;
    cuda_alloc(&d_rewards, rt_count, "cudaMalloc rewards");
    cuda_alloc(&d_values, v_count, "cudaMalloc values");
    cuda_alloc(&d_dones, rt_count, "cudaMalloc dones");
    cuda_alloc(&d_adv, rt_count, "cudaMalloc advantages");
    cuda_alloc(&d_ret, rt_count, "cudaMalloc returns");

    check_cuda(cudaMemcpy(d_rewards, rewards.data(), rt_count * sizeof(float), cudaMemcpyHostToDevice), "copy rewards");
    check_cuda(cudaMemcpy(d_values, values.data(), v_count * sizeof(float), cudaMemcpyHostToDevice), "copy values");
    check_cuda(cudaMemcpy(d_dones, dones.data(), rt_count * sizeof(float), cudaMemcpyHostToDevice), "copy dones");

    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    check_cuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < iterations; ++i) {
        launch_gae_kernel(d_rewards, d_values, d_dones, d_adv, d_ret, N_envs, T, gamma, lam);
    }
    check_kernel("launch gae kernel");
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");

    GaeResult result;
    result.advantages.resize(rt_count);
    result.returns.resize(rt_count);
    result.milliseconds = elapsed_ms(start, stop, iterations);
    check_cuda(cudaMemcpy(result.advantages.data(), d_adv, rt_count * sizeof(float), cudaMemcpyDeviceToHost), "copy advantages");
    check_cuda(cudaMemcpy(result.returns.data(), d_ret, rt_count * sizeof(float), cudaMemcpyDeviceToHost), "copy returns");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_rewards);
    cudaFree(d_values);
    cudaFree(d_dones);
    cudaFree(d_adv);
    cudaFree(d_ret);
    return result;
}

NormalizeResult run_normalize_host(const std::vector<float>& advantages, float eps, int iterations) {
    int M = static_cast<int>(advantages.size());
    float* d_adv = nullptr;
    cuda_alloc(&d_adv, advantages.size(), "cudaMalloc normalize");
    check_cuda(cudaMemcpy(d_adv, advantages.data(), advantages.size() * sizeof(float), cudaMemcpyHostToDevice), "copy normalize input");

    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    check_cuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < iterations; ++i) {
        launch_normalize_kernel(d_adv, M, eps);
    }
    check_kernel("launch normalize kernel");
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");

    NormalizeResult result;
    result.advantages.resize(advantages.size());
    result.milliseconds = elapsed_ms(start, stop, iterations);
    check_cuda(cudaMemcpy(result.advantages.data(), d_adv, advantages.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy normalize output");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_adv);
    return result;
}

GatherResult run_gather_host(
    const std::vector<float>& flat_buffer,
    const std::vector<int>& indices,
    int M,
    int D,
    int iterations) {
    int batch_size = static_cast<int>(indices.size());
    if (static_cast<int>(flat_buffer.size()) != M * D) {
        throw std::invalid_argument("run_gather_host received flat_buffer with unexpected size");
    }

    float *d_flat = nullptr, *d_out = nullptr;
    int* d_indices = nullptr;
    size_t flat_bytes = flat_buffer.size() * sizeof(float);
    size_t out_bytes = static_cast<size_t>(batch_size) * D * sizeof(float);
    cuda_alloc(&d_flat, flat_buffer.size(), "cudaMalloc gather flat");
    cuda_alloc(&d_indices, indices.size(), "cudaMalloc gather indices");
    cuda_alloc(&d_out, static_cast<size_t>(batch_size) * D, "cudaMalloc gather out");
    check_cuda(cudaMemcpy(d_flat, flat_buffer.data(), flat_bytes, cudaMemcpyHostToDevice), "copy gather flat");
    check_cuda(cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice), "copy gather indices");

    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    check_cuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < iterations; ++i) {
        launch_gather_minibatch(d_flat, d_indices, d_out, batch_size, D);
    }
    check_kernel("launch gather kernel");
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");

    GatherResult result;
    result.out.resize(static_cast<size_t>(batch_size) * D);
    result.milliseconds = elapsed_ms(start, stop, iterations);
    check_cuda(cudaMemcpy(result.out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost), "copy gather output");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_flat);
    cudaFree(d_indices);
    cudaFree(d_out);
    return result;
}

int get_device_peak_memory_bandwidth_gbps() {
    int device = 0;
    cudaDeviceProp prop{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    double gbps = 2.0 * static_cast<double>(prop.memoryClockRate) * 1000.0 *
                  (static_cast<double>(prop.memoryBusWidth) / 8.0) / 1.0e9;
    return static_cast<int>(gbps + 0.5);
}
