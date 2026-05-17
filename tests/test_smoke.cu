#include "ppo_kernels.h"

#include <cuda_runtime.h>

#include <cstdio>

int main() {
    const int N = 1;
    const int T = 4;
    float h_rewards[N * T] = {1.0f, 1.0f, 1.0f, 1.0f};
    float h_values[N * (T + 1)] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f};
    float h_dones[N * T] = {0.0f, 0.0f, 0.0f, 1.0f};
    float h_adv[N * T] = {};
    float h_ret[N * T] = {};

    float *d_rewards = nullptr, *d_values = nullptr, *d_dones = nullptr, *d_adv = nullptr, *d_ret = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_rewards), sizeof(h_rewards));
    cudaMalloc(reinterpret_cast<void**>(&d_values), sizeof(h_values));
    cudaMalloc(reinterpret_cast<void**>(&d_dones), sizeof(h_dones));
    cudaMalloc(reinterpret_cast<void**>(&d_adv), sizeof(h_adv));
    cudaMalloc(reinterpret_cast<void**>(&d_ret), sizeof(h_ret));
    cudaMemcpy(d_rewards, h_rewards, sizeof(h_rewards), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, sizeof(h_values), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dones, h_dones, sizeof(h_dones), cudaMemcpyHostToDevice);

    launch_gae_kernel(d_rewards, d_values, d_dones, d_adv, d_ret, N, T, 0.99f, 0.95f);
    cudaDeviceSynchronize();
    cudaMemcpy(h_adv, d_adv, sizeof(h_adv), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ret, d_ret, sizeof(h_ret), cudaMemcpyDeviceToHost);

    std::printf("adv[0]=%.6f ret[0]=%.6f\n", h_adv[0], h_ret[0]);
    cudaFree(d_rewards);
    cudaFree(d_values);
    cudaFree(d_dones);
    cudaFree(d_adv);
    cudaFree(d_ret);
    return 0;
}
