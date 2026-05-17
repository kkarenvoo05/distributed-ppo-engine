#include "ppo_kernels.h"

#include <stdexcept>

__global__ void gae_kernel_impl(
    const float* __restrict__ rewards,
    const float* __restrict__ values,
    const float* __restrict__ dones,
    float* __restrict__ advantages,
    float* __restrict__ returns,
    int N_envs,
    int T,
    float gamma,
    float lam) {
    int env = blockIdx.x;
    int tid = threadIdx.x;
    if (env >= N_envs) {
        return;
    }

    extern __shared__ float smem[];
    float* s_rewards = smem;
    float* s_values = s_rewards + T;
    float* s_dones = s_values + T + 1;

    if (tid < T) {
        int rt = env * T + tid;
        s_rewards[tid] = rewards[rt];
        s_values[tid] = values[env * (T + 1) + tid];
        s_dones[tid] = dones[rt];
    }
    if (tid == 0) {
        s_values[T] = values[env * (T + 1) + T];
    }

    __syncthreads();

    if (tid == 0) {
        float gae = 0.0f;
        for (int t = T - 1; t >= 0; --t) {
            float mask = 1.0f - s_dones[t];
            float delta = s_rewards[t] + gamma * s_values[t + 1] * mask - s_values[t];
            gae = delta + gamma * lam * mask * gae;
            int out_idx = env * T + t;
            advantages[out_idx] = gae;
            returns[out_idx] = gae + s_values[t];
        }
    }
}

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
    cudaStream_t stream) {
    if (N_envs <= 0 || T <= 0 || T > 1024) {
        throw std::invalid_argument("launch_gae_kernel expects N_envs > 0 and 0 < T <= 1024");
    }
    dim3 grid(N_envs);
    dim3 block(T);
    size_t shared_bytes = static_cast<size_t>(3 * T + 1) * sizeof(float);
    gae_kernel_impl<<<grid, block, shared_bytes, stream>>>(
        rewards, values, dones, advantages, returns, N_envs, T, gamma, lam);
}

