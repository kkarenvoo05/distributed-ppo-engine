#include "ppo_kernels.h"

#include <stdexcept>

namespace {

__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void reduce_sum_sumsq_kernel(
    const float* __restrict__ x,
    int M,
    float* __restrict__ out_sum,
    float* __restrict__ out_sumsq) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = idx; i < M; i += stride) {
        float v = x[i];
        sum += v;
        sumsq += v * v;
    }

    sum = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);

    __shared__ float warp_sums[32];
    __shared__ float warp_sumsq[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    int warps_per_block = (blockDim.x + 31) / 32;

    if (lane == 0) {
        warp_sums[warp] = sum;
        warp_sumsq[warp] = sumsq;
    }
    __syncthreads();

    if (warp == 0) {
        sum = (lane < warps_per_block) ? warp_sums[lane] : 0.0f;
        sumsq = (lane < warps_per_block) ? warp_sumsq[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        sumsq = warp_reduce_sum(sumsq);
        if (lane == 0) {
            atomicAdd(out_sum, sum);
            atomicAdd(out_sumsq, sumsq);
        }
    }
}

__global__ void normalize_kernel_impl(
    float* __restrict__ x,
    int M,
    const float* __restrict__ sum,
    const float* __restrict__ sumsq,
    float eps) {
    float mean = sum[0] / static_cast<float>(M);
    float variance = sumsq[0] / static_cast<float>(M) - mean * mean;
    variance = fmaxf(variance, 0.0f);
    float inv_std = 1.0f / (sqrtf(variance) + eps);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < M; i += stride) {
        x[i] = (x[i] - mean) * inv_std;
    }
}

}  // namespace

void launch_normalize_kernel(
    float* advantages,
    int M,
    float eps,
    cudaStream_t stream) {
    if (M <= 0) {
        throw std::invalid_argument("launch_normalize_kernel expects M > 0");
    }

    float* d_sum = nullptr;
    float* d_sumsq = nullptr;
    cudaMallocAsync(reinterpret_cast<void**>(&d_sum), sizeof(float), stream);
    cudaMallocAsync(reinterpret_cast<void**>(&d_sumsq), sizeof(float), stream);
    cudaMemsetAsync(d_sum, 0, sizeof(float), stream);
    cudaMemsetAsync(d_sumsq, 0, sizeof(float), stream);

    cudaDeviceProp prop{};
    int device = 0;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int block = 256;
    int grid = prop.multiProcessorCount > 0 ? prop.multiProcessorCount * 4 : 128;
    reduce_sum_sumsq_kernel<<<grid, block, 0, stream>>>(advantages, M, d_sum, d_sumsq);
    normalize_kernel_impl<<<grid, block, 0, stream>>>(advantages, M, d_sum, d_sumsq, eps);

    cudaFreeAsync(d_sum, stream);
    cudaFreeAsync(d_sumsq, stream);
}
