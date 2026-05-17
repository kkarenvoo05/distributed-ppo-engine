#include "ppo_kernels.h"

#include <stdexcept>

__global__ void gather_minibatch_kernel_impl(
    const float* __restrict__ flat_buffer,
    const int* __restrict__ indices,
    float* __restrict__ out,
    int batch_size,
    int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (b < batch_size && d < D) {
        out[b * D + d] = flat_buffer[indices[b] * D + d];
    }
}

void launch_gather_minibatch(
    const float* flat_buffer,
    const int* indices,
    float* out,
    int batch_size,
    int D,
    cudaStream_t stream) {
    if (batch_size <= 0 || D <= 0) {
        throw std::invalid_argument("launch_gather_minibatch expects batch_size > 0 and D > 0");
    }
    dim3 block(32, 8);
    dim3 grid((D + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    gather_minibatch_kernel_impl<<<grid, block, 0, stream>>>(flat_buffer, indices, out, batch_size, D);
}

