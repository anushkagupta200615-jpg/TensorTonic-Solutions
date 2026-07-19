#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    extern __shared__ float sdata[];  // shared memory buffer

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Each thread loads one element
    float val = 0.0f;
    if (idx < N) {
        val = input[idx];
    }
    sdata[tid] = val;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // First thread in each block contributes its block sum
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Initialize result to 0
    cudaMemset(result, 0, sizeof(float));

    // Launch kernel with dynamic shared memory
    sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(input, result, N);
    cudaDeviceSynchronize();
}
