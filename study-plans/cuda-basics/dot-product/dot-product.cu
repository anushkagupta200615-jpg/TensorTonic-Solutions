#include <cuda_runtime.h>

__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    extern __shared__ float sdata[];  // shared memory buffer

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Each thread computes its product
    float val = 0.0f;
    if (idx < N) {
        val = A[idx] * B[idx];
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

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Initialize result to 0
    cudaMemset(result, 0, sizeof(float));

    // Launch kernel with dynamic shared memory
    dot_kernel<<<blocks, threads, threads * sizeof(float)>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
