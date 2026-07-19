#include <cuda_runtime.h>
#include <math.h>

__global__ void l1_normalize_kernel(const float* input, float* output, int N, float* denom) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    if (idx < N) val = fabsf(input[idx]);
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(denom, sdata[0]);
}

__global__ void normalize_kernel(const float* input, float* output, int N, float denom) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) output[idx] = input[idx] / denom;
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* denom_dev;
    cudaMalloc(&denom_dev, sizeof(float));
    cudaMemset(denom_dev, 0, sizeof(float));

    l1_normalize_kernel<<<blocks, threads, threads * sizeof(float)>>>(input, output, N, denom_dev);
    cudaDeviceSynchronize();

    float denom;
    cudaMemcpy(&denom, denom_dev, sizeof(float), cudaMemcpyDeviceToHost);

    normalize_kernel<<<blocks, threads>>>(input, output, N, denom);
    cudaDeviceSynchronize();

    cudaFree(denom_dev);
}
