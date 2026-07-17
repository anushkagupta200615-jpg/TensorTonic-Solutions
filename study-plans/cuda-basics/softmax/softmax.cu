#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
__shared__ float shared_max;
    __shared__ float shared_sum;

   
    if (threadIdx.x == 0) {
        float max_val = input[0];
        for (int i = 1; i < N; i++) {
            if (input[i] > max_val) max_val = input[i];
        }
        shared_max = max_val;
    }
    __syncthreads();

   
    if (threadIdx.x == 0) {
        float sum_val = 0.0f;
        for (int i = 0; i < N; i++) {
            sum_val += expf(input[i] - shared_max);
        }
        shared_sum = sum_val;
    }
    __syncthreads();

   
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        output[tid] = expf(input[tid] - shared_max) / shared_sum;
    }
}


extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}