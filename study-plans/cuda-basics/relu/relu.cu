#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<N){
        float x=input[t];
        output[t]=x > 0.0f ? x :0.0f;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}