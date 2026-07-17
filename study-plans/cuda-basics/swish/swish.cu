#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* input, float* output, int N) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<N){
          float x = input[t];
        float sigma = 1.0f / (1.0f + expf(-x));
          output[t] = x * sigma;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
