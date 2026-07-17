#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
   int t=blockIdx.x*blockDim.x+threadIdx.x;
   if(t<N){
       float(x)=input[t];
        output[t] = 0.5f * x * (1.0f + erf(x / sqrtf(2.0f)));
   } 
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
