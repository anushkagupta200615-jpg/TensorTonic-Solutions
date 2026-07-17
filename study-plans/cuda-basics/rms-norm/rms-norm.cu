#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sumsq = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float v = input[row * N + j];
        sumsq += v * v;
    }
    shared[tid] = sumsq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float rms;
    if (tid == 0) {
        rms = sqrtf(shared[0] / N + eps);
        shared[0] = rms;
    }
    __syncthreads();
    rms = shared[0];

    for (int j = tid; j < N; j += blockDim.x) {
        float v = input[row * N + j];
        output[row * N + j] = (v / rms) * gamma[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    size_t shmem = threads * sizeof(float);
    rms_norm_kernel<<<blocks, threads, shmem>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}
