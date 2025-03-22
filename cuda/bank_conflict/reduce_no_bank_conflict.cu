#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#define THREAD_PER_BLOCK 256
static int checkCudaError(cudaError_t code, const char *expr, const char *file, int line)
{
    if (code)
    {
        printf("CUDA error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudaGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

#define checkCudaErr(...)                                                        \
    do                                                                           \
    {                                                                            \
        int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);                                                                       \
    } while (0)

// idle Threads
__global__ void reduce2(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

// bank conflict
__global__ void reduce1(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0){
        d_out[blockIdx.x] = sdata[0];
    }

}

std::vector<float> generateRandomVector(int N) {
    std::vector<float> vec(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for(int i = 0; i < N; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

float reduce(float* h_a, int N) {
    float result = 0;
    for(int i=0;i<N;i++){
        result+=h_a[i];
    }
    return result;  
}

int main(int agrc, char **argv) {
    int N = 32*1024*1024;
    std::vector<float> h_a = generateRandomVector(N);
    float *d_a;
    checkCudaErr(cudaMalloc((void **)&d_a, N*sizeof(float)));
    float cpu_result = reduce(h_a.data(),N);
    std::vector<float> h_c(N/THREAD_PER_BLOCK);
    float *d_c;
    checkCudaErr(cudaMalloc((void **)&d_c,(N/THREAD_PER_BLOCK)*sizeof(float)));
    checkCudaErr(cudaMemcpy(d_a, h_a.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    dim3 Grid( N/THREAD_PER_BLOCK,1);
    dim3 Block( THREAD_PER_BLOCK,1);
    if(std::string(argv[1])== "reduce1") {
        reduce1<<<Grid,Block>>>(d_a,d_c);
    } else if(std::string(argv[1]) == "reduce2") {
        reduce2<<<Grid,Block>>>(d_a,d_c);
    }
    checkCudaErr(cudaGetLastError());
    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(h_c.data(), d_c, (N/THREAD_PER_BLOCK)*sizeof(float), cudaMemcpyDeviceToHost));
    float gpu_result = reduce(h_c.data(),N/THREAD_PER_BLOCK);
    std::cout<<"cpu result: "<<cpu_result<<std::endl;
    std::cout<<"gpu result: "<<gpu_result<<std::endl;
    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}