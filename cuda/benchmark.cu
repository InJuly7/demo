#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA kernel for FP32
__global__ void benchmark_fp32(float* a, float* b, float* c, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < iterations; i++) {
            sum = a[idx] * b[idx] + sum;
        }
        c[idx] = sum;
    }
}

// CUDA kernel for FP16
__global__ void benchmark_fp16(__half* a, __half* b, __half* c, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half sum = __float2half(0.0f);
        for (int i = 0; i < iterations; i++) {
            sum = __hadd(__hmul(a[idx], b[idx]), sum);
        }
        c[idx] = sum;
    }
}

int main() {
    const int N = 1 << 24;  // 数组大小
    const int iterations = 1000;  // 迭代次数
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // FP32 测试
    {
        float *a, *b, *c;
        float *d_a, *d_b, *d_c;

        // 分配主机内存
        a = (float*)malloc(N * sizeof(float));
        b = (float*)malloc(N * sizeof(float));
        c = (float*)malloc(N * sizeof(float));

        // 初始化数据
        for (int i = 0; i < N; i++) {
            a[i] = 1.0f;
            b[i] = 2.0f;
        }

        // 分配设备内存
        cudaMalloc(&d_a, N * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_c, N * sizeof(float));

        // 复制数据到设备
        cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

        // 记录时间
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        benchmark_fp32<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N, iterations);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // 计算 FLOPS
        double operations = (double)N * iterations * 2; // 每次迭代两个操作（乘和加）
        double gigaFlops = (operations / (milliseconds / 1000.0)) / 1e9;

        printf("FP32 Performance: %.2f GFLOPS\n", gigaFlops);

        // 清理
        free(a);
        free(b);
        free(c);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // FP16 测试
    {
        __half *a, *b, *c;
        __half *d_a, *d_b, *d_c;

        // 分配主机内存
        a = (__half*)malloc(N * sizeof(__half));
        b = (__half*)malloc(N * sizeof(__half));
        c = (__half*)malloc(N * sizeof(__half));

        // 初始化数据
        for (int i = 0; i < N; i++) {
            a[i] = __float2half(1.0f);
            b[i] = __float2half(2.0f);
        }

        // 分配设备内存
        cudaMalloc(&d_a, N * sizeof(__half));
        cudaMalloc(&d_b, N * sizeof(__half));
        cudaMalloc(&d_c, N * sizeof(__half));

        // 复制数据到设备
        cudaMemcpy(d_a, a, N * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, N * sizeof(__half), cudaMemcpyHostToDevice);

        // 记录时间
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        benchmark_fp16<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N, iterations);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // 计算 FLOPS
        double operations = (double)N * iterations * 2;
        double gigaFlops = (operations / (milliseconds / 1000.0)) / 1e9;

        printf("FP16 Performance: %.2f GFLOPS\n", gigaFlops);

        // 清理
        free(a);
        free(b);
        free(c);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
