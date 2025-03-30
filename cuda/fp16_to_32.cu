#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void printFP16Values(const __half* input) {
    int idx = threadIdx.x;
    printf("Thread %d: FP16 value = %f\n", idx, __half2float(input[idx]));
}

int main() {
    const int size = 4;
    float h_data[size] = {1.5f, 2.75f, -3.25f, 4.0f};
    __half* h_fp16 = new __half[size];

    for(int i = 0; i < size; i++) {
        h_fp16[i] = __float2half(h_data[i]);
    }

    __half* d_fp16;
    cudaMalloc(&d_fp16, size * sizeof(__half));
    cudaMemcpy(d_fp16, h_fp16, size * sizeof(__half), cudaMemcpyHostToDevice);

    printFP16Values<<<1, size>>>(d_fp16);
    cudaDeviceSynchronize();

    delete[] h_fp16;
    cudaFree(d_fp16);
    return 0;
}
