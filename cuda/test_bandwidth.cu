#include <stdio.h>
#include <cuda_runtime.h>

// 检查CUDA错误
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 向量化读写的核函数
__global__ void bandwidth_kernel_vec4(float4* dst, float4* src, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for(size_t i = idx; i < N; i += stride) {
        dst[i] = src[i];
    }
}

int main() {
    // 获取设备属性
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Theoretical Bandwidth: %.2f GB/s\n", 
           2.0 * prop.memoryClockRate * 1000.0 * 
           (prop.memoryBusWidth/8) / 1.0e9);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 测试不同的数据大小
    const size_t sizes[] = {
        1ULL << 26,  // 64 MB
        1ULL << 27,  // 128 MB
        1ULL << 28,   // 256 MB
        1ULL << 29,  // 512 MB
        1ULL << 30,  // 1024 MB
    };

    for (size_t size : sizes) {
        size_t num_elements = size / sizeof(float4); // 注意这里使用float4
        
        // 分配内存
        float *h_src, *h_dst;
        float4 *d_src, *d_dst;
        
        CHECK_CUDA_ERROR(cudaMallocHost(&h_src, size)); // 使用页锁定内存
        CHECK_CUDA_ERROR(cudaMallocHost(&h_dst, size));
        CHECK_CUDA_ERROR(cudaMalloc(&d_src, size));
        CHECK_CUDA_ERROR(cudaMalloc(&d_dst, size));

        // 初始化数据
        for(size_t i = 0; i < size/sizeof(float); i++) {
            h_src[i] = rand() / (float)RAND_MAX;
        }

        // 预热
        CHECK_CUDA_ERROR(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice));
        
        const int num_iterations = 100;
        float elapsed_time;

        // 测试H2D传输
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        for(int i = 0; i < num_iterations; i++) {
            CHECK_CUDA_ERROR(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice));
        }
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
        double h2d_bandwidth = size * num_iterations / (elapsed_time / 1000.0) / 1e9;

        // 测试D2H传输
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        for(int i = 0; i < num_iterations; i++) {
            CHECK_CUDA_ERROR(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost));
        }
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
        double d2h_bandwidth = size * num_iterations / (elapsed_time / 1000.0) / 1e9;

        // 测试设备内存带宽
        int blockSize = 256;
        int numBlocks = (num_elements + blockSize - 1) / blockSize;
        
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        for(int i = 0; i < num_iterations; i++) {
            bandwidth_kernel_vec4<<<numBlocks, blockSize>>>(d_dst, d_src, num_elements);
        }
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
        // 注意这里是2.0，因为我们既读又写
        double device_bandwidth = 2.0 * size * num_iterations / (elapsed_time / 1000.0) / 1e9;

        printf("\nData size: %zu MB\n", size / (1024 * 1024));
        printf("H2D Bandwidth: %.2f GB/s\n", h2d_bandwidth);
        printf("D2H Bandwidth: %.2f GB/s\n", d2h_bandwidth);
        printf("Device Memory Bandwidth: %.2f GB/s\n", device_bandwidth);

        // 清理内存
        CHECK_CUDA_ERROR(cudaFreeHost(h_src));
        CHECK_CUDA_ERROR(cudaFreeHost(h_dst));
        CHECK_CUDA_ERROR(cudaFree(d_src));
        CHECK_CUDA_ERROR(cudaFree(d_dst));
    }

    // 销毁CUDA事件
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}
