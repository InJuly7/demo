#include <stdio.h>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    checkCudaError(error, "cudaGetDeviceCount");

    printf("找到 %d 个CUDA设备\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, dev);
        checkCudaError(error, "cudaGetDeviceProperties");

        printf("设备 %d: \"%s\"\n", dev, deviceProp.name);
        
        // 计算能力
        printf("  CUDA计算能力: %d.%d\n", deviceProp.major, deviceProp.minor);
        
        // 基本硬件信息
        printf("  全局内存总量: %.2f GB\n", deviceProp.totalGlobalMem / (float)1073741824);
        printf("  GPU时钟频率: %.0f MHz\n", deviceProp.clockRate * 1e-3f);
        printf("  内存时钟频率: %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("  内存总线位宽: %d bits\n", deviceProp.memoryBusWidth);
        printf("  L2缓存大小: %d KB\n", deviceProp.l2CacheSize / 1024);
        
        // 多处理器信息
        printf("  多处理器数量: %d\n", deviceProp.multiProcessorCount);
        printf("  每个多处理器的最大线程数: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  每个块的最大线程数: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  每个线程块的最大维度: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  网格的最大维度: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        
        // 内存信息
        printf("  总常量内存: %zu bytes\n", deviceProp.totalConstMem);
        printf("  共享内存每个块: %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  每个块的32位寄存器数量: %d\n", deviceProp.regsPerBlock);
        printf("  Warp大小: %d\n", deviceProp.warpSize);
        
        // 其他特性
        printf("  统一内存支持: %s\n", deviceProp.unifiedAddressing ? "是" : "否");
        printf("  并发核函数支持: %s\n", deviceProp.concurrentKernels ? "是" : "否");
        printf("  异步引擎数量: %d\n", deviceProp.asyncEngineCount);
        printf("  ECC内存支持: %s\n", deviceProp.ECCEnabled ? "是" : "否");
        printf("  设备重叠支持: %s\n", deviceProp.deviceOverlap ? "是" : "否");
        printf("  内核执行超时: %s\n", deviceProp.kernelExecTimeoutEnabled ? "是" : "否");
        printf("  集成GPU: %s\n", deviceProp.integrated ? "是" : "否");
        printf("  可以映射主机内存: %s\n", deviceProp.canMapHostMemory ? "是" : "否");
        printf("  计算模式: %d\n", deviceProp.computeMode);
        
        // PCI信息
        printf("  PCI总线ID: %d\n", deviceProp.pciBusID);
        printf("  PCI设备ID: %d\n", deviceProp.pciDeviceID);
        
        // 纹理对齐
        printf("  纹理对齐: %zu bytes\n", deviceProp.textureAlignment);
        
        printf("\n");
    }

    // 获取当前设备的内存信息
    size_t free, total;
    error = cudaMemGetInfo(&free, &total);
    checkCudaError(error, "cudaMemGetInfo");
    
    printf("当前设备内存使用情况:\n");
    printf("  总内存: %.2f GB\n", total / (1024.0 * 1024.0 * 1024.0));
    printf("  可用内存: %.2f GB\n", free / (1024.0 * 1024.0 * 1024.0));
    printf("  已用内存: %.2f GB\n", (total - free) / (1024.0 * 1024.0 * 1024.0));

    return 0;
}
