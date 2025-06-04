#include <stdio.h>
#include <stdint.h>

// 辅助函数：将32位整数位模式转换为浮点数
static inline float bits_to_float(uint32_t bits) {
    union {
        uint32_t as_bits;
        float as_value;
    } u;
    u.as_bits = bits;
    return u.as_value;
}

// 辅助函数：将浮点数转换为32位整数位模式
static inline uint32_t float_to_bits(float value) {
    union {
        float as_value;
        uint32_t as_bits;
    } u;
    u.as_value = value;
    return u.as_bits;
}

// 简化版的FP16到FP32转换
static float simple_fp16_to_fp32(uint16_t h) {
    // 1. 左移16位，准备转换
    const uint32_t w = (uint32_t)h << 16;
    // 2. 提取符号位
    const uint32_t sign = w & 0x80000000;
    // 3. 将w乘2
    const uint32_t two_w = w + w;
    
    // 4. 计算规格化值
    // 1110 0000 << 23 --> 0111 0000 0000 0000 0000 0000 0000 0000
    const uint32_t exp_offset = 0xE0 << 23;
    const float exp_scale = bits_to_float(0x7800000);
    float normalized = bits_to_float((two_w >> 4) + exp_offset) * exp_scale;
    
    // 5. 计算非规格化值
    const uint32_t magic_mask = 126 << 23;
    float denormalized = bits_to_float((two_w >> 17) | magic_mask) - 0.5f;
    
    // 6. 根据阈值选择返回值
    const uint32_t denorm_cutoff = 1 << 27;
    uint32_t result = sign | (two_w < denorm_cutoff ? float_to_bits(denormalized) : float_to_bits(normalized));
    return bits_to_float(result);
}

int main() {
    // 测试一些值
    uint16_t test = 0x3C00;  // 1.0 in FP16
    float result = simple_fp16_to_fp32(test);
    printf("Result: %f\n", result);
    return 0;
}

// #include <iostream>
// int main() {
//     float f32 = 3.14159f;
//     _Float16 f16 = (_Float16)f32;
//     float f32_converted = (double)f16;
//     std::cout << "Original 32-bit float: " << f32 << std::endl;
//     std::cout << "Converted back from 16-bit: " << f32_converted << std::endl;
    
//     std::cout << "Difference: " << f32 - f32_converted << std::endl;
//     std::cout << "Size of float: " << sizeof(float) << " bytes" << std::endl;
//     std::cout << "Size of _Float16: " << sizeof(_Float16) << " bytes" << std::endl;
//     return 0;
// }

// #include <stdio.h>
// #include <stdint.h>

// // 使用union来进行位模式的转换
// union Float32 {
//     uint32_t bits;
//     float value;
// };

// // 简化版的FP16到FP32转换
// float fp16_to_fp32(uint16_t fp16) {
//     // 提取符号位、指数位和尾数位
//     uint32_t sign = (fp16 & 0x8000) << 16;  // 符号位移到float的位置
//     uint32_t exp = (fp16 & 0x7C00) >> 10;   // 提取指数位
//     uint32_t mantissa = (fp16 & 0x03FF);     // 提取尾数位

//     union Float32 result;
    
//     if (exp == 0) {  // 处理零或非规格化数
//         result.bits = sign;
//     } else if (exp == 0x1F) {  // 处理无穷大或NaN
//         result.bits = sign | 0x7F800000 | (mantissa << 13);
//     } else {  // 处理规格化数
//         // 调整指数偏移（float的指数偏移是127，half的是15）
//         exp = exp - 15 + 127;
//         result.bits = sign | (exp << 23) | (mantissa << 13);
//     }

//     return result.value;
// }

// int main() {
//     // 测试一些典型值
//     uint16_t test_values[] = {
//         0x0000,  // 0
//         0x3C00,  // 1.0
//         0xBC00,  // -1.0
//         0x4000,  // 2.0
//     };

//     printf("FP16 to FP32 conversion examples:\n");
//     for (int i = 0; i < 4; i++) {
//         float f32 = fp16_to_fp32(test_values[i]);
//         printf("FP16: 0x%04X -> FP32: %f\n", test_values[i], f32);
//     }

//     return 0;
// }