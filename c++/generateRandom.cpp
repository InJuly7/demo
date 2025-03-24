#include <vector>
#include <random>
#include <iostream>
void generateRandomArray(float* arr, int N) {
    // 创建随机数生成器
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // 生成随机数
    for(int i = 0; i < N; i++) {
        arr[i] = dis(gen);
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

int main() {
    int N = 20;
    std::vector<float> h_a = generateRandomVector(N);
    // 打印结果
    for(int i = 0 ; i < N; i++) {
        std::cout << h_a[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
