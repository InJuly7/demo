#include <iostream>
#include <utility>  // 包含 pair 的头文件

int main() {
    // 创建不同 pair 实例
    std::pair<int, int> a = {0, 5};    // 第一个元素 0，第二个元素 5
    std::pair<int, int> b = {1, 3};    // 第一个元素 1，第二个元素 3
    std::pair<int, int> c = {1, 4};    // 第一个元素 1，第二个元素 4
    std::pair<int, int> d = {0, 3};    // 第一个元素 0，第二个元素 3
    std::pair<int, int> e = {1, 3};    // 与 b 相同

    // 显示布尔值为 true/false 而不是 1/0
    std::cout << std::boolalpha;

    // 比较不同情况并输出结果
    std::cout << "比较示例：\n\n";

    // 情况 1：第一个元素不同
    std::cout << "a (0,5) < b (1,3)? " << (a < b)  << " → 比较结果由第一个元素 (0 < 1) 决定\n";
    std::cout << "b (1,3) < a (0,5)? " << (b < a)  << " → 比较结果由第一个元素 (1 > 0) 决定\n\n";

    // 情况 2：第一个元素相同，第二个元素不同
    std::cout << "b (1,3) < c (1,4)? " << (b < c)  << " → 第一个元素相等时，比较第二个元素 (3 < 4)\n";
    std::cout << "c (1,4) < b (1,3)? " << (c < b)  << " \n\n";

    // 情况 3：第一个和第二个元素均相等
    std::cout << "b (1,3) < e (等同于 b)? " << (b < e) << " → 相等的 pair 不存在小于关系\n\n";

    // 情况 4：边界比较（如示例中 0 和 1 的比较）
    std::cout << "d (0,3) < a (0,5)? " << (d < a) << " → 第一个元素相等时，比较第二个元素 (3 < 5)\n";
    
    return 0;
}