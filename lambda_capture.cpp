#include <iostream>
#include <vector>
#include <string>
#include <utility>  // 包含 pair 的头文件

int main() {
    // 创建一个 std::vector<std::string>
    std::vector<std::string> nums = {"zero", "one", "two", "three", "four"};
    int n = nums.size();  // n = 5
    int temp = 100;
    // 定义 lambda 函数 get，[&] 按引用捕获外部变量 : n,nums
    // auto get = [&](int i) -> std::pair<int, std::string> {
    //     temp++;
    //     if (i == -1) {
    //         return {0, "Left border"};
    //     } else if(i == n) {
    //         return {0, "Right border"};
    //     }
    //     return {1, nums[i]};  // 返回 {1, nums[i]}
    // };
 
    auto get = [&n, &nums,&temp](int i) -> std::pair<int, std::string> {
        temp++;
        if (i == -1) {
            return {0, "Left border"};
        } else if(i == n) {
            return {0, "Right border"};
        }
        return {1, nums[i]};  // 返回 {1, nums[i]}
    };

    // 测试不同的索引情况
    std::cout << std::boolalpha;  // 输出布尔值为 true/false
    std::cout << "temp = " << temp << std::endl;
    // 情况 1：有效索引
    std::cout << "索引 2 的值：";
    auto result1 = get(2);
    std::cout << "{ " << result1.first << ", " << result1.second << " }\n";  // 输出 {1, "three"}
    std::cout << "temp = " << temp << std::endl;
    // 情况 2：无效索引 (-1)
    std::cout << "索引 -1 的值：";
    auto result2 = get(-1);
    std::cout << "{ " << result2.first << ", " << result2.second << " }\n";  // 输出 {0, ""}
    std::cout << "temp = " << temp << std::endl;
    // 情况 3：无效索引 (n)
    std::cout << "索引 5 的值：";
    auto result3 = get(5);
    std::cout << "{ " << result3.first << ", " << result3.second << " }\n";  // 输出 {0, ""}
    std::cout << "temp = " << temp << std::endl;
    // 情况 4：修改外部变量 nums
    std::cout << "\n修改 nums[1] 为 \"TWO\"\n";
    nums[1] = "TWO";  // 修改外部 vector

    std::cout << "索引 1 的值：";
    auto result4 = get(1);
    std::cout << "{ " << result4.first << ", " << result4.second << " }\n";  // 输出 {1, "TWO"}
    std::cout << "temp = " << temp << std::endl;

    return 0;
}
