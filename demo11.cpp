#include <iostream>
using namespace std;

// Demo1 - 算术类型字面值示例
#if 0
int main() {
    cout << "=== Demo1 - 算术类型字面值示例 ===" << endl;
    
    // 整数字面值
    constexpr int a = 42;           // int字面值
    constexpr long b = 100L;        // long字面值
    constexpr unsigned c = 50u;     // unsigned字面值
    
    // 浮点字面值
    constexpr double d = 3.14;      // double字面值
    constexpr float e = 2.5f;       // float字面值
    
    // 字符字面值
    constexpr char ch = 'A';        // char字面值
    constexpr bool flag = true;     // bool字面值
    
    cout << "int: " << a << endl;
    cout << "long: " << b << endl;
    cout << "unsigned: " << c << endl;
    cout << "double: " << d << endl;
    cout << "float: " << e << endl;
    cout << "char: " << ch << endl;
    cout << "bool: " << flag << endl;
    
    return 0;
}
#endif

// Demo2 - 指针和引用字面值示例
#if 0
constexpr int value = 100; // 全局变量
int main() {
    cout << "=== Demo2 - 指针和引用字面值示例 ===" << endl;
    
    // 空指针字面值
    constexpr int* ptr1 = nullptr;
    constexpr char* ptr2 = nullptr;
    
    // 指向字面值的指针
    constexpr const int* ptr3 = &value;  // 指向constexpr对象的指针
    
    // 引用（必须绑定到对象）
    constexpr const int& ref = value;    // 引用字面值对象
    
    cout << "nullptr指针可以用于constexpr" << endl;
    cout << "value: " << value << endl;
    cout << "通过指针访问: " << *ptr3 << endl;
    cout << "通过引用访问: " << ref << endl;
    
    return 0;
}
#endif

// Demo3 - constexpr函数与字面值类型示例
#if 1
// constexpr函数必须使用字面值类型
constexpr int square(int x) {
    return x * x;  // 算术运算，返回字面值类型
}

constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

int main() {
    cout << "=== Demo3 - constexpr与字面值类型示例 ===" << endl;
    
    // 编译期计算
    constexpr int num = 5;
    constexpr int result1 = square(num);        // 编译期计算
    constexpr int result2 = factorial(4);       // 编译期递归计算
    
    cout << "square(" << num << ") = " << result1 << endl;
    cout << "factorial(4) = " << result2 << endl;
    
    // 数组大小必须是常量表达式（字面值类型）
    constexpr int size = 10;
    int arr[size];  // 合法，size是constexpr
    cout << "数组大小: " << size << endl;
    
    // 字面值类型可以在模板参数中使用
    constexpr int template_param = 42;
    cout << "模板参数: " << template_param << endl;
    
    return 0;
}
#endif
