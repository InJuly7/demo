#include <iostream>
#include <climits>
using namespace std;

int main() {
    cout << "=== C++ 类型转换示例 ===" << endl << endl;
    
    // 1. 非布尔类型转换为布尔类型
    cout << "1. 非布尔类型 -> 布尔类型:" << endl;
    bool b1 = 0;        // 0转换为false
    bool b2 = 42;       // 非0值转换为true
    bool b3 = -5;       // 非0值转换为true
    bool b4 = 0.0;      // 0.0转换为false
    bool b5 = 3.14;     // 非0值转换为true
    
    cout << "int 0 -> bool: " << b1 << endl;
    cout << "int 42 -> bool: " << b2 << endl;
    cout << "int -5 -> bool: " << b3 << endl;
    cout << "double 0.0 -> bool: " << b4 << endl;
    cout << "double 3.14 -> bool: " << b5 << endl << endl;
    
    // 2. 布尔类型转换为非布尔类型
    cout << "2. 布尔类型 -> 非布尔类型:" << endl;
    int i1 = true;      // true转换为1
    int i2 = false;     // false转换为0
    double d1 = true;   // true转换为1.0
    double d2 = false;  // false转换为0.0
    
    cout << "true -> int: " << i1 << endl;
    cout << "false -> int: " << i2 << endl;
    cout << "true -> double: " << d1 << endl;
    cout << "false -> double: " << d2 << endl << endl;
    
    // 3. 浮点数转换为整数类型
    cout << "3. 浮点数 -> 整数类型 (截断小数部分):" << endl;
    double pi = 3.14159;
    double neg_pi = -3.14159;
    int int_pi = pi;           // 3.14159 -> 3
    int int_neg_pi = neg_pi;   // -3.14159 -> -3
    
    cout << "double 3.14159 -> int: " << int_pi << endl;
    cout << "double -3.14159 -> int: " << int_neg_pi << endl;
    
    float f = 9.99f;
    int int_f = f;             // 9.99 -> 9
    cout << "float 9.99 -> int: " << int_f << endl << endl;
    
    // 4. 整数转换为浮点类型
    cout << "4. 整数 -> 浮点类型:" << endl;
    int large_int = 123456789;
    float float_large = large_int;  // 可能有精度损失
    double double_large = large_int; // 通常无精度损失
    
    cout << "int " << large_int << " -> float: " << float_large << endl;
    cout << "int " << large_int << " -> double: " << double_large << endl << endl;
    
    // 5. 无符号类型的溢出 (取模运算)
    cout << "5. 无符号类型溢出 (取模运算):" << endl;
    unsigned char uc1 = 255;    // 最大值
    unsigned char uc2 = 256;    // 256 % 256 = 0
    unsigned char uc3 = 257;    // 257 % 256 = 1
    unsigned char uc4 = -1;     // -1对256取模 = 255
    
    cout << "unsigned char = 255: " << (int)uc1 << endl;
    cout << "unsigned char = 256: " << (int)uc2 << endl;
    cout << "unsigned char = 257: " << (int)uc3 << endl;
    cout << "unsigned char = -1: " << (int)uc4 << endl << endl;
    
    // 6. 带符号类型的溢出 (未定义行为)
    cout << "6. 带符号类型溢出 (未定义行为 - 结果可能不可预测):" << endl;
    cout << "signed char 最大值: " << (int)SCHAR_MAX << endl;
    cout << "signed char 最小值: " << (int)SCHAR_MIN << endl;
    
    signed char sc1 = SCHAR_MAX;
    signed char sc2 = SCHAR_MAX + 1;  // 未定义行为!
    cout << "signed char = SCHAR_MAX: " << (int)sc1 << endl;
    cout << "signed char = SCHAR_MAX + 1 (未定义): " << (int)sc2 << endl;
    
    // 警告: 以下代码可能产生未定义行为
    signed char sc3 = 200;  // 如果signed char范围是-128到127，这是未定义的
    cout << "signed char = 200 (可能未定义): " << (int)sc3 << endl << endl;
    
    // 7. 显式类型转换
    cout << "7. 显式类型转换 (强制转换):" << endl;
    double exact_pi = 3.14159265359;
    int truncated_pi = static_cast<int>(exact_pi);
    
    cout << "使用 static_cast<int> 显式转换:" << endl;
    cout << "double " << exact_pi << " -> int " << truncated_pi << endl;
    
    return 0;
}
