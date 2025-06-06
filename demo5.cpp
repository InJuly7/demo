#include <iostream>
using namespace std;

void show_data_types_size()
{
    cout << "=== C++ 基本数据类型大小演示 ===" << endl;
    
    // 整型系列
    cout << "\n--- 整型系列 ---" << endl;
    cout << "int 大小: " << sizeof(int) << " 字节" << endl;
    cout << "short 大小: " << sizeof(short) << " 字节" << endl; 
    cout << "long 大小: " << sizeof(long) << " 字节" << endl;
    cout << "long long 大小: " << sizeof(long long) << " 字节" << endl;
    
    // 字符型系列
    cout << "\n--- 字符型系列 ---" << endl;
    cout << "char 大小: " << sizeof(char) << " 字节" << endl;
    cout << "signed char 大小: " << sizeof(signed char) << " 字节" << endl;
    cout << "unsigned char 大小: " << sizeof(unsigned char) << " 字节" << endl;
    
    // 浮点型系列
    cout << "\n--- 浮点型系列 ---" << endl;
    cout << "float 大小: " << sizeof(float) << " 字节" << endl;
    cout << "double 大小: " << sizeof(double) << " 字节" << endl;
    cout << "long double 大小: " << sizeof(long double) << " 字节" << endl;
    
    // 布尔型
    cout << "\n--- 其他类型 ---" << endl;
    cout << "bool 大小: " << sizeof(bool) << " 字节" << endl;
    
    // 实际变量的大小
    cout << "\n--- 变量实例大小 ---" << endl;
    int a = 100;
    long long lla = 1000;
    short sa = 200;
    char sc = 'a';
    bool bt = true;
    
    cout << "变量 a(int) 大小: " << sizeof(a) << " 字节" << endl;
    cout << "变量 lla(long long) 大小: " << sizeof(lla) << " 字节" << endl;
    cout << "变量 sa(short) 大小: " << sizeof(sa) << " 字节" << endl;
    cout << "变量 sc(char) 大小: " << sizeof(sc) << " 字节" << endl;
    cout << "变量 bt(bool) 大小: " << sizeof(bt) << " 字节" << endl;
}

int main()
{
    show_data_types_size();
    return 0;
}
