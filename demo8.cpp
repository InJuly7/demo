#if 1
#include <iostream>
using namespace std;

int main() {
    // 普通变量
    int num = 42;
    
    // 指针声明和初始化
    int* ptr = &num;  // ptr指向num的地址
    
    cout << "变量num的值: " << num << endl;
    cout << "变量num的地址: " << &num << endl;
    cout << "指针ptr的值(存储的地址): " << ptr << endl;
    cout << "指针ptr指向的值: " << *ptr << endl;
    cout << "指针ptr本身的地址: " << &ptr << endl;
    
    // 通过指针修改值
    *ptr = 100;
    cout << "通过指针修改后,num的值: " << num << endl;
    
    return 0;
}
#endif