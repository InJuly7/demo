#if 0
// Demo1 - 顶层const：指针本身是常量
#include <iostream>
using namespace std;

int main () {
    cout << "\n=== Demo1: 顶层const示例 ===" << endl;
    
    int a = 10, b = 20;
    
    // 顶层const：指针本身是常量，不能改变指向
    int* const ptr1 = &a;  // ptr1是常量指针
    
    cout << "初始值: *ptr1 = " << *ptr1 << endl;
    
    // 可以修改指针指向的值
    *ptr1 = 15;
    cout << "修改后: *ptr1 = " << *ptr1 << endl;
    
    // 错误！不能修改指针本身的指向
    // ptr1 = &b;  // 编译错误：assignment of read-only variable 'ptr1'
    
    cout << "顶层const：指针本身不可变，但指向的值可变" << endl;
}
#endif

#if 0
// Demo2 - 底层const：指针指向的对象是常量
#include <iostream>
using namespace std;

int main() {
    cout << "\n=== Demo2: 底层const示例 ===" << endl;
    
    int a = 10, b = 20;
    
    // 底层const：指针指向的对象是常量
    const int* ptr2 = &a;  // ptr2指向常量int
    
    cout << "初始值: *ptr2 = " << *ptr2 << endl;
    
    // 可以修改指针的指向
    ptr2 = &b;
    cout << "改变指向后: *ptr2 = " << *ptr2 << endl;
    
    // 错误！不能修改指针指向的值
    // *ptr2 = 25;  // 编译错误：assignment of read-only location
    
    cout << "底层const：指针可以改变指向，但不能修改指向的值" << endl;
}
#endif

#if 0
// Demo3 - 同时具有顶层和底层const
#include <iostream>
using namespace std;

int main() {
    cout << "\n=== Demo3: 同时具有顶层和底层const ===" << endl;
    
    int a = 10, b = 20;
    
    // 既有顶层const又有底层const
    const int* const ptr3 = &a;  // ptr3是指向常量int的常量指针
    
    cout << "初始值: *ptr3 = " << *ptr3 << endl;
    
    // 错误！不能修改指针指向的值（底层const）
    // *ptr3 = 25;  // 编译错误
    
    // 错误！不能修改指针的指向（顶层const）
    // ptr3 = &b;   // 编译错误
    
    cout << "同时具有顶层和底层const：指针和指向的值都不可变" << endl;
}
#endif

#if 0
// Demo4 - 不同数据类型的顶层const
#include <iostream>
using namespace std;

int main() {
    cout << "\n=== Demo4: 各种数据类型的顶层const ===" << endl;
    
    // 算术类型的顶层const
    const int num = 42;
    cout << "常量整数: " << num << endl;
    // num = 50;  // 错误！
    
    // 指针的顶层const
    int value = 100;
    int* const ptr = &value;
    cout << "常量指针指向的值: " << *ptr << endl;
    
    // 对象的顶层const
    const string str = "Hello";
    cout << "常量字符串: " << str << endl;
    // str = "World";  // 错误！
    
    cout << "顶层const适用于任何数据类型" << endl;
}
#endif

#if 1
// Demo5 - const的复制和传递规则
#include <iostream>
using namespace std;

int main() {
    cout << "\n=== Demo5: const的复制和传递规则 ===" << endl;
    
    int a = 10;
    const int b = 20;
    
    // 顶层const在复制时会被忽略
    const int* const ptr1 = &a;  // 顶层+底层const
    const int* ptr2 = ptr1;      // 只保留底层const，顶层const被忽略
    
    cout << "原指针: " << *ptr1 << endl;
    cout << "复制的指针: " << *ptr2 << endl;
    
    // 底层const在复制时必须保留
    int* ptr3;
    // ptr3 = ptr1;  // 错误！不能将底层const赋给非const
    
    // 正确的做法
    ptr2 = &b;  // 可以，保持底层const
    cout << "底层const必须在复制时保留" << endl;
}
#endif
