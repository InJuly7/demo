#if 0
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


#if 0

#include <iostream>
using namespace std;

int main() {
    int num = 42;
    
    // 引用声明和初始化(必须在声明时初始化)
    int& ref = num;  // ref是num的别名
    
    cout << "原始变量num: " << num << endl;
    cout << "引用ref: " << ref << endl;
    cout << "num的地址: " << &num << endl;
    cout << "ref的地址: " << &ref << endl;  // 与num地址相同
    
    // 通过引用修改值
    ref = 100;
    cout << "通过引用修改后:" << endl;
    cout << "num: " << num << endl;
    cout << "ref: " << ref << endl;
    
    // 引用不能重新绑定
    int another = 200;
    // 这不是重新绑定,而是赋值!
    ref = another;  
    ref++;
    cout << "num: " << num << endl;
    cout << "ref: " << ref << endl;
    cout << "another: " << another << endl;
    return 0;
}

#endif


#if 0

#include <iostream>
using namespace std;

void demonstratePointer() {
    cout << "=== 指针特性演示 ===" << endl;
    
    int a = 10, b = 20;
    int* ptr = &a;
    
    cout << "ptr指向a, *ptr = " << *ptr << endl;
    
    // 指针可以重新指向
    ptr = &b;
    cout << "ptr重新指向b, *ptr = " << *ptr << endl;
    
    // 指针可以为空
    ptr = nullptr;
    cout << "ptr设为nullptr: " << (ptr == nullptr ? "是空指针" : "不是空指针") << endl;
    
    // 指针运算
    int arr[] = {1, 2, 3, 4, 5};
    ptr = arr;
    cout << "数组第一个元素: " << *ptr << " 地址: " << ptr << endl;
    ptr++;  // 指向下一个元素
    cout << "数组第二个元素: " << *ptr << " 地址: " << ptr << endl;
}

void demonstrateReference() {
    cout << "\n=== 引用特性演示 ===" << endl;
    
    int a = 10, b = 20;
    int& ref = a;  // 必须初始化
    
    cout << "ref绑定到a, ref = " << ref << endl;
    
    // 引用不能重新绑定到其他变量
    // int& ref2;  // 错误!引用必须初始化
    // ref = b;    // 这是赋值,不是重新绑定
    
    ref = b;  // 实际上是将b的值赋给a
    cout << "执行ref=b后, a = " << a << ", ref = " << ref << endl;
    
    // 引用不能为空,不支持引用运算
}

int main() {
    demonstratePointer();
    demonstrateReference();
    return 0;
}
#endif



#if 0

#include <iostream>
using namespace std;

int main() {
    int num = 42;
    int* ptr = &num;        // 普通指针
    int*& ptrRef = ptr;     // 指向指针的引用
    
    cout << "=== 指向指针的引用演示 ===" << endl;
    cout << "num的值: " << num << endl;
    cout << "ptr指向的值: " << *ptr << endl;
    cout << "ptrRef指向的值: " << *ptrRef << endl;
    
    // 通过指针引用修改指向的值
    *ptrRef = 100;
    cout << "通过ptrRef修改后, num = " << num << endl;
    
    // 通过指针引用改变指针指向
    int another = 200;
    ptrRef = &another;  // 等价于 ptr = &another
    
    cout << "ptrRef重新指向another后:" << endl;
    cout << "ptr指向的值: " << *ptr << endl;
    cout << "ptrRef指向的值: " << *ptrRef << endl;
    
    return 0;
}

#endif


#if 0

#include <iostream>
using namespace std;

// 值传递
void byValue(int x) {
    x = 100;
    cout << "函数内byValue: x = " << x << endl;
}

// 指针传递
void byPointer(int* x) {
    if (x != nullptr) {
        *x = 200;
        cout << "函数内byPointer: *x = " << *x << endl;
    }
}

// 引用传递
void byReference(int& x) {
    x = 300;
    cout << "函数内byReference: x = " << x << endl;
}

// 修改指针本身(指针的引用)
void changePointer(int*& ptr, int* newPtr) {
    ptr = newPtr;
    cout << "函数内changePointer: 指针已重新指向" << endl;
}

int main() {
    int num = 42;
    int another = 999;
    int* ptr = &num;
    
    cout << "初始值: num = " << num << endl;
    
    // 值传递 - 不会修改原变量
    byValue(num);
    cout << "值传递后: num = " << num << endl;
    
    // 指针传递 - 修改指向的值
    byPointer(ptr);
    cout << "指针传递后: num = " << num << endl;
    
    // 引用传递 - 修改原变量
    byReference(num);
    cout << "引用传递后: num = " << num << endl;
    
    // 修改指针指向
    cout << "\n修改指针指向演示:" << endl;
    cout << "修改前ptr指向的值: " << *ptr << endl;
    changePointer(ptr, &another);
    cout << "修改后ptr指向的值: " << *ptr << endl;
    
    return 0;
}

#endif



#if 0

#include <iostream>
using namespace std;

int main() {
    int num1 = 10;
    int num2 = 20;
    
    cout << "=== 常量指针与指向常量的指针 ===" << endl;
    
    // 1. 普通指针
    int* ptr1 = &num1;
    *ptr1 = 15;     // 可以修改指向的值
    ptr1 = &num2;   // 可以修改指针指向
    cout << "普通指针: " << *ptr1 << endl;
    
    // 2. 指向常量的指针(pointer to const)
    const int* ptr2 = &num1;
    // *ptr2 = 25;  // 错误!不能通过ptr2修改值
    ptr2 = &num2;   // 可以修改指针指向
    cout << "指向常量的指针: " << *ptr2 << endl;
    
    // 3. 常量指针(const pointer)
    int* const ptr3 = &num1;
    *ptr3 = 30;     // 可以修改指向的值
    // ptr3 = &num2; // 错误!不能修改指针指向
    cout << "常量指针: " << *ptr3 << endl;
    
    // 4. 指向常量的常量指针(const pointer to const)
    const int* const ptr4 = &num1;
    // *ptr4 = 35;   // 错误!不能修改指向的值
    // ptr4 = &num2; // 错误!不能修改指针指向
    cout << "指向常量的常量指针: " << *ptr4 << endl;
    
    // 5. 常量引用
    const int& ref = num1;
    // ref = 40;     // 错误!不能通过常量引用修改值
    cout << "常量引用: " << ref << endl;
    
    return 0;
}

#endif


#if 0

#include <iostream>
using namespace std;

int main() {
    cout << "=== 指针数组与数组指针 ===" << endl;
    
    // 1. 指针数组 - 数组的每个元素都是指针
    int a = 10, b = 20, c = 30;
    int* ptrArray[3] = {&a, &b, &c};  // 包含3个int*的数组
    
    cout << "指针数组演示:" << endl;
    for (int i = 0; i < 3; i++) {
        cout << "ptrArray[" << i << "] 指向的值: " << *ptrArray[i] << endl;
    }
    
    // 2. 数组指针 - 指向数组的指针
    int arr[4] = {100, 200, 300, 400};
    int (*arrPtr)[4] = &arr;  // 指向包含4个int的数组的指针 注意运算符优先级
    
    cout << "\n数组指针演示:" << endl;
    cout << "通过数组指针访问元素:" << endl;
    for (int i = 0; i < 4; i++) {
        cout << "(*arrPtr)[" << i << "] = " << (*arrPtr)[i] << endl;
        // 或者使用: arrPtr[0][i]
    }
    
    // 3. 二维数组与指针
    int matrix[2][3] = {{1, 2, 3}, {4, 5, 6}};
    int (*matrixPtr)[3] = matrix;  // 指向包含3个int的数组
    
    cout << "\n二维数组指针演示:" << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << "matrixPtr[" << i << "][" << j << "] = " 
                 << matrixPtr[i][j] << " ";
        }
        cout << endl;
    }
    
    return 0;
}


#endif


#if 0
#include <iostream>
using namespace std;

void printArray1(int arr[]) {  // 实际上是 int* arr
    cout << "函数内 sizeof(arr) = " << sizeof(arr) << " 字节" << endl;  // 8字节
}

void printArray2(int (*arr)[4]) {  // 真正的数组指针
    cout << "函数内 sizeof(*arr) = " << sizeof(*arr) << " 字节" << endl;  // 16字节
}

int main() {
    int arr[4] = {100, 200, 300, 400};
    
    cout << "=== 数组 vs 退化指针 vs 数组指针 ===" << endl;
    
    // 1. 原始数组
    cout << "原始数组:" << endl;
    cout << "  sizeof(arr) = " << sizeof(arr) << " 字节" << endl;
    
    // 2. 退化后的指针
    cout << "\n退化指针:" << endl;
    int* degradedPtr = arr;  // 数组名退化为指向第一个元素的指针
    cout << "  sizeof(degradedPtr) = " << sizeof(degradedPtr) << " 字节" << endl;
    cout << "  degradedPtr指向: " << degradedPtr << endl;
    
    // 3. 数组指针
    cout << "\n数组指针:" << endl;
    int (*arrayPtr)[4] = &arr;  // 指向整个数组的指针
    cout << "  sizeof(arrayPtr) = " << sizeof(arrayPtr) << " 字节" << endl;
    cout << "  sizeof(*arrayPtr) = " << sizeof(*arrayPtr) << " 字节" << endl; // 解引用
    cout << "  arrayPtr指向: " << arrayPtr << endl;
    
    cout << "\n=== 指针运算对比 ===" << endl;
    cout << "degradedPtr + 1 = " << (degradedPtr + 1) << endl;  // 移动4字节
    cout << "arrayPtr + 1 = " << (arrayPtr + 1) << endl;        // 移动16字节!
    
    cout << "\n=== 函数传参对比 ===" << endl;
    printArray1(arr);      // 数组退化为指针
    printArray2(&arr);     // 传入数组指针
    
    return 0;
}


#endif


#if 1
#include <iostream>
#include <memory>
using namespace std;

class MyClass {
public:
    MyClass(int val) : value(val) {
        cout << "MyClass构造: " << value << endl;
    }
    
    ~MyClass() {
        cout << "MyClass析构: " << value << endl;
    }
    
    void show() {
        cout << "MyClass值: " << value << endl;
    }
    
private:
    int value;
};

int main() {
    cout << "=== 智能指针演示 ===" << endl;
    
    // 1. unique_ptr - 独占所有权
    {
        cout << "\n--- unique_ptr演示 ---" << endl;
        unique_ptr<MyClass> ptr1 = make_unique<MyClass>(100);
        ptr1->show();
        
        // unique_ptr<MyClass> ptr2 = ptr1;  // 错误!不能拷贝
        unique_ptr<MyClass> ptr2 = move(ptr1);  // 可以移动
        
        if (!ptr1) {
            cout << "ptr1已为空" << endl;
        }
        if (ptr2) {
            cout << "ptr2拥有对象" << endl;
            ptr2->show();
        }
    }  // ptr2在此处自动析构对象
    
    // 2. shared_ptr - 共享所有权
    {
        cout << "\n--- shared_ptr演示 ---" << endl;
        shared_ptr<MyClass> ptr1 = make_shared<MyClass>(200);
        cout << "引用计数: " << ptr1.use_count() << endl;
        
        {
            shared_ptr<MyClass> ptr2 = ptr1;  // 可以拷贝
            cout << "引用计数: " << ptr1.use_count() << endl;
            ptr2->show();
        }  // ptr2离开作用域,引用计数减1
        
        cout << "引用计数: " << ptr1.use_count() << endl;
        ptr1->show();
    }  // ptr1离开作用域,对象被析构
    
    cout << "\n程序结束" << endl;
    return 0;
}

#endif