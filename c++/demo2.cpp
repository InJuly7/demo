// #include <iostream>
// using namespace std;

// class Base {
// public:
//     virtual void func1() { cout << "Base virtual func" << endl; }
//     virtual void func2() { cout << "Base virtual func2" << endl; }
// };

// class Child : public Base {
// public:
//     void func1() override { cout << "Child virtual func" << endl; }
//     // func2 没有重写，继承Base的版本
// };

// int main() {
//     cout << "Base对象大小: " << sizeof(Base) << " bytes" << endl;
//     cout << "Child对象大小: " << sizeof(Child) << " bytes" << endl;
    
//     Base base;
//     Child child;
    
//     base.func1();     // 直接调用
//     child.func1();    // 直接调用
    
//     Base* ptr = &child;
//     ptr->func1();     // 通过vtable调用Child::func1
    
//     return 0;
// }




// #include <iostream>
// #include <iomanip>
// using namespace std;

// class Base {
// public:
//     virtual void func1() { cout << "Base::func1()" << endl; }
//     virtual void func2() { cout << "Base::func2()" << endl; }
//     virtual ~Base() = default;
// };

// class Child : public Base {
// public:
//     void func1() override { cout << "Child::func1()" << endl; }
//     // func2 继承Base版本
// };

// // 获取虚函数表指针的辅助函数
// void* getVTablePtr(void* obj) {
//     return *static_cast<void**>(obj);
// }

// // 获取虚函数地址的辅助函数  
// void* getVTableFunction(void* vtable, int index) {
//     return static_cast<void**>(vtable)[index];
// }

// int main() {
//     Base base;
//     Child child;
    
//     cout << "=== Base类虚函数表分析 ===" << endl;
//     cout << "Base对象地址: " << &base << endl;
//     void* base_vtable = getVTablePtr(&base);
//     cout << "vtable地址: " << base_vtable << endl;
//     cout << "func1地址: " << getVTableFunction(base_vtable, 0) << endl;
//     cout << "func2地址: " << getVTableFunction(base_vtable, 1) << endl;
    
//     cout << "\n=== Child类虚函数表分析 ===" << endl;
//     cout << "Child对象地址: " << &child << endl;
//     void* child_vtable = getVTablePtr(&child);
//     cout << "vtable地址: " << child_vtable << endl;
//     cout << "func1地址: " << getVTableFunction(child_vtable, 0) << endl;
//     cout << "func2地址: " << getVTableFunction(child_vtable, 1) << endl;
    
//     cout << "\n=== 验证vtable是否不同 ===" << endl;
//     cout << "Base和Child的vtable相同吗? " << 
//             (base_vtable == child_vtable ? "是" : "否") << endl;
    
//     return 0;
// }


// #include <iostream>
// using namespace std;

// class Base1 {
// public:
//     virtual void func1() { cout << "Base1::func1" << endl; }
//     virtual ~Base1() = default;
// };

// class Base2 {
// public:
//     virtual void func2() { cout << "Base2::func2" << endl; }
//     virtual ~Base2() = default;
// };

// class Child : public Base1, public Base2 {
// public:
//     void func1() override { cout << "Child::func1 from Base1" << endl; }
//     void func2() override { cout << "Child::func2 from Base2" << endl; }
// };

// int main() {
//     cout << "Base1对象大小: " << sizeof(Base1) << " bytes" << endl;
//     cout << "Base2对象大小: " << sizeof(Base2) << " bytes" << endl;
//     cout << "Child对象大小: " << sizeof(Child) << " bytes" << endl;
    
//     Child child;
    
//     // 直接调用
//     child.func1();
//     child.func2();
    
//     // 通过不同基类指针调用
//     Base1* ptr1 = &child;
//     Base2* ptr2 = &child;
    
//     ptr1->func1();  // 通过Base1的vtable
//     ptr2->func2();  // 通过Base2的vtable
    
//     // 展示指针地址差异(多重继承中子对象地址不同)
//     cout << "\n=== 地址分析 ===" << endl;
//     cout << "Child对象地址: " << &child << endl;
//     cout << "Base1*指针地址: " << static_cast<Base1*>(&child) << endl;
//     cout << "Base2*指针地址: " << static_cast<Base2*>(&child) << endl;
    
//     return 0;
// }



#include <iostream>
#include <iomanip>
using namespace std;

class Base {
protected:
    int base_data = 10;
    
public:
    virtual void func1() { cout << "Base::func1 called" << endl; }
    virtual void func2() { cout << "Base::func2 called" << endl; }
    virtual ~Base() = default;
};

class Child : public Base {
private:
    int member_data = 42;
    
public:
    void func1() override { cout << "Child::func1 called" << endl; }
    // func2 继承Base版本
    
    int getMemberData() const { return member_data; }
};


int main() {
    Child child;
    
    cout << "=== 内存布局分析 ===" << endl;
    
    // 获取对象的内存布局
    cout << "Child对象内存布局:" << endl;
    cout << "[" << &child << "] vptr -> vtable" << endl;
    
    // 获取vptr (虚函数表指针)
    void** vptr = *reinterpret_cast<void***>(&child);
    cout << "[";
    cout << hex << vptr << "] vtable" << endl;
    
    // 获取 func1 (函数指针)
    void* func1 = *reinterpret_cast<void**>(vptr);
    cout << "[";
    cout << hex << func1 << "] func1" << endl;

    cout << "func1地址: " << reinterpret_cast<void*>(&Child::func1) << endl;



    cout << "\nvtable内容:" << endl;
    // 注意：这种直接访问vtable的方式是平台相关和编译器相关的
    cout << "[0] -> Child::func1" << endl;
    cout << "[1] -> Base::func2" << endl;  
    cout << "[2] -> Child::~Child" << endl;
    
    cout << "\n=== 通过vtable调用验证 ===" << endl;
    
    // 模拟编译器的虚函数调用过程
    Base* basePtr = &child;
    
    // 这相当于: basePtr->func1();
    // 编译器生成的代码类似：
    // 1. 获取对象的vptr
    // 2. 从vtable中获取函数地址
    // 3. 调用函数

    // ┌─────────────┐
    // │    vptr     │ ──┐
    // ├─────────────┤   │
    // │  成员变量1   │   │
    // ├─────────────┤   │
    // │  成员变量2   │   │
    // └─────────────┘   │
    //                   │
    //                   ▼
    //             ┌─────────────┐ vtable
    //             │ func1的地址  │
    //             ├─────────────┤
    //             │ func2的地址  │  
    //             ├─────────────┤
    //             │ 析构函数地址  │
    //             └─────────────┘



    basePtr->func1();  // 实际调用Child::func1
    basePtr->func2();  // 实际调用Base::func2
    
    return 0;
}



