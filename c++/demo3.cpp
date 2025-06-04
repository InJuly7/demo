// #include <iostream>
// #include <iomanip>
// using namespace std;

// class Base1 {
// public:
//     int base1_data = 100;
//     virtual void func1() { cout << "Base1::func1" << endl; }
//     virtual ~Base1() = default;
// };

// class Base2 {
// public:
//     int base2_data = 200;
//     virtual void func2() { cout << "Base2::func2" << endl; }
//     virtual ~Base2() = default;
// };

// class Child : public Base1, public Base2 {
// public:
//     int child_data = 300;
    
//     void func1() override { cout << "Child::func1" << endl; }
//     void func2() override { cout << "Child::func2" << endl; }
    
//     void printData() {
//         cout << "base1_data: " << base1_data << endl;
//         cout << "base2_data: " << base2_data << endl; 
//         cout << "child_data: " << child_data << endl;
//     }
// };

// int main() {
//     Child child;
    
//     cout << "=== Child对象内存布局分析 ===" << endl;
//     cout << "Child对象起始地址: " << &child << endl;
//     cout << "Base1子对象地址: " << static_cast<Base1*>(&child) << endl;
//     cout << "Base2子对象地址: " << static_cast<Base2*>(&child) << endl;
    
//     // 计算偏移量
//     char* child_addr = reinterpret_cast<char*>(&child);
//     char* base1_addr = reinterpret_cast<char*>(static_cast<Base1*>(&child));
//     char* base2_addr = reinterpret_cast<char*>(static_cast<Base2*>(&child));
    
//     cout << "Base1偏移: " << (base1_addr - child_addr) << " bytes" << endl;
//     cout << "Base2偏移: " << (base2_addr - child_addr) << " bytes" << endl;
//     cout << "地址差异: " << (base2_addr - base1_addr) << " bytes" << endl;
    
//     cout << "\n=== 对象大小分析 ===" << endl;
//     cout << "sizeof(Base1): " << sizeof(Base1) << " bytes" << endl;
//     cout << "sizeof(Base2): " << sizeof(Base2) << " bytes" << endl;
//     cout << "sizeof(Child): " << sizeof(Child) << " bytes" << endl;
    
//     return 0;
// }






// #include <iostream>
// #include <iomanip>
// using namespace std;

// class Base1 {
// public:
//     int base1_data = 100;
//     virtual void func1() { cout << "Base1::func1" << endl; }
//     virtual ~Base1() = default;
// };

// class Base2 {  
// public:
//     char base2_data = 'a';
//     virtual void func2() { cout << "Base2::func2" << endl; }
//     virtual ~Base2() = default;
// };

// class Child : public Base1, public Base2 {
// public:
//     char child_data = 'c';
    
//     void func1() override { cout << "Child::func1 from Base1" << endl; }
//     void func2() override { cout << "Child::func2 from Base2" << endl; }
// };

// void printMemoryLayout(void* ptr, size_t size) {
//     unsigned char* bytes = static_cast<unsigned char*>(ptr);
//     cout << "内存内容 (16进制):" << endl;
    
//     for(size_t i = 0; i < size; i += 8) {
//         cout << "偏移 " << setw(2) << i << ": ";
//         for(size_t j = 0; j < 8 && i + j < size; ++j) {
//             cout << hex << setw(2) << setfill('0') 
//                  << static_cast<int>(bytes[i + j]) << " ";
//         }
//         cout << endl;
//     }
//     cout << dec << setfill(' ');
// }

// int main() {
//     Child child;
    
//     cout << "=== Child对象详细内存布局 ===" << endl;
//     cout << "Child对象总大小: " << sizeof(Child) << " bytes" << endl;
//     cout << "Child对象地址: " << &child << endl;
    
//     // 获取各个组件的地址
//     void* child_addr = &child;
//     void* base1_addr = static_cast<Base1*>(&child);  
//     void* base2_addr = static_cast<Base2*>(&child);
    
//     cout << "\n=== 地址映射 ===" << endl;
//     cout << "Child起始:  " << child_addr << endl;
//     cout << "Base1部分:  " << base1_addr << endl;
//     cout << "Base2部分:  " << base2_addr << endl;
    
//     // 计算各部分偏移
//     ptrdiff_t base1_offset = static_cast<char*>(base1_addr) - static_cast<char*>(child_addr);
//     ptrdiff_t base2_offset = static_cast<char*>(base2_addr) - static_cast<char*>(child_addr);
    


//     cout << "\n=== 内存布局图解 ===" << endl;
//     cout << "┌─────────────────────┐ <- " << child_addr << endl;
//     cout << "│   Base1 vptr        │   偏移 " << base1_offset << " 地址: " << base1_addr << endl;
//     cout << "├─────────────────────┤" << endl; 
//     cout << "│   base1_data (100)  │   偏移 " << (base1_offset + 8) << " 地址: " << &child.base1_data << endl;
//     cout << "├─────────────────────┤ <- " << base2_addr << endl;
//     cout << "│   Base2 vptr        │   偏移 " << base2_offset << " 地址: " << base2_addr << endl;
//     cout << "├─────────────────────┤" << endl;
//     cout << "│   base2_data (200)  │   偏移 " << (base2_offset + 8) << " 地址: " << (void*)&child.base2_data << endl;
//     cout << "├─────────────────────┤" << endl;
//     cout << "│   child_data (300)  │   偏移 " << (base2_offset + 9) << " 地址: " << (void*)&child.child_data << endl;
//     cout << "└─────────────────────┘" << endl;
        
//     // 验证数据访问
//     cout << "\n=== 数据访问验证 ===" << endl;
//     cout << "通过Child访问:" << endl;
//     cout << "  base1_data = " << child.base1_data << endl;
//     cout << "  base2_data = " << child.base2_data << endl;  
//     cout << "  child_data = " << child.child_data << endl;
    
//     cout << "通过Base1*访问:" << endl;
//     Base1* p1 = &child;
//     cout << "  base1_data = " << p1->base1_data << endl;
    
//     cout << "通过Base2*访问:" << endl;
//     Base2* p2 = &child;
//     cout << "  base2_data = " << p2->base2_data << endl;
    
//     return 0;
// }




#include <iostream>
using namespace std;

class Base1 {
public:
    int data1 = 111;
    virtual void func1() { cout << "Base1::func1, data1=" << data1 << endl; }
    virtual ~Base1() = default;
};

class Base2 {
public:
    int data2 = 222;
    virtual void func2() { cout << "Base2::func2, data2=" << data2 << endl; }
    virtual ~Base2() = default;
};

class Child : public Base1, public Base2 {
public:
    int data3 = 333;
    
    void func1() override { 
        cout << "Child::func1, data1=" << data1 << ", data3=" << data3 << endl; 
    }
    void func2() override { 
        cout << "Child::func2, data2=" << data2 << ", data3=" << data3 << endl; 
    }
    
    void showAllData() {
        cout << "Child::showAllData: " << data1 << ", " << data2 << ", " << data3 << endl;
    }
};

int main() {
    Child child;
    Child* childPtr = &child;
    
    cout << "=== 指针转换机制分析 ===" << endl;
    cout << "原始Child*: " << static_cast<void*>(childPtr) << endl;
    
    // 向上转型 - 编译器会自动调整指针
    Base1* base1Ptr = childPtr;  // 隐式转换, 无需偏移
    Base2* base2Ptr = childPtr;  // 隐式转换, 需要偏移
    
    cout << "转换为Base1*: " << static_cast<void*>(base1Ptr) << " (偏移: " 
         << (static_cast<char*>(static_cast<void*>(base1Ptr)) - 
             static_cast<char*>(static_cast<void*>(childPtr))) << ")" << endl;
    cout << "转换为Base2*: " << static_cast<void*>(base2Ptr) << " (偏移: "
         << (static_cast<char*>(static_cast<void*>(base2Ptr)) - 
             static_cast<char*>(static_cast<void*>(childPtr))) << ")" << endl;
    
    cout << "\n=== 通过不同指针调用虚函数 ===" << endl;
    // 每个指针都能正确找到对应的vtable
    base1Ptr->func1();  // 使用Base1部分的vtable，但调用Child::func1
    base2Ptr->func2();  // 使用Base2部分的vtable，但调用Child::func2
    
    cout << "\n=== 反向转换测试 ===" << endl; 
    // 向下转型 - 编译器同样会调整指针
    Child* convertedBack = static_cast<Child*>(base2Ptr);
    cout << "Base2*转回Child*: " << static_cast<void*>(convertedBack) << endl;
    cout << "转换正确性验证: " << 
            (convertedBack == childPtr ? "成功" : "失败") << endl;
    
    convertedBack->showAllData();
    
    cout << "\n=== 编译器做了什么？===" << endl;
    cout << "当你写 Base2* p = &child; 时，编译器实际执行:" << endl;
    cout << "Base2* p = (Base2*)((char*)&child + " 
         << (static_cast<char*>(static_cast<void*>(base2Ptr)) - 
             static_cast<char*>(static_cast<void*>(childPtr))) << ");" << endl;
    
    return 0;
}
