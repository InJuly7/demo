// 左值(lvalue)：可以取地址、有名字的值，通常出现在赋值语句的左边
// 右值(rvalue)：不能取地址, 临时值、字面量，通常出现在赋值语句的右边

// 左值引用(&)：对左值的别名，必须绑定到左值
// 右值引用(&&)：对右值的别名，可以绑定到右值，C++11引入



// 左值引用基础示例
// #include <iostream>
// using namespace std;

// int main() {
//     // 左值：有名字的变量
//     int a = 10;
//     int b = 20;
    
//     // 左值引用：必须绑定到左值
//     int& ref_a = a;        // 正确：绑定到左值a
//     // int& ref_b = 10;    // 错误：不能绑定到字面量(右值)
//     // int& ref_c = a + b; // 错误：不能绑定到表达式结果(右值)
    
//     cout << "a = " << a << endl;
//     cout << "ref_a = " << ref_a << endl;
    
//     ref_a = 99;  // 修改引用，实际修改的是a
//     cout << "修改后 a = " << a << endl;
    
//     return 0;
// }


// 右值引用基础示例
// #include <iostream>
// using namespace std;

// int main() {
//     int a = 10;
//     int b = 20;
    
//     // 右值引用：绑定到右值
//     int&& rref1 = 42;           // 绑定到字面量
//     int&& rref2 = a + b;        // 绑定到表达式结果
//     int&& rref3 = std::move(a); // 强制转换左值为右值
    
//     cout << "rref1 = " << rref1 << endl;
//     cout << "rref2 = " << rref2 << endl;
//     cout << "rref3 = " << rref3 << endl;
    
//     // 注意：右值引用本身是左值！
//     int&& rref4 = std::move(rref1); // 需要move才能绑定
//     cout << "rref4 = " << rref4 << endl;
//     return 0;
// }



// // 函数参数中的引用类型
// #include <iostream>
// using namespace std;

// void func_lref(int& x) {
//     cout << "左值引用版本: " << x << endl;
// }

// void func_rref(int&& x) {
//     cout << "右值引用版本: " << x << endl;
// }

// // 重载函数
// void func_overload(int& x) {
//     cout << "调用左值引用重载: " << x << endl;
// }

// void func_overload(int&& x) {
//     cout << "调用右值引用重载: " << x << endl;
// }

// int main() {
//     int a = 10;
    
//     // 左值引用只能接受左值
//     func_lref(a);           // 正确
//     // func_lref(20);       // 错误：不能传递右值
    
//     // 右值引用只能接受右值
//     func_rref(30);          // 正确
//     func_rref(std::move(a)); // 正确
//     // func_rref(a);        // 错误：不能传递左值
    
//     // 重载函数自动选择合适的版本
//     int b = 5;
//     func_overload(b);       // 调用左值引用版本
//     func_overload(100);     // 调用右值引用版本
    
//     return 0;
// }



// Demo4 - 移动语义示例s
#include <iostream>
#include <vector>
#include <string>
using namespace std;

class MyClass {
private:
    string data;
    
public:
    // 构造函数
    MyClass(const string& str) : data(str) {
        cout << "构造: " << data << endl;
    }
    
    // 拷贝构造函数
    MyClass(const MyClass& other) : data(other.data) {
        cout << "拷贝构造: " << data << endl;
    }
    
    // 移动构造函数
    MyClass(MyClass&& other) noexcept : data(std::move(other.data)) {
        cout << "移动构造: " << data << endl;
    }
    
    // 拷贝赋值
    MyClass& operator=(const MyClass& other) {
        if (this != &other) {
            data = other.data;
            cout << "拷贝赋值: " << data << endl;
        }
        return *this;
    }
    
    // 移动赋值
    MyClass& operator=(MyClass&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            cout << "移动赋值: " << data << endl;
        }
        return *this;
    }
    
    void print() const {
        cout << "数据: " << data << endl;
    }
};

MyClass createObject() {
    return MyClass("临时对象");
}

int main() {
    cout << "=== 拷贝 vs 移动 ===" << endl;
    
    MyClass obj1("对象1");
    MyClass obj2 = obj1;        // 拷贝构造
    MyClass obj3 = std::move(obj1); // 移动构造
    
    cout << "\n=== 函数返回值 ===" << endl;
    MyClass obj4 = createObject(); // 通常会优化掉，但展示概念
    
    cout << "\n=== 容器中的移动 ===" << endl;
    vector<MyClass> vec;
    vec.push_back(MyClass("push对象")); // 移动构造
    
    return 0;
}





// #include <iostream>
// #include <utility>
// using namespace std;

// void process(int& x) {
//     cout << "处理左值引用: " << x << endl;
// }

// void process(int&& x) {
//     cout << "处理右值引用: " << x << endl;
// }

// // 完美转发函数模板
// template<typename T>
// void wrapper(T&& param) {
//     cout << "wrapper接收参数，转发给process" << endl;
//     process(std::forward<T>(param));  // 完美转发
// }

// // 对比：不使用完美转发
// template<typename T>
// void bad_wrapper(T&& param) {
//     cout << "bad_wrapper接收参数，直接传递给process" << endl;
//     process(param);  // param本身是左值！
// }

// int main() {
//     int a = 10;
    
//     cout << "=== 使用完美转发 ===" << endl;
//     wrapper(a);          // 转发左值
//     wrapper(20);         // 转发右值
    
//     cout << "\n=== 不使用完美转发 ===" << endl;
//     bad_wrapper(a);      // 都当作左值处理
//     bad_wrapper(30);     // 都当作左值处理
    
//     return 0;
// }
