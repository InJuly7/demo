// #include <iostream>
// using namespace std;

// class Base {
// public:
//     // 普通公共函数 - 静态绑定，不支持多态
//     void normalFunction() {
//         cout << "Base normal function called" << endl;
//     }
    
//     void anotherFunction() {
//         cout << "Base another function" << endl;
//     }
// };

// class Child : public Base {
// public:
//     // 重新定义(隐藏)基类的normalFunction
//     // 注意：这不是重写(override)，而是隐藏(hiding)
//     void normalFunction() {
//         cout << "Child normal function called" << endl;
//     }
// };

// int main() {
//     Child child;
//     Base* basePtr = &child;
    
//     cout << "=== 直接调用子类对象 ===" << endl;
//     child.normalFunction();        // 调用Child版本
//     child.anotherFunction();       // 调用继承的Base版本
    
//     cout << "=== 通过基类指针调用子类对象 ===" << endl;  
//     basePtr->normalFunction();     // 调用Base版本！(静态绑定)
//     basePtr->anotherFunction();    // 调用Base版本
    
//     return 0;
// }



// #include <iostream>
// using namespace std;

// class Base {
// public:
//     // 虚函数 - 支持多态，运行时动态绑定
//     virtual void virtualFunction() {
//         cout << "Base virtual function called" << endl;
//     }
    
//     virtual void anotherVirtual() {
//         cout << "Base another virtual function" << endl;
//     }
    
//     // 虚析构函数确保正确的析构顺序
//     virtual ~Base() {
//         cout << "Base destructor" << endl;
//     }
// };

// class Child : public Base {
// public:
//     // 重写(override)基类的虚函数
//     void virtualFunction() override {
//         cout << "Child virtual function called" << endl;
//     }
    
//     // anotherVirtual没有重写，使用基类版本
    
//     ~Child() {
//         cout << "Child destructor" << endl;
//     }
// };

// int main() {
//     Base* basePtr;
//     Child child;
    
//     // 直接调用
//     child.virtualFunction();       // Child版本
    
//     // 通过基类指针调用 - 展示多态
//     basePtr = &child;
//     basePtr->virtualFunction();    // Child版本(动态绑定)
//     basePtr->anotherVirtual();     // Base版本(子类未重写)
    
//     return 0;
// }


// #include <iostream>
// #include <vector>
// #include <memory>
// using namespace std;

// // 抽象基类 - 包含纯虚函数
// class Shape {
// public:
//     // 纯虚函数 - 必须被子类实现
//     virtual double getArea() = 0;
    
//     // 普通虚函数 - 可选择性重写
//     virtual void printInfo() {
//         cout << "This is a shape with area: " << getArea() << endl;
//     }
    
//     // 虚析构函数
//     virtual ~Shape() = default;
// };

// class Circle : public Shape {
// private:
//     double radius;
    
// public:
//     Circle(double r) : radius(r) {}
    
//     // 必须实现纯虚函数
//     double getArea() override {
//         return 3.14159 * radius * radius;
//     }
    
//     void printInfo() override {
//         cout << "Circle area: " << getArea() << endl;
//     }
// };

// class Rectangle : public Shape {
// private:
//     double width, height;
    
// public:
//     Rectangle(double w, double h) : width(w), height(h) {}
    
//     // 必须实现纯虚函数
//     double getArea() override {
//         return width * height;
//     }
    
//     void printInfo() override {
//         cout << "Rectangle area: " << getArea() << endl;
//     }
// };

// int main() {
//     // Shape shape;  // 错误！不能实例化抽象类
    
//     Circle circle(5.0);
//     Rectangle rect(4.0, 6.0);
    
//     // 直接调用
//     circle.printInfo();
//     rect.printInfo();
    
//     // 多态数组
//     vector<unique_ptr<Shape>> shapes;
//     shapes.push_back(make_unique<Circle>(5.0));
//     shapes.push_back(make_unique<Rectangle>(4.0, 6.0));
    
//     for(auto& shape : shapes) {
//         shape->printInfo();  // 动态绑定到正确的实现
//     }
    
//     return 0;
// }


#include <iostream>
using namespace std;

class Base {
public:
    // 1. 普通公共函数
    void normalFunction() {
        cout << "Base normal function" << endl;
    }
    
    // 2. 虚函数
    virtual void virtualFunction() {
        cout << "Base virtual function" << endl;
    }
    
    // 3. 纯虚函数
    virtual void pureVirtualFunction() = 0;
    
    virtual ~Base() = default;
};

class Child : public Base {
public:
    // 隐藏基类普通函数
    void normalFunction() {
        cout << "Child normal function" << endl;
    }
    
    // 重写基类虚函数
    void virtualFunction() override {
        cout << "Child virtual function" << endl;
    }
    
    // 必须实现纯虚函数
    void pureVirtualFunction() override {
        cout << "Child pure virtual implementation" << endl;
    }
};

int main() {
    Child child;
    Base* basePtr = &child;
    
    cout << "=== 直接调用子类对象 ===" << endl;
    child.normalFunction();        // Child版本
    child.virtualFunction();       // Child版本
    child.pureVirtualFunction();   // Child版本
    
    cout << "=== 通过基类指针调用 ===" << endl;
    basePtr->normalFunction();     // Base版本(静态绑定)
    basePtr->virtualFunction();    // Child版本(动态绑定)
    basePtr->pureVirtualFunction(); // Child版本(动态绑定)
    
    return 0;
}
