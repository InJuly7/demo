// 声明(Declaration): 告诉编译器某个名字的存在和类型，但不分配内存
// 定义(Definition): 不仅声明了名字，还为其分配了内存空间

#if 1
// variable_demo.cpp
#include <iostream>

// 声明全局变量 (extern关键字)
extern int global_var;  // 仅声明，不定义

// 定义全局变量
int global_var = 100;   // 既声明又定义

int main() {
    // 声明并定义局部变量
    int local_var = 50;
    
    std::cout << "Global: " << global_var << std::endl;
    std::cout << "Local: " << local_var << std::endl;
    
    return 0;
}
#endif


#if 0
// function_demo.cpp
#include <iostream>

// 函数声明 (函数原型)
int add(int a, int b);          // 仅声明
void printMessage();            // 仅声明
double multiply(double x, double y);  // 仅声明

int main() {
    int result = add(5, 3);
    std::cout << "5 + 3 = " << result << std::endl;
    
    printMessage();
    
    double product = multiply(2.5, 4.0);
    std::cout << "2.5 * 4.0 = " << product << std::endl;
    
    return 0;
}

// 函数定义
int add(int a, int b) {
    return a + b;
}

void printMessage() {
    std::cout << "Hello from declared function!" << std::endl;
}

double multiply(double x, double y) {
    return x * y;
}

#endif




#if 0
// class_demo.cpp
#include <iostream>
#include <string>

// 类声明
class Person {
private:
    std::string name;
    int age;

public:
    // 构造函数声明
    Person(const std::string& n, int a);
    
    // 成员函数声明
    void introduce();
    void setAge(int newAge);
    int getAge() const;
    std::string getName() const;
};

// 构造函数定义
Person::Person(const std::string& n, int a) : name(n), age(a) {}

// 成员函数定义
void Person::introduce() {
    std::cout << "Hi, I'm " << name << ", " << age << " years old." << std::endl;
}

void Person::setAge(int newAge) {
    if (newAge >= 0) {
        age = newAge;
    }
}

int Person::getAge() const {
    return age;
}

std::string Person::getName() const {
    return name;
}

int main() {
    Person person("Alice", 25);
    person.introduce();
    
    person.setAge(26);
    std::cout << "Updated age: " << person.getAge() << std::endl;
    
    return 0;
}

#endif


#if 0
// 头文件 demo
#include <iostream>
#include "demo7_2.h"

int main() {
    // 使用声明的函数
    std::cout << "5! = " << factorial(5) << std::endl;
    std::cout << "2^3 = " << power(2, 3) << std::endl;
    
    // 使用声明的类
    Calculator calc;
    calc.add(10);
    calc.multiply(2);
    std::cout << "Calculator result: " << calc.getResult() << std::endl;
    
    return 0;
}

#endif


#if 0
// extern_demo.cpp
#include <iostream>

// 声明外部变量
extern int global_counter;
extern const char* global_message;

// 声明外部函数
extern void incrementCounter();
extern void printGlobalData();

int main() {
    std::cout << "Initial state:" << std::endl;
    printGlobalData();
    
    incrementCounter();
    incrementCounter();
    
    std::cout << "After incrementing:" << std::endl;
    printGlobalData();
    
    // 直接访问外部变量
    global_counter += 5;
    std::cout << "After direct modification: " << global_counter << std::endl;
    
    return 0;
}
#endif
