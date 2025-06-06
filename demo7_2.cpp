#include "demo7_2.h"

// 函数定义
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

double power(double base, int exp) {
    double result = 1.0;
    for (int i = 0; i < exp; i++) {
        result *= base;
    }
    return result;
}

// 类成员定义
Calculator::Calculator() : result(0.0) {}

void Calculator::add(double value) {
    result += value;
}

void Calculator::multiply(double value) {
    result *= value;
}

void Calculator::clear() {
    result = 0.0;
}

double Calculator::getResult() const {
    return result;
}





// 展示extern关键字在多文件中的使用
#include <iostream>

// 定义全局变量
int global_counter = 0;
const char* global_message = "Hello World";

void incrementCounter() {
    global_counter++;
}

void printGlobalData() {
    std::cout << "Counter: " << global_counter << std::endl;
    std::cout << "Message: " << global_message << std::endl;
}
