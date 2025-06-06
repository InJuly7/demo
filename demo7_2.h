#ifndef DEMO7_2_H
#define DEMO7_2_H

// 函数声明
int factorial(int n);
double power(double base, int exp);

// 类声明
class Calculator {
private:
    double result;

public:
    Calculator();
    void add(double value);
    void multiply(double value);
    void clear();
    double getResult() const;
};

#endif
