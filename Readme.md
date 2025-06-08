### 模板
### Demo_x - xxxx示例
功能:
```bash
# 使用方法:
# 预期输出:
```


### Demo1 - 普通公共函数, 虚函数, 纯虚函数(抽象类), 综合对比示例
功能: 演示普通公共函数的静态绑定特性, 演示虚函数的多态特性和动态绑定, 演示纯虚函数定义抽象接口，强制子类实现
```bash
# 使用方法: 
g++ -o demo1 demo1.cpp && ./demo1


# 预期输出: 
=== 直接调用子类对象 ===
Child normal function called
Base another function
=== 通过基类指针调用子类对象 ===
Base normal function called
Base another function


Child virtual function called
Child virtual function called
Base another virtual function
Child destructor
Base destructor


Circle area: 78.5397
Rectangle area: 24
Circle area: 78.5397
Rectangle area: 24
```


### Demo2 - 虚函数表基本概念, 虚函数表内部结构, 多重继承的虚函数表, 虚函数表内存布局可视化
功能: 演示虚函数表的存在和基本工作机制, 通过内存分析查看虚函数表指针和函数地址, 展示复杂继承关系下的虚函数表结构, 详细展示虚函数表在内存中的组织结构

```bash
# 使用方法:
g++ demo2.cpp -o demo2 && ./demo2



Base对象大小: 8 bytes
Child对象大小: 8 bytes
Base virtual func
Child virtual func
Child virtual func



=== Base类虚函数表分析 ===
Base对象地址: 0x7ffd2523a708
vtable地址: 0x5612187c0d68
func1地址: 0x5612187be5b2
func2地址: 0x5612187be5ec

=== Child类虚函数表分析 ===
Child对象地址: 0x7ffd2523a700
vtable地址: 0x5612187c0d38
func1地址: 0x5612187be626
func2地址: 0x5612187be5ec

=== 验证vtable是否不同 ===
Base和Child的vtable相同吗? 否




Base1对象大小: 8 bytes
Base2对象大小: 8 bytes
Child对象大小: 16 bytes
Child::func1 from Base1
Child::func2 from Base2
Child::func1 from Base1
Child::func2 from Base2

=== 地址分析 ===
Child对象地址: 0x7fff16df89a0
Base1*指针地址: 0x7fff16df89a0
Base2*指针地址: 0x7fff16df89a8



=== 内存布局分析 ===
Child对象内存布局:
[0x7fff4f1c6310] vptr -> vtable
[0x5626671c8d38] vtable
[0x5626671c66ce] func1
func1地址: 0x5626671c66ce

vtable内容:
[0] -> Child::func1
[1] -> Base::func2
[2] -> Child::~Child

=== 通过vtable调用验证 ===
Child::func1 called
Base::func2 called
```

### Demo3 - 多重继承内存布局示例, 内存布局可视化, 指针转换的内部机制
功能: 深入理解多重继承中对象的内存布局和地址转换, 可视化展示多重继承中每个字节的具体内容, 展示编译器如何处理多重继承中的指针转换
```bash
# 使用方法:
g++ demo3.cpp -o demo3 && ./demo3


# 预期输出:
=== Child对象内存布局分析 ===
Child对象起始地址: 0x7ffcb2c5b6a0
Base1子对象地址: 0x7ffcb2c5b6a0
Base2子对象地址: 0x7ffcb2c5b6b0
Base1偏移: 0 bytes
Base2偏移: 16 bytes
地址差异: 16 bytes

=== 对象大小分析 ===
sizeof(Base1): 16 bytes
sizeof(Base2): 16 bytes
sizeof(Child): 32 bytes



=== Child对象详细内存布局 ===
Child对象总大小: 32 bytes
Child对象地址: 0x7ffe8394f820

=== 地址映射 ===
Child起始:  0x7ffe8394f820
Base1部分:  0x7ffe8394f820
Base2部分:  0x7ffe8394f830

=== 内存布局图解 ===
┌─────────────────────┐ <- 0x7ffe8394f820
│   Base1 vptr        │   偏移 0 地址: 0x7ffe8394f820
├─────────────────────┤
│   base1_data (100)  │   偏移 8 地址: 0x7ffe8394f828
├─────────────────────┤ <- 0x7ffe8394f830
│   Base2 vptr        │   偏移 16 地址: 0x7ffe8394f830
├─────────────────────┤
│   base2_data (200)  │   偏移 24 地址: 0x7ffe8394f838
├─────────────────────┤
│   child_data (300)  │   偏移 25 地址: 0x7ffe8394f839
└─────────────────────┘

=== 数据访问验证 ===
通过Child访问:
  base1_data = 100
  base2_data = a
  child_data = c
通过Base1*访问:
  base1_data = 100
通过Base2*访问:
  base2_data = a



Child对象内存布局:
┌─────────────────┐ <- Child对象起始地址
│  Base1子对象     │ <- Base1*指向这里 (偏移0)
│  - vptr1        │
│  - base1成员    │  
├─────────────────┤ 
│  Base2子对象     │ <- Base2*指向这里 (偏移16)
│  - vptr2        │
│  - base2成员    │
├─────────────────┤
│  Child自己的成员 │
└─────────────────┘
=== 指针转换机制分析 ===
原始Child*: 0x7ffd010774e0
转换为Base1*: 0x7ffd010774e0 (偏移: 0)
转换为Base2*: 0x7ffd010774f0 (偏移: 16)

=== 通过不同指针调用虚函数 ===
Child::func1, data1=111, data3=333
Child::func2, data2=222, data3=333

=== 反向转换测试 ===
Base2*转回Child*: 0x7ffd010774e0
转换正确性验证: 成功
Child::showAllData: 111, 222, 333

=== 编译器做了什么？===
当你写 Base2* p = &child; 时，编译器实际执行:
Base2* p = (Base2*)((char*)&child + 16);
```


### Demo4 - 左值/右值引用示例
功能:
```bash
# 使用方法:
# 预期输出:
```


### Demo5 - 基本数据类型大小示例
功能: 演示C++中各种基本数据类型的内存大小，理解不同类型占用的字节数

```bash
# 使用方法: 编译并运行程序
g++ -o demo1 demo1.cpp && ./demo1

# 预期输出 (在64位系统上):
=== C++ 基本数据类型大小演示 ===

--- 整型系列 ---
int 大小: 4 字节
short 大小: 2 字节
long 大小: 8 字节
long long 大小: 8 字节

--- 字符型系列 ---
char 大小: 1 字节
signed char 大小: 1 字节
unsigned char 大小: 1 字节

--- 浮点型系列 ---
float 大小: 4 字节
double 大小: 8 字节
long double 大小: 16 字节

--- 其他类型 ---
bool 大小: 1 字节

--- 变量实例大小 ---
变量 a(int) 大小: 4 字节
变量 lla(long long) 大小: 8 字节
变量 sa(short) 大小: 2 字节
变量 sc(char) 大小: 1 字节
变量 bt(bool) 大小: 1 字节

```

### Demo6 - 类型转换示例
功能： 介绍 bool, int, float, double, signed char, unsigned char 之间的类型转换
```bash
# 使用方法: 
g++ demo6.cpp -o demo6 && ./demo6


# 预期输出: 
=== C++ 类型转换示例 ===

1. 非布尔类型 -> 布尔类型:
int 0 -> bool: 0
int 42 -> bool: 1
int -5 -> bool: 1
double 0.0 -> bool: 0
double 3.14 -> bool: 1

2. 布尔类型 -> 非布尔类型:
true -> int: 1
false -> int: 0
true -> double: 1
false -> double: 0

3. 浮点数 -> 整数类型 (截断小数部分):
double 3.14159 -> int: 3
double -3.14159 -> int: -3
float 9.99 -> int: 9

4. 整数 -> 浮点类型:
int 123456789 -> float: 1.23457e+08
int 123456789 -> double: 1.23457e+08

5. 无符号类型溢出 (取模运算):
unsigned char = 255: 255
unsigned char = 256: 0
unsigned char = 257: 1
unsigned char = -1: 255

6. 带符号类型溢出 (未定义行为 - 结果可能不可预测):
signed char 最大值: 127
signed char 最小值: -128
signed char = SCHAR_MAX: 127
signed char = SCHAR_MAX + 1 (未定义): -128
signed char = 200 (可能未定义): -56

7. 显式类型转换 (强制转换):
使用 static_cast<int> 显式转换:
double 3.14159 -> int 3
```


### Demo7 - 各种情况声明与定义示例
功能： 变量， 函数， 类， 头文件， 外部变量 声明与定义
```js
# 使用方法:
g++ demo7_2.cpp demo7_1.cpp -o demo7 && ./demo7

# 预期输出:

  # variable_demo.cpp
  Global: 100
  Local: 50

  # function_demo.cpp
  5 + 3 = 8
  Hello from declared function!
  2.5 * 4.0 = 10
  
  
  # class_demo.cpp
  Hi, I'm Alice, 25 years old.
  Updated age: 26

  
  # 头文件 demo
  5! = 120
  2^3 = 8
  Calculator result: 20


  # extern demo
  Initial state:
  Counter: 0
  Message: Hello World
  After incrementing:
  Counter: 2
  Message: Hello World
  After direct modification: 7
```


### Demo8 - 指针与引用
