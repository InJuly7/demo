#include <iostream>
#include <array>
using namespace std;

// Demo1 - constexpr变量基础示例
#if 0
constexpr int compile_time_val = 42;           // 编译期常量
constexpr double pi = 3.14159;                // 编译期浮点常量
constexpr char hello[] = "Hello constexpr";   // 编译期字符串

int main() { 
    cout << "=== Demo1 - constexpr变量基础 ===" << endl;
    // constexpr变量必须在编译期就能确定值
    constexpr int size = 10;
    int arr[size];  // 可以用作数组大小，因为是编译期常量
    
    cout << "compile_time_val: " << compile_time_val << endl;
    cout << "pi: " << pi << endl;
    cout << "hello: " << hello << endl;
    cout << "数组大小（编译期确定）: " << size << endl;
    
    // const vs constexpr
    const int runtime_const = rand();      // 运行期常量
    // constexpr int error = runtime_const;       // 错误！不能在编译期确定
    cout << endl;
    return 0; 
}
#endif

// Demo2 - constexpr与指针示例
#if 0
int global_var = 100;
const int const_global = 200;
constexpr int constexpr_global = 300;

int main()
{
    cout << "=== Demo2 - constexpr指针 ===" << endl;

    // 1. constexpr指针：指针本身是常量（不能改变指向）
    constexpr int *ptr1 = &global_var; // ptr1是常量指针，指向global_var
    // ptr1 = &const_global;                    // 错误！不能改变constexpr指针的指向
    *ptr1 = 101; // 可以修改所指向的值

    // 2. 指向编译期常量的constexpr指针
    constexpr const int *ptr2 = &constexpr_global; // ptr2指向编译期常量
    // *ptr2 = 301;                             // 错误！不能修改常量

    // 3. constexpr指向编译期常量（最严格）
    constexpr const int *ptr3 = &constexpr_global;

    // 4. 数组与constexpr指针
    static constexpr int arr[] = {1, 2, 3, 4, 5}; //编译器确定 数组地址 (全局数组, 静态局部数组)
    constexpr const int *arr_ptr = arr; // 指向数组首元素的constexpr指针

    cout << "global_var通过constexpr指针修改后: " << global_var << endl;
    cout << "constexpr数组元素: " << arr_ptr[2] << endl;

    // 5. nullptr与constexpr
    constexpr int *null_ptr = nullptr; // constexpr空指针

    cout << endl;
    return 0;
}
#endif

// Demo3 - constexpr函数示例
#if 0
// constexpr函数：如果参数是编译期常量，则在编译期求值
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int fibonacci(int n) {
    return (n <= 1) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}

// 更复杂的constexpr函数
constexpr int power(int base, int exp) {
    int result = 1;
    for (int i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

int main() { 
    cout << "=== Demo3 - constexpr函数 ===" << endl;
    // 编译期计算
    constexpr int fact5 = factorial(5);         // 编译期计算 5!
    constexpr int fib10 = fibonacci(10);        // 编译期计算第10个斐波那契数
    constexpr int pow_result = power(2, 10);    // 编译期计算 2^10
    
    cout << "5! = " << fact5 << endl;
    cout << "第10个斐波那契数 = " << fib10 << endl;
    cout << "2^10 = " << pow_result << endl;
    
    // 运行期计算（当参数不是编译期常量时）
    int runtime_n = 6;
    int runtime_fact = factorial(runtime_n);    // 运行期计算
    cout << "运行期计算 6! = " << runtime_fact << endl;
    
    // constexpr函数可以用于需要编译期常量的地方
    constexpr int size = power(2, 3);
    int compile_time_array[size];               // 数组大小在编译期确定
    
    cout << endl;
    return 0; 
}
#endif

// Demo4 - constexpr与类/构造函数示例
// constexpr可以修改对象（C++14起）
#if 0
// constexpr类（字面值类型）
class Point {
private:
    int x_, y_;
    
public:
    // constexpr构造函数
    constexpr Point(int x = 0, int y = 0) : x_(x), y_(y) {}
    
    // constexpr成员函数
    constexpr int x() const { return x_; }
    constexpr int y() const { return y_; }
    constexpr int distance_squared() const {
        return x_ * x_ + y_ * y_;
    }
    
    // constexpr可以修改对象（C++14起）
    constexpr void move(int dx, int dy) {
        x_ += dx;
        y_ += dy;
    }
};

constexpr Point create_points() {
    Point p(3, 4);
    p.move(1, 1);  // C++14起支持在constexpr函数中修改
    return p;
}

int main() { 
    cout << "=== Demo4 - constexpr类与构造函数 ===" << endl;
    
    // 编译期创建对象
    constexpr Point origin;                     // (0, 0)
    constexpr Point p1(3, 4);                  // (3, 4)
    constexpr Point p2 = create_points();      // 编译期函数调用
    
    // 编译期计算成员函数
    constexpr int dist_sq = p1.distance_squared();  // 编译期计算 3²+4² = 25
    
    cout << "原点坐标: (" << origin.x() << ", " << origin.y() << ")" << endl;
    cout << "点p1坐标: (" << p1.x() << ", " << p1.y() << ")" << endl;
    cout << "点p1到原点距离平方: " << dist_sq << endl;
    cout << "函数创建的点: (" << p2.x() << ", " << p2.y() << ")" << endl;
    
    // constexpr对象可以用于需要编译期常量的地方
    constexpr int array_size = p1.x() + p1.y();  // 3 + 4 = 7
    int compile_time_sized_array[array_size];
    
    cout << "编译期确定的数组大小: " << array_size << endl;
    cout << endl;
    return 0; 
}
#endif

// Demo5 - constexpr综合应用示例
// constexpr if (C++17)
#if 1
#include <array>

// 编译期计算素数
constexpr bool is_prime(int n) {
    if (n < 2) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

// 编译期生成素数数组
constexpr std::array<int, 10> generate_primes() {
    std::array<int, 10> primes = {};
    int count = 0;
    int candidate = 2;
    
    while (count < 10) {
        if (is_prime(candidate)) {
            primes[count++] = candidate;
        }
        ++candidate;
    }
    return primes;
}


int main() { 
    cout << "=== Demo5 - constexpr综合应用 ===" << endl;
    
    // 编译期生成前10个素数
    constexpr auto primes = generate_primes();
    
    cout << "前10个素数（编译期计算）: ";
    for (int prime : primes) {
        cout << prime << " ";
    }
    cout << endl;
    
    // constexpr if (C++17)
    auto process_number = [](auto n) {
        if constexpr (sizeof(n) == sizeof(int)) {
            cout << "处理int类型: " << n << endl;
        } else {
            cout << "处理其他类型: " << n << endl;
        }
    };
    
    process_number(42);
    process_number(42L);
    
    cout << endl;
    return 0;
}
#endif
