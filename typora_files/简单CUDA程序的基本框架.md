<p>
    <center><h1>CUDA中的线程组织</h1></center>
	<br />
    <p name="top" id="top" align="center">
        <b>作者：</b><b><a href="https://www.cnblogs.com/dan-baishucaizi/">elfin</a></b>&nbsp;&nbsp;
        <b>资料来源：<a href="">CUDA编程与实践</a></b>
	</p>
</p>

---

[TOC]

---

# 3.1 数组相加C++版

我们直接上代码(C++代码比较简单)：

```c++
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
	const int N = 1e8;
	const int M = sizeof(double) * N;
	double  *x = (double *) malloc(M);
	double  *y = (double *) malloc(M);
	double  *z = (double *) malloc(M);
	for (int n = 0; n < N; ++n)
	{
		x[n] = a;
		y[n] = b;
	}
	add(x, y, z, N);
	check(z, N);
	free(x);
	free(y);
	free(z);
	return 0;
}

void add(const double *x, const double *y, double *z, const int N)
{
	for (int n = 0;  n <N; ++n)
	{
		z[n] = x[n] + y[n];
	}
}

void check(const double* z, const int N)
{
	bool has_error = false;
	for (int n = 0; n < N; ++n)
	{
		if (fabs(z[n] - c) > EPSILON)
		{
			has_error = true;
		}
	}
	printf("%s\n" , has_error ? "Has errors" : "No errors");
}
```

该程序的解释如下：

*   第17~19行先定义了3个双精度浮点类型的指针变量，然后将他们指向由函数malloc()分配的内存，从而得到3个长度为$10^{8}$的一维数组。
*   第22~23行是将数组x、y的所有元素全部初始化为1.23 和 2.34。
*   第27~29行释放内存，这非常重要！

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

# 3.2 CUDA程序的基本框架

一个典型的CUDA程序的基本框架

```txt
头文件包含
常量定义(或者宏定义)
C++自定义函数和CUDA核函数的声明(原型)
int main(void)
{
	分配主机和设备内存
	初始化主机中的数据
	将某些数据从主句复制到设备
	调用核函数在设备上进行计算
	将某些数据从设备复制到主机
	释放主机与设备内存
}
c++ 自定义函数和CUDA核函数的定义(实现)
```

以上面求和为例，书写一个cuda程序：

```c++
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void __global__ add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main(void)
{
	const int N = 1e8;
	const int M = sizeof(double) * N;
	double *h_x = (double *) malloc(M);
	double *h_y = (double *) malloc(M);
	double *h_z = (double *) malloc(M);
	for (int n = 0; n < N; ++n)
	{
		h_x[n] = a;
		h_y[n] = b;
	}
	double *d_x, *d_y, *d_z;          // 这里定义了double类型的指针变量
	cudaMalloc ((void **) &d_x, M);   // 这里将指针变量指向GPU的地址
	cudaMalloc ((void **) &d_y, M);
	cudaMalloc ((void **) &d_z, M);
	cudaMemcpy(d_x, h_x, M,  cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, M,  cudaMemcpyHostToDevice);

	const int block_size = 128;
	const int grid_size = N / block_size;
	add<<<grid_size, block_size>>>(d_x, d_y, d_z);
	cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
	
	check(h_z, N);
	free(h_x);
	free(h_y);
	free(h_z);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	return 0;
}

void __global__ add(const double *x, const double *y, double *z)
{
	const int n = blockDim.x * blockIdx.x + threadIdx.x;
	z[n] = x[n] + y[n];
}

void check(const double* z, const int N)
{
	bool has_error = false;
	for (int n = 0; n < N; ++n)
	{
		if (fabs(z[n] - c) > EPSILON)
		{
			has_error = true;
		}
	}
	printf("%s\n", has_error ? "Has errors" : "No errors");
}
```

注意，这里我们的网格大小是$10^{8} / 128 = 781250$，这个大小比较大，如果是使用CUDA8.0，而且又没有指定计算能力，则默认是2.0的计算能力，而对于该计算能力，网格大小在每个方向上的限制为65535。

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 3.2.1 隐形的设备初始化

​	在 CUDA runtime的API中，没有明显初始化设备的函数。在第一次调用一个与设备管理及版本查询功能无关的runtime API时，设备会自动初始化。如上面的代码中cudaMalloc。

## 3.2.2 设备内存的分配与释放

​	在上面的程序中，我们还是在主体上定义了3个数组，然后使用又定义了3个数组，并使用cudaMalloc将其指向了GPU设备。所有的CUDA runtime API都是以cuda开头，具体详情参考：https://docs.nvidia.com/cuda/cuda-runtime-api .

正如C++中使用`malloc()`分配内存一样，CUDA中使用`cudaMalloc()`进行显存的动态分配。该函数的原型如下：

```c++
cudaError_t cudaMalloc(void **address, size_t size);
```

其中：

*   address是待分配设备内存的指针。因为内存(地址)本身就是一个指针，所以待分配设备内存的指针就是指针的指针，即双重指针。
*   第二个参数是待分配的内存字节数。
*   返回是一个错误代码，如果调用成功，则返回`cudaSuccess`。



​	调用函数`cudaMalloc()`时传入的第一个参数`(void **)&d_x`比较难理解。我们知道`d_x`是一个double类型的指针，那么他的地址就是double类型的双重指针。而`(void **)`就是一个强制类型转换操作。这种类型转换可以不明确写出来，即调用的时候直接写为`cudaMalloc(&d_x, M)`。



在cuda中我们需要使用`cudaFree()`函数释放显存，该函数的原型是：

```c++
cudaError_t cudaFree(void* address);
```

参数address指向就是待释放的设备内存中的变量（不是双重指针），返回值是一个错误代码，成功调用就返回`cudaSuccess`。



主机内存的分配可由new、delete分配释放；使用free释放显存程序退出后会出错。

```shell
$ ./add2free 
No errors
段错误 (核心已转储)
```



---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 3.2.3 主机与设备之间数据的传递

CUDA runtime API的`cudaMemcpy()`的原型：

```c++
cudaError_t cudaMemcpy
(
	void                *dst,
    const void          *src,
    size_t              count,
    enum cudaMemcpyKind kind
);
```

其中：

*   (1) 第一个参数dst是目标地址

*   (2) 第二个参数src是源地址

*   (3) 第三个参数count是复制数据的字节数

*   (4) 第四个参数kind是一个枚举类型的变量，标志数据传递方向，只能取：
    *   1）cudaMemcpyHostToHost，表示从主机复制到主机
    *   2）cudaMemcpyHostToDevice，表示从主机复制到设备
    *   3）cudaMemcpyDeviceToHost，表示从设备复制到主机
    *   4）cudaMemcpyDeviceToDevice，表示从设备复制到设备
    *   5）cudaMemcpyDefault，根据src与dst所指地址自动判断传输的方向。（这要求系统具有虚拟寻址功能，即64位系统）

测试错误的传输方向：

```shell
$ ./add3error 
Has errors
```



---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 3.2.4 核函数中数据与线程的对应

​	c++版本的add与cuda版本的add，两者之间的差异主要是设备函数中没有循环，它是用“单指令-多线程”的方式编写代码，所以可以去掉循环，只需将数组元素指标与线程指标一一对应即可。

我们在核函数指定了数组索引为：

```c++
const int n = blockDim.x * blockIdx.x + threadIdx.x;
```

对于每一个线程，`blockDim.x`、`blockIdx.x`、`threadIdx.x`是线程的自带属性，这样我们生成了网格中的唯一表示n，用它作为数组的索引，就可以实现每个线程操作一次加法操作了！

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 3.2.5 核函数的要求

CUDA核函数的编写是重要的部分，我们编写时需要注意以下几点：

1.  核函数的返回必须是void。所以核函数中可以用return关键字，但不可返回任何值。
2.  必须使用限定符`__global__`。也可以加上一些其他C++中的限定符，如static。限定符的次序可任意。
3.  函数名没有特殊要求，且支持C++中的重载，即可以用同一个函数名表示具有不同的参数列表的函数。
4.  不支持可变参数数量的参数列表，即参数的个数必须确定。
5.  可以向核函数传递非指针变量（如例子中的`int N`），其内容对每个线程可见。(用执行配置<<<1,1>>>时，核函数使用循环)
6.  除非使用统一内存编程机制，否则传给核函数的数组(指针)必须指向设备内存。
7.  核函数不能是类的成员。通常的做法是用一个包装函数调用核函数，而将包装函数定义为类的成员。
8.  计算能力3.5之后，核函数可以调用核函数，因为有动态并行机制，甚至可以自己调用自己。
9.  无论从主机调用还是从设备调用，核函数都是在设备中执行。调用核函数时，必须指定执行配置，即三括号及其中的参数。



## 3.2.6 核函数中if语句的必要性

​	前面的add1.cu中我们并没有使用N，当N是blockDim.x的整数倍时，不会引起问题；反之就会出现问题。如$N=10^{8} + 1$，那么如果我们依然取block_size为128，N就不能被整除。此时，我们面临一个grid_size的选择问题，我们选择N整除block_size，余数为1，那么有一个元素无法处理，更一般的处理是使用如下的grid_size:

```c++
int grid_size = (N-1) / block_size + 1;
// 或者
int grid_size = (N + block_szie - 1) / block_size;
// 上面两种表述等价于
int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
```

虽然核函数不能有返回值，但是还是可以用return的，所以我们可以进行如下的修改：

```c++
void __global__ add(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= N) return;
    z[n] = x[n] + y[n];
}
```

不进行if语句的控制，就可能出现非法的设备内存操作。

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

# 3.3 自定义设备函数

​	核函数可以调用不带执行配置的自定义函数，这样的自定义自定义函数称为设备函数。它在设备中执行，且只能在设备中调用。而核函数是在设备中执行，但是在主机中调用。

## 3.3.1 函数执行空间标识符

在CUDA程序中，由以下标识符确定一个函数在哪里被调用，以及在哪里被执行：

* (1) 用 `__global__` 修饰的函数称为核函数，一般由主机调用，在设备中执行。如果使用动态并行，则可以在核函数中调用自己或者其他核函数。
* (2) 用 `__device__` 修饰的函数称为设备函数，只能由核函数和其他设备函数调用，在设备中执行。
* (3) 用 `__host__` 修饰的函数就是主机端的普通C++函数，主机中调用、执行。对于主机端的函数，该修饰符可以省略。之所以提供这样一个修饰符，是因为有时会用 `__host__` 和 `__device__` 同时修饰一个函数，使得该函数既是主机中的普通C++函数，又是一个设备函数。这样做可以减少冗余代码。编译器将针对主机和设备分别编译该函数。
* (4) 不能同时用 `__device__` 和 `__global__` 修饰一个函数，即不能同时是设备函数和核函数。
* (5) 也不能同时用 `__host__` 和 `__global__` 修饰一个函数，即不能同时为主机函数和核函数。
* (6) 编译器决定把设备函数当作内联函数或非内联函数，当然我们也可以使用 `__noinline__` 修饰符建议一个设备函数为非内联函数(编译器不一定接受)，也可以用修饰符 `__forceinline__` 建议一个设备函数为内联函数。



## 3.3.2 为数组相加核函数定义一个设备函数

```c++
// 版本1：有返回值的设备函数
double __device__ add1_device(const double x, const double y)
{
    return (x + y);
}

void __global__ add1(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= N) return;
    z[n] = add1_device(x[n], y[n]);
}

// 版本2：用指针的设备函数
void __device__ add2_device(const double x, const double y , double *z)
{
    *z = x + y;
}

void __global__ add2(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= N) return;
   add2_device(x[n], y[n], &z[n]);
}

// 版本3：用引用(reference)的设备函数
void __device__ add3_device(const double x, const double y, double &z)
{
    z = x + y;
}

void __global__ add3(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n >= N) return;
   add3_device(x[n], y[n], z[n]);
}
```







---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

<p name="bottom" id="bottom">
    <b>完！</b>
</p>