<p>
    <center><h1>CUDA程序的错误检测</h1></center>
	<br />
    <p name="top" id="top" align="center">
        <b>作者：</b><b><a href="https://www.cnblogs.com/dan-baishucaizi/">elfin</a></b>&nbsp;&nbsp;
        <b>资料来源：<a href="">CUDA编程与实践</a></b>
	</p>
</p>

---

[TOC]

---

​	编写CUDA程序，难免会出现各种各样的错误，有些错误在编译阶段就会被编译器捕捉，称为编译错误；有些错误可以正常编译，在运行时才会出错，称为运行时错误。本章就讨论如何检测运行时错误，包括使用一个检查CUDA运行时的API函数返回值的宏函数及使用CUDAMEMCHECK工具。

---

# 4.1 一个检测CUDA运行时错误的宏函数

​	前面我们使用的所有CUDA运行时API都是以cuda开头的，并且他们返回一个cudaError_t类型的返回值，成功是cudaSuccess，其他是返回错误代码。根据这样的规则，我们可以写一个如下的错误宏函数（error.cuh）：

```c++
#pragma once
#include <stdio.h>

# define CHECK(call)                                                                                                                \
do                                                                                                                                                     \
{                                                                                                                                                         \
    const cudaError_t error_code = call;                                                                            \
    if (error_code != cudaSuccess)                                                                                        \
    {                                                                                                                                                     \
        printf("CUDA Error: \n");                                                                                                \
        printf("    file:   %s\n", __FILE__);                                                                                \
        printf("    line:   %d\n", __LINE__);                                                                             \
        printf("    Error code: %d\n", error_code);                                                              \
        printf("    Error text: %s\n", cudaGetErrorString(error_code));                      \
        exit(1);                                                                                                                                    \
    }                                                                                                                                                     \
}while (0)
```

**error.cuh解释如下：**

* 该文件以`#pragma once`开头，这是一个预处理指令，其作用是确保当前文件在一个编译单元中不被重复包含。该指令等价于：

  ```c++
  #ifndef ERROR_CUH_
  #define ERROR_CUH_
  	头文件中的内容
  #endif
  ```

* 该宏函数的名称是CHECK，参数call是一个CUDA运行时API函数。

* 在定义宏时，如果一行写不下，则需要在行末写`\`，表示续行。

* 第7行定义了一个cudaError_t类型的变量error_code，并初始化为参数call的返回值。

* if语句判断是否返回成功运行。如果不是就打印哪个文件、哪一行、报了什么错误代码，并使用`cudaGetErrorString(error_code)`将错误代码转换成文字描述。



**宏函数的使用**

在使用宏函数时，只要将一个CUDA运行时API当作参数传入该宏函数即可，例如：

```c++
CHECK(cudaFree(d_x));
```



为什么我们要用一个do-while语句，这里显得好像很多余？不用这个语句实际上也是可以的，但是在某些情况下不安全。也可以不用宏函数，而用普通的函数，但此时必须将宏 `__FILE__` 和 `__LINE__` 传给该函数，这样用起来不如宏函数简洁。

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 4.1.1 检查运行时API函数

​	这里我们对第3章的`add3error.cu`进行CUDA运行时API函数都使用宏函数CHECK进行封装，得到`check1api.cu`：

```c++
#include "error.cuh"
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
	double *d_x, *d_y, *d_z;
	CHECK(cudaMalloc ((void **) &d_x, M));
	CHECK(cudaMalloc ((void **) &d_y, M));
	CHECK(cudaMalloc ((void **) &d_z, M));
	CHECK(cudaMemcpy(d_x, h_x, M,  cudaMemcpyDeviceToHost)); // 测试错误的传输方向
	CHECK(cudaMemcpy(d_y, h_y, M,  cudaMemcpyDeviceToHost));

	const int block_size = 128;
	const int grid_size = N / block_size;
	add<<<grid_size, block_size>>>(d_x, d_y, d_z);
	CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
	
	check(h_z, N);
	free(h_x);
	free(h_y);
	free(h_z);
	CHECK(cudaFree(d_x));
	CHECK(cudaFree(d_y));
	CHECK(cudaFree(d_z));
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

编译后执行有：

```shell
$ ./check1 
CUDA Error: 
    file:   check1api.cu
    line:   30
    Error code: 1
    Error text: invalid argument
```

这里执行的时候成功地定位到了我们故意写的错误，用了宏函数我们就可以准确地定位错误了，而不是一个错误的运行结果。从现在起，我们应该用这个宏函数封装大部分CUDA运行时API函数。**有一个例外是cudaEventQuery()函数，因为它很可能返回CUDAErrorNotReady，但又不代表程序出错了**。

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 4.1.2 核函数的检查

​	上面的方法是从返回值进行封装的宏函数，但是核函数是没有返回值的，这怎么检查呢？有一个方法可以捕捉调用核函数可能发生的错误，即在调用核函数之后加上下面两条语句：

```c++
CHECK(cudaGetLastError());
CHECK(cudaDeviceSynchronize());
```

​	第一条语句是捕捉第二条语句之前的最后一个错误，第二条语句是主机与设备同步。之所以要同步主机与设备，是因为核函数的调用是异步的，即主机发出调用核函数的命令后会立即执行后面的语句，不会等核函数执行完。需要注意的是，同步操作是比较耗时的，如果在程序的内层循环调用，很可能会严重降低程序的性能。只要核函数的调用之后还有对其他任何能返回错误值的API进行同步调用，都能触发主机与设备的同步并捕捉到核函数调用中可能发生的错误。

​	为了展示核函数错误检查，我们将第3章中的add进行简单更改，将block_size 改为1280，因为现在的GPU限制block_size最大为1024，所以我们看是否能捕捉到这个错误(代码见：[check2kelnel.cu](../CUDA文件/Chapter04/check2kelnel.cu))：

```shell
$ nvcc -arch=sm_86 check2kernel.cu -o check2
$ ./check2
CUDA Error: 
    file:   check2kernel.cu
    line:   36
    Error code: 9
    Error text: invalid configuration argument
```

​	这里的36行即为`CHECK(cudaGetLastError());`，说明错误已经定位到了，但是注意这里的指向是核函数add，如果你把两条语句再往后放，可能就不能很好定位了，这也就是有这两句还是要将运行时API用CHECK封装的原因。

这里不写同步函数也可以捕捉到异常，因为后面数据传输函数隐式同步了主机与设备。一般情况下要获得精确的出错位置，还是显示的同步，如使用`cudaDeviceSynchronize()`，或者使用环境变量`export CUDA_LAUNCH_BLOCKING=1`。这样设置环境变量后所有的核函数调用都是同步的，但是这非常影响性能，所以建议还是老实用上面的两条语句。

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

# 4.2 用CUDA-MEMCHECK检查内存错误

CUDA提供了一个CUDA-MEMCHECK工具，包括memcheck、racecheck、initcheck、synccheck共四个工具。它们可以由可执行文件cuda-memcheck调用：

```shell
$ cuda-memcheck --tool memcheck [options] app_name [options]
$ cuda-memcheck --tool racecheck [options] app_name [options]
$ cuda-memcheck --tool initcheck [options] app_name [options]
$ cuda-memcheck --tool synccheck [options] app_name [options]
```

对于memcheck工具，可以简化为：

```shell
$ cuda-memcheck  [options] app_name [options]
```



**内存错误案例**

我们将check1api.cu的数据传输方向修改为正常，并修改`const int N = 1e8+1;`，以造成内存错误。

```shell
$ nvcc -arch=sm_86 memcheck1.cu -o check3
$ ./check3
Has errors
```

这里到底是哪里有问题，我们根本不知道，所以现在使用CUDA-MEMCHECK检查内存工具：

```shell
$ cuda-memcheck --tool memcheck ./check3
……
CUDA Error: 
    file:   memcheck1.cu
    line:   36
    Error code: 719
    Error text: unspecified launch failure
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaMemcpy.
=========     Saved host backtrace up to driver entry point at error
=========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x3bd253]
=========     Host Frame:./check3 [0x5b71d]
=========     Host Frame:./check3 [0x7aaf]
=========     Host Frame:/lib/x86_64-linux-gnu/libc.so.6 (__libc_start_main + 0xe7) [0x21c87]
=========     Host Frame:./check3 [0x755a]
=========
========= ERROR SUMMARY: 33 errors
```

添加if语句后:

```shell
$ cuda-memcheck ./check4
========= CUDA-MEMCHECK
No errors
========= ERROR SUMMARY: 0 errors
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