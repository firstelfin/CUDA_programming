<p>
    <center><h1>获得GPU加速的关键</h1></center>
	<br />
    <p name="top" id="top" align="center">
        <b>作者：</b><b><a href="https://www.cnblogs.com/dan-baishucaizi/">elfin</a></b>&nbsp;&nbsp;
        <b>资料来源：<a href="">CUDA编程与实践</a></b>
	</p>
</p>


---

[TOC]

---

​	前几章主要关注程序的正确性，没有强调程序的性能（执行速度）。在开发CUDA程序时往往要验证某些改变是否提高了程序的性能，这就需要对程序进行比较精确的计时。所以我们就从给主机核设备函数的计时讲起。

---

# 5.1 用CUDA事件计时

​	在c++中，有多种可以对一段代码进行计时的方法，包括使用GCC和MSVC都有的clock函数和头文件`<chrono>`对应的时间库、GCC中的`gettimeofday()`函数及MSVC中的`QueryPerformanceCounter()`和`QueryPerformanceFrequency()`函数等。CUDA提供了一种基于CUDA事件（CUDA event）的计时方式，可用来给一段CUDA代码(可能包含主机代码和设备代码)计时。为简单起见，我们这里仅介绍基于CUDA事件的计时方法。下面我们给出常用的计时方式：

```c++
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));
CHECK(cudaEventRecord(start));
cudaEventQuery(start);  // 此处不能用CHECK宏函数

需要计时的代码块

CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));
float elapsed_time;
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
printf("Time = %g ms .\n", elapsed_time);
CHECK(cudaEventDestroy(start));
CHECK(cudaEventDestroy(stop));
```

其中：

*   (1) 第1行定义了两个CUDA事件类型(cudaEvent_t)的变量start和stop，第2行和第3行用cudaEventCreate()函数初始化它们。
*   (2) 第4行将start传入cudaEventRecord()函数，在需要计时的代码块之前记录一个代表开始的事件。
*   (3) 第5行对处于TCC驱动模式的GPU来说可以省略，但对处于WDDM驱动模式的GPU来说必须保留。因为，在处于WDDM驱动模式的GPU中，一个CUDA流（CUDA stream）中的操作（如这里的cudaEventRecord()函数）并不是直接提交给GPU执行，而是先提交到一个软件队列，需要添加一条对该流的cudaEventQuery操作（或者cudaEventSynchronize）刷新队列，才能促使前面的操作在GPU执行。
*   (4) 第7行代表一个需要计时的代码块，可以是一段主机代码，也可以是一段设备代码，还可以是一段混合代码。
*   (5) 第9行将stop传入cudaEventRecord()函数，在需要计时的代码块之后记录一个代表结束的事件。
*   (6) 第10行的cudaEventSynchronize()函数让主机等待时间stop被记录完毕。
*   (7) 第11~13行调用cudaEventElapsedTime()函数计算start到stop的时间差并输出到屏幕(单位：ms)。
*   (8) 第15~16行调用cudaEventDestroy()函数销毁start和stop这两个CUDA事件。这是本笔记唯一使用事件的地方，故这里不对CUDA事件做进一步讨论。下面我们尝试对数组相加的代码块计时。

---

## 5.1.1 为C++程序计时

这里先考虑C++版本的程序。我们使用add1cpu.cu验证为C++程序计时，相对于第三章的add.cpp有如下的改动：

* (1) 即使该程序没有核函数，我们也将源文件的扩展名写为`.cu`，这样就不用包含一些CUDA头文件了。若还是使用`.cpp`，用nvcc 编译时需要明确地增加一些头文件的包含，用gcc编译时还要明确地链接一些CUDA库。

* (2) 从现在起，我们用调节编译的方式选择程序中所用浮点数的精度。在程序开头部分，添加如下几行代码：

  ```c++
  #ifdef USE_DP
      typedef double real;
      const real EPSILON = 1.0e-15;
  #else
      typedef float real;
      const real EPSILON = 1.0e-6f;
  #endif
  ```

  当宏USE_DP有定义时，程序的real代表double，否则代表float。该宏可以通过编译选项定义(具体见后吗的编译命令)。

* (3) 我们用CUDA时间对该程序中函数add()的调用进行了计时，而且重复了11次。我们忽略第一次的时间，因为第一次机器可能处于预热状态，测得的时间往往偏大。我们根据后10次测得时间的平均值标记add的执行用时。详情参考add1cpp.cu(我的实现)、add1cpp_.cu(樊老师的实现)。



**我们使用nvcc进行编译，需要注意以下几点：**

* (1) C++程序的性能显著地依赖于优化选项，我们将总是使用`-O3`选项。
* (2) 编译时使用`-DUSE_DP`的命令行参数，程序中的宏将有定义，从而使用双精度浮点数，否则使用单精度浮点数。

编译命令：

```shell
$ nvcc -O3 -arch=sm_86  add1cpu.cu -o add1float
$ ./add1float
Time = 171.325 ms .
No errors
$ nvcc -O3 -arch=sm_86  add1cpu_.cu -o add1float_
$ ./add1float_
Time = 220.226 ms.
Time = 87.8418 ms.
Time = 88.4715 ms.
Time = 89.0419 ms.
Time = 88.2954 ms.
Time = 86.6734 ms.
Time = 86.9325 ms.
Time = 86.8219 ms.
Time = 87.2632 ms.
Time = 87.2272 ms.
Time = 87.2445 ms.
Time = 87.5813 +- 0.7526 ms.
No errors
$ nvcc -O3 -arch=sm_86 -DUSE_DP  add1cpu.cu -o add1double
$ ./add1double
Time = 176.765 ms .
No errors
$ nvcc -O3 -arch=sm_86 -DUSE_DP  add1cpu_.cu -o add1double_
$ ./add1double_
Time = 399.767 ms.
Time = 175.862 ms.
Time = 175.972 ms.
Time = 176.373 ms.
Time = 175.953 ms.
Time = 176.814 ms.
Time = 176.315 ms.
Time = 175.655 ms.
Time = 176.055 ms.
Time = 176.876 ms.
Time = 175.824 ms.
Time = 176.17 +- 0.392806 ms.
No errors
```

这里add1cpu_.cu的双精度版本比单精度版本慢2倍，这和我们的预期是一样的，但是add1cpu.cu的双精度单精度却是一样的！这是为什么呢？

TODO：执行时间一样问题查找？

---













---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>












---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

<p name="bottom" id="bottom">
    <b>完！</b>
</p>