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