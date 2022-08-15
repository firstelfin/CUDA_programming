<p>
    <center><h1>CUDA中的线程组织</h1></center>
	<br />
    <p name="top" id="top" align="center">
        <b>作者：</b><b><a href="https://www.cnblogs.com/dan-baishucaizi/">elfin</a></b>&nbsp;&nbsp;
        <b>资料来源：<a href="">CUDA编程与实践</a></b>
	</p>
</p>

[TOC]

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

> 注：本章实验环境Linux18.04  CUDA11.3  GPU3060

# 2.1 C++语言中的hello world程序

​	我们用VS Code书写一个如下的hello world程序：

```c++
# include <stdio.h>

int main(void){
    printf("Hello World!\n");
    return 0;
}
```

使用gcc编译为可执行文件：

```shell
$ gcc  hello.cpp -o hello
$ ls
hello  hello.cpp
$ ./hello
Hello World!
```

至此一个最简单的c++程序就被书写出来了。

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

# 2.2 CUDA中的hello world程序

## 2.2.1 只有主机函数的CUDA程序

​	其实上面的程序就可以是一个cuda程序，因为nvcc工具支持编译C++代码。一般CUDA文件是包含C++代码，也包含真正的CUDA代码，nvcc在编译C++部分代码时会将这部分编译工作交给C++的编译器，nvcc自己只处理剩余的部分！

**编译hello.cpp为一个CUDA文件**

```shell
# 先将hello.cpp修改为hello.cu
$ nvcc hello.cu
$ ls
a.out  hello  hello.cpp  hello.cu
$ ./a.out
Hello World!
```

这里也执行成功了！但是我们不知道程序是真的在GPU上面执行的吗？实际上程序就没有使用GPU，我使用`watch -n 1 nvidia-smi`也没有看见任何相关进程。

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 2.2.2 使用核函数的CUDA程序

​	要使GPU工作，主机需要下达明确的指令。一个真正利用了GPU的CUDA程序，需要有**主机代码**(前面的a.out就全是主机代码)与**设备代码**(需要GPU设备执行的代码)。主机对设备的调用都是通过核函数来实现的。

**典型的CUDA程序结构**

一个典型的CUDA程序应该具有如下的形式：

```c++
int main(viod){
    主机代码;
    核函数的调用;
    主机代码;
    return 0;
}
```



**核函数的要求**

必须使用限定词`__global__`修饰；返回必须是`void`。这两个词的顺序是可调换的。

下面我们写一个hello2.cu：

```c++
# include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello wolrd from the GPU\n");
}

int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```



**编译CUDA文件**

```shell
$ nvcc hello2.cu -o hello2
$ ./hello2
Hello wolrd from the GPU
```

此时，我们使用`watch -n 1 nvidia-smi`就会监听到这个进程(*执行时间很短，可能需要多执行几次才能看见*)。

**CUDA函数的解释**

1. **核函数的调用**

   **`hello_from_gpu<<<1, 1>>>();`**(这种结构我们把它叫做**执行配置**)

   这和C++程序调用是不一样的，这里多了`<<<i, j>>>`的结构。这个结构是因为我们的设备有多个计算核心，可以支持很多线程。因此主机在调用核函数时，必须指定线程的分配，三括号中的数字就是指定线程数目及其排列情况。这个结构表示：<<<block_num, thread_num>>>，这个和GPU显卡的架构有关，我们一个核函数分配一个grid网格(实际就是你指定的所有block)，一个grid是由很多block组成的，也即指定的第一个参数block_num；每一个block又是由很多线程组成，也即我们指定的第二个参数。这里的<<<1, 1>>>都是一维数据，实际上我们可以指定高维的，后面会讲到这种情况。

2. 核函数的printf的使用方式和C++一样，同时也是需要导入头文件<stdio.h>。但是核函数不支持C++的**iostream**。经过测试：核函数中使用cout直接报错 未定义标识符‘cout’。

3. **cudaDeviceSynchronize()的使用**

   这是一个CUDA的runtime API，不写这个就不能输出字符串。因为调用输出函数时，输出流是先存放在缓存区的，而缓存区不会自动刷新，只有遇到某种同步操作缓存区才会刷新。cudaDeviceSynchronize()就是同步主机与设备，所以缓存区能够被刷新。



---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

# 2.3 CUDA中的线程组织

## 2.3.1 使用多个线程的核函数

```c++
# include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello wolrd from the GPU\n");
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

网格的大小是2，block的大小是4，所以总的线程数量是$2 \times 4 = 8$。即该程序中，核函数将指派8个线程。核函数中代码的执行方式是**单指令-多线程**模式，即每一个线程都执行同一串指令。所以上面的文件编译后，执行，屏幕将打印8行相同的数据！

```shell
$ ./hello3
Hello wolrd from the GPU
Hello wolrd from the GPU
Hello wolrd from the GPU
Hello wolrd from the GPU
Hello wolrd from the GPU
Hello wolrd from the GPU
Hello wolrd from the GPU
Hello wolrd from the GPU
```

上面的输出我们都不知道是那个线程的输出，下面将讨论这个问题。

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 2.3.2 使用线程索引

​	如前面所述，一个核函数可以指派多个线程，而这些线程的组织结构由**执行配置 <<<grid_size, block_size>>>**决定。其中grid_size就是block的“数量”，block就是线程的“数量”，注意这里的数量是引号，它可能并不是一个数值，而是一个结构体(标识一个多维数组，即grid_size、block_size是可以有x,y,z分量的，不是必须得是整型数据)。

​	目前的GPU架构block_size最大为1024，线程束大小为32(固定，可参考[博客](https://blog.csdn.net/weixin_44742084/article/details/125225243))，grid_size每个维度可以达到$2^{31}-1$。这个线程数量太大了，目前我们的设备核心数或者说流处理器单元肯定没有那么多，所以当指定的线程数量比GPU计算核心数大的时候，我们就能充分利用计算资源了。虽然我们可以指定很多线程，但是执行核函数时，能够同时活跃的线程数是由设备的CUDA核心和核函数中的代码决定的。

​	为了控制每一个线程，核函数中由执行配置指定的所有线程都有一个唯一的身份标识。这个身份标识和执行配置的参数grid_size、block_size有关：

* gridDim.x：该数值等于执行配置中的grid_size数值；
* blockDim.x：该数值等于执行配置中的block_size数值；

类似地，核函数预定义了标识线程的内建变量：

* blockIdx.x：标识了某个线程在网格中的线程块的索引，取值范围为$[0, grid\_size -1]$
* threadIdx.x：指定某个线程在一个线程块中的索引，取值范围为$[0, block\_size -1]$

下面我们使用上面的内建变量查看输出是由哪个线程控制的。这里修改上面的cuda文件为hello4.cu：

```c++
# include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Hello wolrd from  block %d and thread %d!\n", bid, tid);
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

编译后执行结果如下：

```shell
$ ./hello4
Hello wolrd from  block 0 and thread 0!
Hello wolrd from  block 0 and thread 1!
Hello wolrd from  block 0 and thread 2!
Hello wolrd from  block 0 and thread 3!
Hello wolrd from  block 1 and thread 0!
Hello wolrd from  block 1 and thread 1!
Hello wolrd from  block 1 and thread 2!
Hello wolrd from  block 1 and thread 3!
$ ./hello4
Hello wolrd from  block 1 and thread 0!
Hello wolrd from  block 1 and thread 1!
Hello wolrd from  block 1 and thread 2!
Hello wolrd from  block 1 and thread 3!
Hello wolrd from  block 0 and thread 0!
Hello wolrd from  block 0 and thread 1!
Hello wolrd from  block 0 and thread 2!
Hello wolrd from  block 0 and thread 3!
```

**<font color="red">这里反应了一个很重要的事实，即线程块block之间是独立的，执行顺序是随机的！</font>**



## 2.3.3 执行配置推广到多维

前面的内建变量都使用了C++的结构体或者类的成员变量的语法，其中：

- **<font color="red">blockIdx和threadIdx是类型为uint3的变量</font>**，它的定义为：

  ```c++
  struct __device__builtin__ uint3
  {
      unsigned int x, y, z;
  };
  typedef __device_builtin__ struct uint3 uint3;
  ```

  即该结构体是由3个无符号证书类型的成员构成。

- **<font color="red">gridDim和blockDim是类型为dim3的变量</font>**

  该类型和uint3类似，定义为：

  ```c++
  struct __device_builtin__ dim3
  {
      unsigned int x, y, z;
  #if defined(__cplusplus)
  #if __cplusplus >= 201103L
      __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
      __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
      __host__ __device__ constexpr operator uint3(void) const { return uint3{x, y, z}; }
  #else
      __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
      __host__ __device__ dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
      __host__ __device__ operator uint3(void) const { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
  #endif
  #endif /* __cplusplus */
  };
  typedef __device_builtin__ struct dim3 dim3;
  ```

这些内建变量都只在核函数中有效，且满足：

* blockIdx.x 的取值范围是从 0 到 gridDim.x-1
* blockIdx.y 的取值范围是从 0 到 gridDim.y-1
* blockIdx.z 的取值范围是从 0 到 gridDim.z-1
* threadIdx.x 的取值范围是从 0 到 blockDim.x-1
* threadIdx.y 的取值范围是从 0 到 blockDim.y-1
* threadIdx.z 的取值范围是从 0 到 blockDim.z-1



下面我们就使用这两种结构体创建多维网格(hello5.cu)：

```c++
# include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello wolrd from  block-(%d,%d) and thread-(%d,%d)!\n", bx, by, tx, ty);
}

int main(void)
{
    const dim3 gridsize(2, 2);  // grid_size(2, 2)也可以
    const dim3 blocksize(2, 2);
    hello_from_gpu<<<gridsize, blocksize>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

编译执行后，显示：

```shell
$ ./hello5
Hello wolrd from  block-(0,0) and thread-(0,0)!
Hello wolrd from  block-(0,0) and thread-(1,0)!
Hello wolrd from  block-(0,0) and thread-(0,1)!
Hello wolrd from  block-(0,0) and thread-(1,1)!
Hello wolrd from  block-(1,1) and thread-(0,0)!
Hello wolrd from  block-(1,1) and thread-(1,0)!
Hello wolrd from  block-(1,1) and thread-(0,1)!
Hello wolrd from  block-(1,1) and thread-(1,1)!
Hello wolrd from  block-(1,0) and thread-(0,0)!
Hello wolrd from  block-(1,0) and thread-(1,0)!
Hello wolrd from  block-(1,0) and thread-(0,1)!
Hello wolrd from  block-(1,0) and thread-(1,1)!
Hello wolrd from  block-(0,1) and thread-(0,0)!
Hello wolrd from  block-(0,1) and thread-(1,0)!
Hello wolrd from  block-(0,1) and thread-(0,1)!
Hello wolrd from  block-(0,1) and thread-(1,1)!
```



需要注意的是tread的索引出现是有固定规律的：(0,0)、(1,0)、(0,1)、(1,1)。实际上，cuda的线程块内，多维情况下，x轴的取值变换最快，z轴最慢。如果我们要计算某个线程在block中的顺序时，可以使用如下的公式计算：
$$
\text{tid} = {\color{Red}\text{treadIdx.x} } + {\color{Green}\text{blockDim.x}\times \text{treadIdx.y} }+ {\color{Orange}\text{blockDim.x}\times \text{blockDim.y} \times \text{treadIdx.z}}
$$
多维网格中，block是没有顺序的，但是我们可以类似地定义其平铺的索引：
$$
\text{bid} = {\color{Red}\text{blockIdx.x} } + {\color{Green}\text{gridDim.x}\times \text{blockIdx.y} }+ {\color{Orange}\text{gridDim.x}\times \text{gridDim.y} \times \text{blockIdx.z}}
$$


---

**<font color="red">不管怎样定义你的线程块，都需要注意线程块最大只能有1024个线程。</font>**

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

# 2.4 CUDA的头文件

​	在编写C++文件时，我们都需要导入一些标准的头文件，也许你已经注意到了，我们的cu文件只包含了C++的<stdio.h>文件，并没有包含其他任何CUDA头文件。实际上CUDA也有一些头文件，只是在使用ncvv进行编译的时候，将自动包含必要的头文件，如**<font color="red"><cuda.h></font>**和**<font color="red"><cuda_runtime.h></font>**。这里<cuda.h>是包含<stdio.h>的，所以nvcc甚至不需要在cu文件中包含<stdlib.h>文件。当然这些头文件你还是可以写的，这里也建议作为初学者的自己书写这些头文件。

​	在使用一些利用CUDA进行加速的应用程序库时，需要包含一些必要的头文件，并有可能还需要指定链接选项。(TODO: 链接学习资源第14章)

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

# 2.5 用nvcc编译CUDA程序

## 2.5.1 虚拟架构与真实架构计算能力

​	CUDA编译器nvcc现将全部源代码分离为主机代码和设备代码。主机代码是完全支持C++语法的，但设备代码只部分支持C++。nvcc是将设备代码编译为PTX(parallel thread execution)伪汇编代码，再将PTX代码编译为二进制的cubin目标代码。在将源代码编译为PTX代码时，需要用选项`-arch=compute_XY`指定一个虚拟框架的计算能力，用以确定代码中能够使用的CUDA功能。在PTX编译为cubin代码时，需要使用选项`-code=sm_ZW`指定一个真实架构的计算能力，用以确定可执行文件能够使用的GPU。真实架构的计算能力必须不小于虚拟架构的计算能力。如：

```c++
-arch=compute_35  -code=sm_60
```

千万不能写为`-arch=compute_60  -code=sm_35`。如果仅仅针对一个GPU编译程序，一般情况下可以将以上两个计算能力都选为GPU的计算能力。



## 2.5.2 真实架构计算能力参数说明

​	用以上方式编译的可执行文件只能在少数几个GPU上面执行。选项`-code=sm_ZW`指定了GPU的真实架构为`Z.W`。对应的可执行文件只能在主版本号为`z`次版，次版本号大于等于`W`的GPU中运行。例如：

```c++
-arch=compute_35  -code=sm_35
```

那么，可执行文件只能在计算能力为3.5和3.7的GPU中执行，而编译选项

```c++
-arch=compute_35  -code=sm_60
```

编译的可执行文件只能在所有的帕斯卡架构的GPU中执行。



**指定多组计算能力**

​	如果希望编译出来的可执行文件能够在更多的GPU中执行，可以同时指定多组计算能力，每一组用如下形式的编译选项：

```c++
-gencode arch=compute_XY,code=sm_ZW
```

例如：

```c++
-gencode arch=compute_35,code=sm_35
-gencode arch=compute_50,code=sm_50
-gencode arch=compute_60,code=sm_60
-gencode arch=compute_70,code=sm_70
```

编译出来的可执行文件将包含4个二进制版本，分别对应了开普勒架构(不包含3.0,3.2的计算能力)、麦克斯韦架构、帕斯卡架构、伏特架构。这样的可执行文件称为胖二进制文件。在不同的GPU架构中运行时会自动选择对应的二进制版本。上面我们指定了7.0的计算能力，即至少是CUDA9.0，如果编译选项指定了不被支持的计算能力，编译器会报错。当然指定的计算能力越多，编译越慢，文件越大。

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 2.5.3 即时编译

​	即时编译是指，可在运行可执行文件时保留其中的PTX代码，每次临时编译一个cubin目标代码。要在可执行文件中保留一个这样的PTX代码，就必须使用如下方式指定所保留PTX代码的虚拟架构：

```c++
-gencode arch=compute_XY,code=compute_XY
```

这里两个都是虚拟架构，所以计算能力必须一致。例如，假设我们最高支持到CUDA8.0(**不支持伏特架构**)，但我们希望编译的文件尽可能适用于更多GPU，则可以采用如下的编译选项：

```c++
-gencode arch=compute_35,code=sm_35
-gencode arch=compute_50,code=sm_50
-gencode arch=compute_60,code=sm_60
-gencode arch=compute_60,code=compute_60
```

其中，前三行的选项分别对应3个真实架构的cubin目标代码，第四行对应保留的PTX代码。这样编译出来的可执行文件可以直接在伏特架构的GPU中运行，只不过不能充分利用其性能。在伏特架构的GPU中运行时，会根据虚拟架构为6.0的PTX代码即时编译一个适用于当前GPU的目标代码。



## 2.5.4 计算能力选项简化书写

一个简化的编译选项可以是：

```c++
-arch=sm_XY
```

它等价于：

```c++
-gencode arch=compute_XY,code=sm_XY
-gencode arch=compute_XY,code=compute_XY
```



## 2.5.5 默认计算能力

1. CUDA 6.0 及更早：默认计算能力为 1.0；
2. CUDA 6.5～CUDA 8.0：默认计算能力 2.0；
3. CUDA 9.0～CUDA 10.2：默认计算能力 3.0；

计算能力实际是GPU的属性，和架构有关，查询自己的显卡计算能力可以查看：https://developer.nvidia.com/zh-cn/cuda-gpus 。

| 虚拟计算架构                           | 支持的显卡系列                     |
| -------------------------------------- | ---------------------------------- |
| compute_35， 和 compute_37             | 开普勒支持统一内存编程动态并行支持 |
| compute_50, compute_52， 和 compute_53 | + 麦克斯韦支持                     |
| compute_60, compute_61， 和 compute_62 | + 帕斯卡支持                       |
| compute_70和compute_72                 | + 伏特支持                         |
| compute_75                             | + 图灵支持                         |
| compute_80, compute_86和 compute_87    | + NVIDIA Ampere GPU 架构支持       |

更多编译细节参考：https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/





---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

<p name="bottom" id="bottom">
    <b>完！</b>
</p>