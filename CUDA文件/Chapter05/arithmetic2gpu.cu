#include "error.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const real a = 1.23;
const real x0=10.0;
void __global__ arithmetic(real *d_x, const real x0, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *h_x = (real*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
    }

    real *d_x;
    CHECK(cudaMalloc((void **)&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;


    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    arithmetic<<<grid_size, block_size>>>(d_x, x0, N);

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time = %g ms.\n", elapsed_time);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ arithmetic(real *d_x, const real x0, const int N)
{
    const int n = blockDim.x  * blockIdx.x + threadIdx.x;
    if (n>=N) return;
    real x_temp = d_x[n];
    while (sqrt(x_temp) < x0)
    {
        ++x_temp;
    }
    d_x[n] = x_temp;
}