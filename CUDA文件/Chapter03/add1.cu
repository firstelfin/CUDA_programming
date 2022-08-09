#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// 定义要运算的基本常量
const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void __global__ add(const double *x, const double *y, double *z, const int N);
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
	cudaMalloc ((void **) &d_x, M);
	cudaMalloc ((void **) &d_y, M);
	cudaMalloc ((void **) &d_z, M);
	cudaMemcpy(d_x, h_x, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, cudaMemcpyHostToDevice);

	const int block_size = 128;
	const it grid_size = N / block_size;
	add<<<grid_size, block_size>>>(d_x, d_y, d_z);
	cudaMemcpy(h_z, d_z, cudaMemcpyDeviceToHost);
	
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
	printf("s%\n", has_error ? "Has errors" : "No errors");
}