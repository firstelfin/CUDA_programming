#include "error.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

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
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    for (int n=0; n<10; ++n)
    {
        add(x, y, z, N);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time = %g ms .\n", elapsed_time / 10);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
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