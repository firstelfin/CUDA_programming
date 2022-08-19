#include "error.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const real a = 1.23;
const real x0=10.0;

void arithmetic(real *x, const real x0, const int N);

int main(int argc, char **argv)
{
	if (argc != 2) 
    {
        printf("usage: %s N\n", argv[0]);
        exit(1);
    }
    const int N = atoi(argv[1]);
	const int M = sizeof(real) * N;
	real  *x = (real *) malloc(M);
	for (int n = 0; n < N; ++n)
	{
		x[n] = a;
	}
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);
    arithmetic(x, x0, N);

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time = %g ms .\n", elapsed_time);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
	free(x);
	return 0;
}

void arithmetic(real *x, const real x0, const int N)
{
    for (int n = 0; n  < N; ++n)
    {
        real x_temp = x[n];
        while (sqrt(x_temp) < x0)
        {
            ++x_temp;
        }
        x[n] = x_temp;
    }
}