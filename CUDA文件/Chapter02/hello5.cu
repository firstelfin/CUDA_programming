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