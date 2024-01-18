#include<stdio.h>

__global__ void hello_from_gpu()
{
    printf("hello word from the gpu! [%d, %d], [%d, %d]\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main()
{
    printf("======================================================\n");
    hello_from_gpu<<<1,1>>>();
    cudaDeviceSynchronize();

    printf("======================================================\n");
    hello_from_gpu<<<3,2>>>();
    cudaDeviceSynchronize();

    printf("======================================================\n");
    // Kernel 线程配置
    dim3 numBlocks(2, 1);
    dim3 threadsPerBlock(4, 3);

    hello_from_gpu<<<numBlocks,threadsPerBlock>>>();
    cudaDeviceSynchronize();

    printf("======================================================\n");
    printf("hello world\n");
    return 0;
}