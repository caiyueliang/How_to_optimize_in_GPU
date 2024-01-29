#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <stdio.h>

#define THREAD_PER_BLOCK 256

// __global__ void reduce0(float *d_in,float *d_out) {
//     __shared__ float sdata[THREAD_PER_BLOCK];

//     // each thread loads one element from global to shared mem
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//     sdata[tid] = d_in[i];
//     __syncthreads();

//     // do reduction in shared mem
//     for(unsigned int s=1; s < blockDim.x; s *= 2) {
//         if (tid % (2*s) == 0) {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();
//     }

//     // write result for this block to global mem
//     if (tid == 0) {
//         d_out[blockIdx.x] = sdata[0];
//     }
// }

__global__ void reduce0(float*vec_in, float*vec_out) {
    __shared__ float shared_vec[THREAD_PER_BLOCK];            // 由__shared__修饰的变量。block内的线程共享。

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    shared_vec[tid] = vec_in[gid];
    __syncthreads();
    // ------------------------------------------------------------------------------------------
    // reduce_baseline版本
    for (unsigned int n = 1; n < blockDim.x; n = n * 2) {
        // if (tid % n == 0) {               // 这样写，有bug
        if (tid % (n * 2) == 0) {            // 这样写，才是正确的
            shared_vec[tid] = shared_vec[tid] + shared_vec[tid + n];
        }
        __syncthreads();
    }
    if (tid == 0) {
        vec_out[blockIdx.x] = shared_vec[0];
    }
}

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}

int main(){
    const int N=32*1024*1024;
    // float *a=(float *)malloc(N*sizeof(float));
    // float *d_a;
    // cudaMalloc((void **)&d_a,N*sizeof(float));
    float *a;
    cudaMallocManaged((void**)&a, N*sizeof(float));

    int block_num=N/THREAD_PER_BLOCK;
    // float *out=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));
    // float *d_out;
    // cudaMalloc((void **)&d_out,(N/THREAD_PER_BLOCK)*sizeof(float));
    float *out;
    cudaMallocManaged((void**)&out, block_num * sizeof(float));
    float *res=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=1;
    }

    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<THREAD_PER_BLOCK;j++){
            cur+=a[i*THREAD_PER_BLOCK+j];
        }
        res[i]=cur;
    }

    cudaStream_t stream;            // 声明一个stream
    cudaStreamCreate(&stream);      // 分配stream
    // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);      // 用于使并发不受人为控制（如MPI）

    // cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid( N/THREAD_PER_BLOCK,1);
    dim3 Block( THREAD_PER_BLOCK,1);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);//此处不能用CHECK宏函数

    //需要计时的代码块
    reduce0<<<Grid, Block, 0, stream>>>(a, out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time = %g ms .\n", elapsed_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);
    
    if(check(out, res, block_num))
        printf("the ans is right\n");
    else {
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaStreamDestroy(stream);          // 取消分配的stream，在stream中的work完成后同步host端。
    cudaFree(a);
    cudaFree(out);
}
