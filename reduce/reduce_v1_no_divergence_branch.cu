#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

// bank conflict
__global__ void reduce1(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    // for(unsigned int s=1; s < blockDim.x; s *= 2) {
    //     if (tid % (2*s) == 0) {
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }
    // ------------------------------
    // 对tid进行了计算，使判断的值变大。
    // 原本：第tid个线程，操作第tid个数据。
    // 修改后：第tid个线程，操作第index (2 * s * tid)个数据。
    // 虽然代码依旧存在着if语句，但是却与reduce0代码有所不同。
    // 我们继续假定block中存在256个thread，即拥有: 256 / 32 = 8个warp。
    // - 当进行第1次迭代时，0-3号（前4个） warp 的index < blockDim.x(即256)， 4-7号（后4个）warp的 index >= blockDim.x。对于每个warp而言，都只是进入到一个分支内，所以并不会存在warp divergence的情况。
    // - 当进行第2次迭代时，0、1号两个warp进入计算分支。
    // - 当进行第3次迭代时，只有0号warp进入计算分支。
    // - 当进行第4次迭代时，只有0号warp的前16个线程进入分支。
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;                    // 2 * [1,2,4,...] * [0~255]
        if (index < blockDim.x) {                   // blockDim.x = 256
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
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
    float *a=(float *)malloc(N*sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    int block_num=N/THREAD_PER_BLOCK;
    float *out=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,(N/THREAD_PER_BLOCK)*sizeof(float));
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

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid( N/THREAD_PER_BLOCK,1);
    dim3 Block( THREAD_PER_BLOCK,1);

    reduce1<<<Grid,Block>>>(d_a,d_out);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);
}
