#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

// // bank conflict
// __global__ void reduce1(float *d_in,float *d_out){
//     __shared__ float sdata[THREAD_PER_BLOCK];

//     // each thread loads one element from global to shared mem
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//     sdata[tid] = d_in[i];
//     __syncthreads();

//     // do reduction in shared mem
//     for(unsigned int s=1; s < blockDim.x; s *= 2) {
//         int index = 2 * s * tid;                    // 2 * [1,2,4,...] * [0~255]
//         if (index < blockDim.x) {                   // blockDim.x = 256
//             sdata[index] += sdata[index + s];
//         }
//         __syncthreads();
//     }

//     // write result for this block to global mem
//     if (tid == 0) {
//         d_out[blockIdx.x] = sdata[0];
//     }
// }

__global__ void reduce1(float*vec_in, float*vec_out) {
    __shared__ float shared_vec[THREAD_PER_BLOCK];
    // extern  __shared__ float shared_vec[];                      // 由__shared__修饰的变量。block内的线程共享。长度由外部传入。

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    shared_vec[tid] = vec_in[gid];
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;                    // 2 * [1,2,4,...] * [0~255]
        if (index < blockDim.x) {                   // blockDim.x = 256
            shared_vec[index] += shared_vec[index + s];
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
