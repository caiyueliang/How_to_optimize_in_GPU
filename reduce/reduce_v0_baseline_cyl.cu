#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce0(float* a, float* out) {
    int i = Block.x;

}


bool check(float *out, float *res, int n) {
    for (int i=0; i<n; i++) {
        if (out[i] != res[i])
            return false;
    }
    return true;
}

int main(){
    const int N = 32 * 1024 * 1024;
    int nBytes = N * sizeof(float);
    // float *a = (float *)malloc(N*sizeof(float));
    // float *d_a;
    // cudaMalloc((void **)&d_a,N*sizeof(float));
    float *a;
    cudaMallocManaged((void**)&a, nBytes);

    int block_num = N / THREAD_PER_BLOCK;   // 128 * 1024
    // float *out=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));
    // float *d_out;
    // cudaMalloc((void **)&d_out,(N/THREAD_PER_BLOCK)*sizeof(float));
    float *res=(float *)malloc(block_num * sizeof(float));
    float *out;
    cudaMallocManaged((void**)&out, block_num * sizeof(float));

    for(int i=0; i<N; i++) {
        a[i]=1;
    }

    for(int i=0; i<block_num; i++) {
        float cur=0;
        for(int j=0; j<THREAD_PER_BLOCK; j++) {
            cur+=a[i*THREAD_PER_BLOCK+j];
        }
        res[i]=cur;
    }
    printf("res[0]: %f, res[1]: %f", res[0], res[1]);

    //cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);            // {131072, 1, 1}
    dim3 Block(THREAD_PER_BLOCK, 1);    // {256, 1, 1}
    printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
        Grid.x, Grid.y, Grid.z, Block.x, Block.y, Block.z);

    reduce0<<<Grid, Block>>>(a, out);

    //cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if(check(out, res, block_num)) {
        printf("the ans is right\n");
    }
    else {
        printf("the ans is wrong\n");
        // for(int i=0; i<block_num; i++) {
        //     printf("%lf ", out[i]);
        // }
        // printf("\n");
    }

    cudaFree(a);
    cudaFree(out);
}
