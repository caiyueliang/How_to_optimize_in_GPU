#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <iostream>

#define THREAD_PER_BLOCK 256

// ===================================================================================================
// reduce0版本：baseline
__global__ void reduce0(float*vec_in, float*vec_out) {
    //__shared__ float* shared_vec = THREAD_PER_BLOCK * sizeof(float);
    //__shared__ float shared_vec[THREAD_PER_BLOCK];            // 由__shared__修饰的变量。block内的线程共享。
    extern  __shared__ float shared_vec[];                      // 由__shared__修饰的变量。block内的线程共享。长度由外部传入。

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;

    // if (gid % blockDim.x == 0) {
    //     printf("threadIdx.x:%d = id:%d ; blockDim.x:%d * blockIdx.x:%d + threadIdx.x:%d = gid:%d\n", 
    //             threadIdx.x, id, blockDim.x, blockIdx.x, threadIdx.x, gid);
    // }
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

    // ------------------------------------------------------------------------------------------
    // if (gid % blockDim.x == 0) {           // 这种写法和下面写法一样，目前没报错
    //     vec_out[int(gid/blockDim.x)] = shared_vec[id];
    //     // if (vec_out[int(gid/blockDim.x)] != 256.0) {
    //     //     for (int n = 0; n < blockDim.x; n ++) {
    //     //         printf("[last] id: %d ; gid: %d; shared_vec[%d]: %lf\n", id, gid, n, shared_vec[n]);
    //     //     }
    //     // }
    // }
    if (tid == 0) {
        vec_out[blockIdx.x] = shared_vec[0];
    }
}

// ===================================================================================================
// reduce_v1版本:
// v0版本问题: reduce0存在的最大问题就是warp divergent的问题。对于一个block而言，它所有的thread都是执行同一条指令。如果存在if-else这样的分支情况的话，thread会执行所有的分支。
//              只是不满足条件的分支，所产生的结果不会记录下来。可以在上图中看到，在每一轮迭代中都会产生两个分支，分别是红色和橙色的分支。这严重影响了代码执行的效率。
// v1解决方案: 解决的方式也比较明了，就是尽可能地让所有线程走到同一个分支里面。
//              对 gid 进行了计算，使判断的值变大。
//              原本：第 gid 个线程，操作第 gid 个数据。
//              修改后：第 gid 个线程，操作第index (2 * s * gid)个数据。
//              虽然代码依旧存在着if语句，但是却与reduce0代码有所不同。
//              我们继续假定block中存在256个thread，即拥有: 256 / 32 = 8个warp。
//              - 当进行第1次迭代时，0-3号（前4个） warp 的index < blockDim.x(即256)， 4-7号（后4个）warp的 index >= blockDim.x。对于每个warp而言，都只是进入到一个分支内，所以并不会存在warp divergence的情况。
//              - 当进行第2次迭代时，0、1号两个warp进入计算分支。
//              - 当进行第3次迭代时，只有0号warp进入计算分支。
//              - 当进行第4次迭代时，只有0号warp的前16个线程进入分支。   
__global__ void reduce1(float*vec_in, float*vec_out) {
    extern  __shared__ float shared_vec[];                      // 由__shared__修饰的变量。block内的线程共享。长度由外部传入。

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;

    shared_vec[tid] = vec_in[gid];
    __syncthreads();

    // ------------------------------------------------------------------------------------------
    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;                    // 2 * [1,2,4,...] * [0~255]
        if (index < blockDim.x) {                   // blockDim.x = 256
            shared_vec[index] += shared_vec[index + s];
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------------------------------
    if (tid == 0) {
        vec_out[blockIdx.x] = shared_vec[0];
    }
}

// ===================================================================================================
// reduce_v2版本:
// v1版本问题: reduce_v1的最大问题是bank冲突。
//              在第一次迭代中，0号线程需要去取 load shared memory的0号地址以及1号地址的数，然后写回到0号地址。而此时，这个warp中的16号线程，需要去取load shared memory中的32号地址和33号地址。可以发现，0号地址跟32号地址产生了2路的bank冲突。
//              在第2次迭代中，0号线程需要去load shared memory中的0号地址和2号地址。这个warp中的8号线程需要load shared memory中的32号地址以及34号地址，16号线程需要load shared memory中的64号地址和68号地址，24号线程需要load shared memory中的96号地址和100号地址。又因为0、32、64、96号地址对应着同一个bank，所以此时产生了4路的bank冲突。
//              现在，可以继续算下去，8路bank冲突，16路bank冲突。由于bank冲突，所以reduce1性能受限。下图说明了在load第一个数据时所产生的bank冲突。
// v2解决方案: 解决bank冲突的方式就是把for循环逆着来。原来stride从0到256，现在stride从128到0。
//              把目光继续看到这个for循环中，并且只分析0号warp。
//              - 第1轮迭代：0号线程需要load shared memory的0号元素以及128号元素。1号线程需要load shared memory中的1号元素和129号元素。这一轮迭代中，在读取第一个数时，warp中的32个线程刚好load 一行shared memory数据。
//              - 第2轮迭代：0号线程load 0号元素和64号元素，1号线程load 1号元素和65号元素。咦，也是这样，每次load shared memory的一行。
//              - 第3轮迭代，0号线程load 0号元素和32号元素，接下来不写了，总之，一个warp load shared memory的一行。没有bank冲突。
//              - 到了4轮迭代，0号线程load 0号元素和16号元素。那16号线程呢，16号线程啥也不干，因为s=16，16-31号线程啥也不干，跳过去了。
__global__ void reduce2(float*vec_in, float*vec_out) {
    extern  __shared__ float shared_vec[];                      // 由__shared__修饰的变量。block内的线程共享。长度由外部传入。

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;

    shared_vec[tid] = vec_in[gid];
    __syncthreads();

    // ------------------------------------------------------------------------------------------
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) { 
        if (tid < s) {
            shared_vec[tid] += shared_vec[tid + s];
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------------------------------
    if (tid == 0) {
        vec_out[blockIdx.x] = shared_vec[0];
    }
}

// ===================================================================================================

// ===================================================================================================


// ===================================================================================================
// reduce_v6版本:
template <typename T>
T warpReduceSum(T val) {
    // T local_var = xxx;
    for(int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(uint(-1), val, mask, 32);
    }
    return val;
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* cache, unsigned int tid){
    if (blockSize >= 64)cache[tid]+=cache[tid+32];
    if (blockSize >= 32)cache[tid]+=cache[tid+16];
    if (blockSize >= 16)cache[tid]+=cache[tid+8];
    if (blockSize >= 8)cache[tid]+=cache[tid+4];
    if (blockSize >= 4)cache[tid]+=cache[tid+2];
    if (blockSize >= 2)cache[tid]+=cache[tid+1];
}

__global__ void reduce6(float*vec_in, float*vec_out) {
    extern  __shared__ float shared_vec[];                      // 由__shared__修饰的变量。block内的线程共享。长度由外部传入。

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;

    shared_vec[tid] = vec_in[gid];
    __syncthreads();

    // ------------------------------------------------------------------------------------------
    if (blockDim.x >= 512) {
        if (tid < 256) {
            shared_vec[tid] += shared_vec[tid + 256];
        }
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) {
            shared_vec[tid] += shared_vec[tid + 128];
        }
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) {
            shared_vec[tid] += shared_vec[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) {
        // shared_vec = warpReduceSum(shared_vec);
        warpReduce<THREAD_PER_BLOCK>(shared_vec, tid);
    }
    // ------------------------------------------------------------------------------------------
    if (tid == 0) {
        vec_out[blockIdx.x] = shared_vec[0];
    }
}

// ===================================================================================================
bool check(float *out, float *res, int n) {
    for (int i=0; i<n; i++) {
        if (out[i] != res[i])
            return false;
    }
    return true;
}

int main(int argc, char **argv) {
    // 检查是否有足够的命令行参数
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << "<block_num> <block_size> <version>" << std::endl;
        return 1;
    }

    // 获取并打印传入的参数
    int block_num = std::atoi(argv[1]);
    int block_size = std::atoi(argv[2]);
    int version = std::atoi(argv[3]);
    std::cout << "Parameter  [block_num]: " << block_num << std::endl;
    std::cout << "Parameter [block_size]: " << block_size << std::endl;
    std::cout << "Parameter    [version]: " << version << std::endl;

    //const int N = 32 * 1024 * 1024;
    const int N = block_num * block_size;
    int nBytes = N * sizeof(float);
    /printf("N: %d, nBytes: %d \n", N, nBytes);

    float *a = (float *)malloc(nBytes);
    float *d_a;
    cudaMalloc((void **)&d_a, nBytes);
    // float *a;
    // cudaMallocManaged((void**)&a, nBytes);

    // int block_num = N / THREAD_PER_BLOCK;   // 128 * 1024
    float *out=(float *)malloc((block_num)*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (block_num)*sizeof(float));
    float *res=(float *)malloc(block_num * sizeof(float));
    // float *out;
    // cudaMallocManaged((void**)&out, block_num * sizeof(float));

    for(int i=0; i<N; i++) {
        a[i]=1;
    }

    for(int i = 0; i < block_num; i ++) {
        float cur = 0;
        for(int j = 0; j < block_size; j++) {
            cur += a[i * block_size + j];
        }
        res[i] = cur;
    }
    printf("res[0]: %f, res[1]: %f \n", res[0], res[1]);

    cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);            // {131072, 1, 1}
    dim3 Block(block_size, 1);          // {256, 1, 1}
    printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
        Grid.x, Grid.y, Grid.z, Block.x, Block.y, Block.z);

    if (version == 0) {   
        reduce0<<<Grid, Block, block_size*sizeof(float)>>>(d_a, d_out);
    } else if (version == 1) {   
        reduce1<<<Grid, Block, block_size*sizeof(float)>>>(d_a, d_out);
    } else if (version == 2) {   
        reduce2<<<Grid, Block, block_size*sizeof(float)>>>(d_a, d_out);
    } else if (version == 6) {   
        reduce6<<<Grid, Block, block_size*sizeof(float)>>>(d_a, d_out);
    } else {
        reduce0<<<Grid, Block, block_size*sizeof(float)>>>(d_a, d_out);
        reduce1<<<Grid, Block, block_size*sizeof(float)>>>(d_a, d_out);
        reduce2<<<Grid, Block, block_size*sizeof(float)>>>(d_a, d_out);
        reduce6<<<Grid, Block, block_size*sizeof(float)>>>(d_a, d_out);
    }
    cudaMemcpy(out, d_out, block_num*sizeof(float), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    if(check(out, res, block_num)) {
        printf("the ans is right\n");
    }
    else {
        printf("the ans is wrong\n");
        for(int i=0; i<block_num; i++) {
            printf("%lf ", out[i]);
        }
        printf("\n");
    }

    cudaFree(a);
    cudaFree(out);
}
