#include <iostream>
#include <chrono>

// 两个向量加法kernel，grid和block均为一维
__global__ void add(float* x, float * y, float* z, int n)  // x,y,z 是指针，所以可以返回内容
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    if (index % (1 << 20) == 0) {
        printf("threadIdx.x:%d + blockIdx.x:%d * blockDim.x:%d = index:%d\n", 
            threadIdx.x, blockIdx.x, blockDim.x, index);
        printf("blockDim.x:%d * gridDim.x:%d = stride:%d\n", 
            blockDim.x, gridDim.x, stride);
    }
    // for (int i = index; i < n; i += stride) {
    //     z[i] = x[i] + y[i];
    // }
    z[index] = x[index] + y[index];
}

int main(int argc, char **argv)
{
    // 检查是否有足够的命令行参数
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <block_size> <epoch>" << std::endl;
        return 1;
    }

    // 获取并打印传入的参数
    int block_size = std::atoi(argv[1]);
    int epoch = std::atoi(argv[2]);
    std::cout << "Parameter [block_size]: " << block_size << std::endl;
    std::cout << "Parameter [epoch]: " << epoch << std::endl;


    int N = 1 << 23;
    int nBytes = N * sizeof(float);
    std::cout << "N: " << N << "; nBytes: " << nBytes << std::endl;

    // 申请host内存
    // float *x, *y, *z;
    // x = (float*)malloc(nBytes);
    // y = (float*)malloc(nBytes);
    // z = (float*)malloc(nBytes);
    // 申请托管内存
    float *x, *y, *z;
    cudaMallocManaged((void**)&x, nBytes);
    cudaMallocManaged((void**)&y, nBytes);
    cudaMallocManaged((void**)&z, nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // 申请device内存
    // float *d_x, *d_y, *d_z;
    // cudaMalloc((void**)&d_x, nBytes);
    // cudaMalloc((void**)&d_y, nBytes);
    // cudaMalloc((void**)&d_z, nBytes);

    // // 将host数据拷贝到device
    // cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    // cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);

    // 定义kernel的执行配置
    dim3 blockSize(block_size);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // std::cout << "blockSize: " << blockSize << "; gridSize: " << gridSize << std::endl;
    printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
        gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);

    // 获取第一个时间点
    auto start = std::chrono::high_resolution_clock::now();

    // 执行kernel
    for (int i = 0; i < epoch; i++) {
        add << < gridSize, blockSize >> >(x, y, z, N);
    }

    // 获取第二个时间点
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时间差
    std::chrono::duration<double, std::milli> duration = end - start;
    // 输出结果
    std::cout << "计算耗时: " << duration.count() << " milliseconds." << std::endl;

        
    // // 将device得到的结果拷贝到host
    // cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);
    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();

    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    }
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放device内存
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    // // 释放host内存
    // free(x);
    // free(y);
    // free(z);

    return 0;
}