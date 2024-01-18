// 两个向量加法kernel，grid和block均为一维
__global__ void add(float* x, float * y, float* z, int n)  // x,y,z 是指针，所以可以返回内容
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}


