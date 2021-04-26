#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <cuda_runtime.h>
#include <chrono>  // for high_resolution_clock

#define BLOCK_SIZE 32
#define RADIUS 4

__global__ void gaussianConv(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, cv::cuda::PtrStep<float> d_kernelGaussConv)
{

    __shared__ uchar3 temp[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    // local indices
    int lindex_X = threadIdx.x + RADIUS;
    int lindex_Y = threadIdx.y + RADIUS;

    int dst_x = blockDim.x * blockIdx.x + lindex_X;
    int dst_y = blockDim.y * blockIdx.y + lindex_Y;

    if (dst_x < cols && dst_y < rows)
    {

        // Read input elements into shared memory
        temp[lindex_Y][lindex_X] = src(dst_y, dst_x);
        if (threadIdx.x < RADIUS) {
            temp[lindex_Y][lindex_X - RADIUS] = src(dst_y, dst_x - RADIUS);
            if (dst_x + BLOCK_SIZE < cols)
                temp[lindex_Y][lindex_X + BLOCK_SIZE] = src(dst_y, dst_x + BLOCK_SIZE);
        }

        if (threadIdx.y < RADIUS) {
            temp[lindex_Y - RADIUS][lindex_X] = src(dst_y - RADIUS, dst_x);
            if (dst_y + BLOCK_SIZE < rows)
                temp[lindex_Y + BLOCK_SIZE][lindex_X] = src(dst_y + BLOCK_SIZE, dst_x);
        }

        if (threadIdx.y < RADIUS && threadIdx.x < RADIUS) {
            temp[lindex_Y - RADIUS][lindex_X - RADIUS] = src(dst_y - RADIUS, dst_x - RADIUS);
            if (dst_y + BLOCK_SIZE < rows && dst_x + BLOCK_SIZE < cols)
                temp[lindex_Y + BLOCK_SIZE][lindex_X + BLOCK_SIZE] = src(dst_y + BLOCK_SIZE, dst_x + BLOCK_SIZE);
            if (dst_x + BLOCK_SIZE < cols)
                temp[lindex_Y - RADIUS][lindex_X + BLOCK_SIZE] = src(dst_y - RADIUS, dst_x + BLOCK_SIZE);
            if (dst_y + BLOCK_SIZE < rows)
                temp[lindex_Y + BLOCK_SIZE][lindex_X - RADIUS] = src(dst_y + BLOCK_SIZE, dst_x - RADIUS);
        }
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    if (dst_x < cols - RADIUS && dst_y < rows - RADIUS && dst_x > RADIUS && dst_y > RADIUS)
    {

        // Apply the kernel
        float tmp[3];

        for (int i = -RADIUS; i <= RADIUS; i++)
        {
            for (int j = -RADIUS; j <= RADIUS; j++)
            {
                tmp[0] += temp[lindex_Y + j][lindex_X + i].x * d_kernelGaussConv(j + RADIUS, i + RADIUS);
                tmp[1] += temp[lindex_Y + j][lindex_X + i].y * d_kernelGaussConv(j + RADIUS, i + RADIUS);
                tmp[2] += temp[lindex_Y + j][lindex_X + i].z * d_kernelGaussConv(j + RADIUS, i + RADIUS);
            }
        }

        dst(dst_y, dst_x).x = (unsigned char)(tmp[0]);
        dst(dst_y, dst_x).y = (unsigned char)(tmp[1]);
        dst(dst_y, dst_x).z = (unsigned char)(tmp[2]);
    }

}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void gaussianConvCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& d_kernelGaussConv)
{

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    gaussianConv << <grid, block >> > (src, dst, dst.rows, dst.cols, d_kernelGaussConv);

}
