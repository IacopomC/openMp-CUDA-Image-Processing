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

__global__ void gaussianSepHor(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst,
    int rows, int cols, cv::cuda::PtrStep<float> d_kernelGaussConv)
{

    __shared__ uchar3 temp[BLOCK_SIZE][BLOCK_SIZE + 2 * RADIUS];

    // local indices
    int lindex_X = threadIdx.x + RADIUS;
    int lindex_Y = threadIdx.y;

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
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    if (dst_x < cols - RADIUS && dst_y < rows && dst_x > RADIUS && dst_y > 0)
    {

        // Apply the kernel
        float tmp[3] = { 0,0,0 };

        for (int i = -RADIUS; i <= RADIUS; i++)
        {
            tmp[0] += (float)(temp[lindex_Y][lindex_X + i].x) * d_kernelGaussConv(0, i + RADIUS);
            tmp[1] += (float)(temp[lindex_Y][lindex_X + i].y) * d_kernelGaussConv(0, i + RADIUS);
            tmp[2] += (float)(temp[lindex_Y][lindex_X + i].z) * d_kernelGaussConv(0, i + RADIUS);
        }

        dst(dst_y, dst_x).x = (unsigned char)(tmp[0]);
        dst(dst_y, dst_x).y = (unsigned char)(tmp[1]);
        dst(dst_y, dst_x).z = (unsigned char)(tmp[2]);

    }

}

__global__ void gaussianSepVer(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst,
    int rows, int cols, cv::cuda::PtrStep<float> d_kernelGaussConv)
{

    __shared__ uchar3 temp2[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE];

    // local indices
    int lindex_X2 = threadIdx.x;
    int lindex_Y2 = threadIdx.y + RADIUS;

    int dst_x2 = blockDim.x * blockIdx.x + lindex_X2;
    int dst_y2 = blockDim.y * blockIdx.y + lindex_Y2;

    if (dst_x2 < cols && dst_y2 < rows - RADIUS && dst_x2 > 0 && dst_y2 > RADIUS)
    {

        // Read input elements into shared memory
        temp2[lindex_Y2][lindex_X2] = src(dst_y2, dst_x2);
        if (threadIdx.y < RADIUS) {
            temp2[lindex_Y2 - RADIUS][lindex_X2] = src(dst_y2 - RADIUS, dst_x2);
            if (dst_y2 + BLOCK_SIZE < rows)
                temp2[lindex_Y2 + BLOCK_SIZE][lindex_X2] = src(dst_y2 + BLOCK_SIZE, dst_x2);
        }
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    if (dst_x2 < cols && dst_y2 < rows)
    {

        // Apply the kernel
        float tmp2[3] = { 0,0,0 };

        for (int i = -RADIUS; i <= RADIUS; i++)
        {
            tmp2[0] += (float)(temp2[lindex_Y2 + i][lindex_X2].x) * d_kernelGaussConv(0, i + RADIUS);
            tmp2[1] += (float)(temp2[lindex_Y2 + i][lindex_X2].y) * d_kernelGaussConv(0, i + RADIUS);
            tmp2[2] += (float)(temp2[lindex_Y2 + i][lindex_X2].z) * d_kernelGaussConv(0, i + RADIUS);
        }


        dst(dst_y2, dst_x2).x = (unsigned char)(tmp2[0]);
        dst(dst_y2, dst_x2).y = (unsigned char)(tmp2[1]);
        dst(dst_y2, dst_x2).z = (unsigned char)(tmp2[2]);
    }

}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void gaussianSepCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& d_kernelGaussConv, int hor_pass)
{

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    if (hor_pass == 1)
    {
        gaussianSepHor << <grid, block >> > (src, dst, dst.rows, dst.cols, d_kernelGaussConv);
    }
    else
    {
        gaussianSepVer << <grid, block >> > (src, dst, dst.rows, dst.cols, d_kernelGaussConv);
    }

}
