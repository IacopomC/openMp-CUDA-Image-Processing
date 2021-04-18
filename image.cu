#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    uchar3 full = make_uchar3(255, 255, 255);


    if (dst_x < cols && dst_y < rows)
    {
        uchar3 val = src(dst_y, dst_x);

        dst(dst_y, dst_x).x = full.x - val.x;
        dst(dst_y, dst_x).y = full.y - val.y;
        dst(dst_y, dst_x).z = full.z - val.z;
    }
}

__global__ void gaussian(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    uchar3 full = make_uchar3(255, 255, 255);


    if (dst_x < cols && dst_y < rows)
    {
        uchar3 val = src(dst_y, dst_x);

        dst(dst_y, dst_x).x = full.x - val.x;
        dst(dst_y, dst_x).y = full.y - val.y;
        dst(dst_y, dst_x).z = full.z - val.z;
    }
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY)
{
    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process << <grid, block >> > (src, dst, dst.rows, dst.cols);

}

void gaussianCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    gaussian << <grid, block >> > (src, dst, dst.rows, dst.cols);

}

