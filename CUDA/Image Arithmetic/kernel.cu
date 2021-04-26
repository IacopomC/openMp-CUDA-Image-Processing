#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__global__ void imageCombination(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, const cv::cuda::PtrStep<uchar3> src2, int rows, int cols, int imageComb, float offSet, float scaleFactor)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows)
    {
        if (imageComb == 0) {
            dst(dst_y, dst_x).x = src(dst_y, dst_x).x + src2(dst_y, dst_x).x;
            dst(dst_y, dst_x).y = src(dst_y, dst_x).y + src2(dst_y, dst_x).y;
            dst(dst_y, dst_x).z = src(dst_y, dst_x).z + src2(dst_y, dst_x).z;
        }
        else if (imageComb == 1) {
            dst(dst_y, dst_x).x = src(dst_y, dst_x).x - src2(dst_y, dst_x).x;
            dst(dst_y, dst_x).y = src(dst_y, dst_x).y - src2(dst_y, dst_x).y;
            dst(dst_y, dst_x).z = src(dst_y, dst_x).z - src2(dst_y, dst_x).z;
        }
        else if (imageComb == 2) {
            dst(dst_y, dst_x).x = src(dst_y, dst_x).x * src2(dst_y, dst_x).x;
            dst(dst_y, dst_x).y = src(dst_y, dst_x).y * src2(dst_y, dst_x).y;
            dst(dst_y, dst_x).z = src(dst_y, dst_x).z * src2(dst_y, dst_x).z;
        }
        else if (imageComb == 3) {
            dst(dst_y, dst_x).x = src2(dst_y, dst_x).x == 0 ? src(dst_y, dst_x).x : src(dst_y, dst_x).x / src2(dst_y, dst_x).x;
            dst(dst_y, dst_x).y = src2(dst_y, dst_x).y == 0 ? src(dst_y, dst_x).y : src(dst_y, dst_x).y / src2(dst_y, dst_x).y;
            dst(dst_y, dst_x).z = src2(dst_y, dst_x).z == 0 ? src(dst_y, dst_x).z : src(dst_y, dst_x).z / src2(dst_y, dst_x).z;
        }

        dst(dst_y, dst_x).x *= scaleFactor;
        dst(dst_y, dst_x).y *= scaleFactor;
        dst(dst_y, dst_x).z *= scaleFactor;

        dst(dst_y, dst_x).x += offSet;
        dst(dst_y, dst_x).y += offSet;
        dst(dst_y, dst_x).z += offSet;
    }
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void imageCombCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& src2, int dimX, int dimY, int imageComb, float offSet, float scaleFactor)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    imageCombination << <grid, block >> > (src, dst, src2, dst.rows, dst.cols, imageComb, offSet, scaleFactor);

}