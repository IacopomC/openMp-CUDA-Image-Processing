#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ float linearInt(float s, float e, float t) {
    return s + (e - s) * t;
}

__device__ uchar3 bInterp(uchar3  c00, uchar3  c10, uchar3  c01, uchar3  c11, float tx, float ty) {

    float bchannel = linearInt(linearInt((float)c00.x, (float)c10.x, tx), linearInt((float)c01.x, (float)c11.x, tx), ty);
    float gchannel = linearInt(linearInt((float)c00.y, (float)c10.y, tx), linearInt((float)c01.y, (float)c11.y, tx), ty);
    float rchannel = linearInt(linearInt((float)c00.z, (float)c10.z, tx), linearInt((float)c01.z, (float)c11.z, tx), ty);

    uchar3 interp;
    interp.x = (unsigned char)bchannel;
    interp.y = (unsigned char)gchannel;
    interp.z = (unsigned char)rchannel;

    return interp;
}

__global__ void scaling(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int oldRows, int oldCols, float scaleFactor)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    float x_ratio = float(oldCols - 1) / cols;
    float y_ratio = float(oldRows - 1) / rows;

    if (dst_x < cols && dst_y < rows)
    {
        float gx, gy, tx, ty;

        // Break into an integral and a fractional part
        gx, tx = modf(dst_x * x_ratio, &gx);
        gy, ty = modf(dst_y * y_ratio, &gy);
        uchar3 c00 = src(gy, gx);
        uchar3 c10 = src(gy, gx + 1);
        uchar3 c01 = src(gy + 1, gx);
        uchar3 c11 = src(gy + 1, gx + 1);
        dst(dst_y, dst_x) = bInterp(c00, c10, c01, c11, tx, ty);

    }
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void scalingCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, float scaleFactor)
{
    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    scaling << <grid, block >> > (src, dst, dst.rows, dst.cols, src.rows, src.cols, scaleFactor);
}