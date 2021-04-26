#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__global__ void gaussianConv(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols,
    cv::cuda::PtrStep<float> d_kernelGaussConv, int kernelSize, int sigma)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    const float k = (kernelSize - 1.0) / 2.0;

    if (dst_x < cols - k && dst_y < rows - k &&
        dst_x > k && dst_y > k)
    {
        float tmp[3] = { 0.0, 0.0, 0.0, };

        for (int u = dst_x - k; u <= dst_x + k; u++)
        {
            for (int v = dst_y - k; v <= dst_y + k; v++)
            {
                tmp[0] += (float)src(v, u).x * d_kernelGaussConv(v - dst_y + k, u - dst_x + k);

                tmp[1] += (float)src(v, u).y * d_kernelGaussConv(v - dst_y + k, u - dst_x + k);

                tmp[2] += (float)src(v, u).z * d_kernelGaussConv(v - dst_y + k, u - dst_x + k);
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

void gaussianConvCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, cv::cuda::GpuMat& d_kernelGaussConv, int kernelSize, int sigma)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    gaussianConv << <grid, block >> > (src, dst, dst.rows, dst.cols, d_kernelGaussConv, kernelSize, sigma);

}