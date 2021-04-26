#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__global__ void laplacianFilter(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols - 1 && dst_y < rows - 1 &&
        dst_x > 1 && dst_y > 1)
    {

        // Sum of pixel values 
        float sum[3] = { 0.0, 0.0, 0.0 };
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                sum[0] += (float)(src(dst_y + i, dst_x + j).x * -1.0);
                sum[1] += (float)(src(dst_y + i, dst_x + j).y * -1.0);
                sum[2] += (float)(src(dst_y + i, dst_x + j).z * -1.0);
            }
        }
        dst(dst_y, dst_x).x = sum[0] + (src(dst_y, dst_x).x * 9.0);
        dst(dst_y, dst_x).y = sum[1] + (src(dst_y, dst_x).y * 9.0);
        dst(dst_y, dst_x).z = sum[2] + (src(dst_y, dst_x).z * 9.0);

    }
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void laplacianCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    laplacianFilter << <grid, block >> > (src, dst, dst.rows, dst.cols);

}