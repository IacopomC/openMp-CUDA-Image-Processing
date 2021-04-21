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

__global__ void gaussianConv(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, cv::cuda::PtrStep<float> d_kernelGaussConv, int kernelSize, int sigma)
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
        /*
        dst(dst_y, dst_x).x = src(dst_y, dst_x).x;
        dst(dst_y, dst_x).y = src(dst_y, dst_x).y;
        dst(dst_y, dst_x).z = src(dst_y, dst_x).z;*/
    }

}

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

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY)
{
    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process << <grid, block >> > (src, dst, dst.rows, dst.cols);

}

void gaussianConvCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, cv::cuda::GpuMat& d_kernelGaussConv, int kernelSize, int sigma)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    gaussianConv << <grid, block >> > (src, dst, dst.rows, dst.cols, d_kernelGaussConv, kernelSize, sigma);

}

void laplacianCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    laplacianFilter << <grid, block >> > (src, dst, dst.rows, dst.cols);

}

void imageCombCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& src2, int dimX, int dimY, int imageComb, float offSet, float scaleFactor)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    imageCombination << <grid, block >> > (src, dst, src2, dst.rows, dst.cols, imageComb, offSet, scaleFactor);

}

