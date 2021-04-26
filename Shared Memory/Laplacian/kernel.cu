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
#define RADIUS 1

__global__ void laplacianFilter(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
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
        float tmp[3] = { 0, 0, 0 };

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                uchar3 cur = temp[lindex_Y + i][lindex_X + j];
                tmp[0] -= (float)cur.x;
                tmp[1] -= (float)cur.y;
                tmp[2] -= (float)cur.z;
            }
        }

        dst(dst_y, dst_x).x = (unsigned char)(tmp[0] + ((float)src(dst_y, dst_x).x * 9.0));
        dst(dst_y, dst_x).y = (unsigned char)(tmp[1] + ((float)src(dst_y, dst_x).y * 9.0));
        dst(dst_y, dst_x).z = (unsigned char)(tmp[2] + ((float)src(dst_y, dst_x).z * 9.0));

    }
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void laplacianCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    laplacianFilter << <grid, block >> > (src, dst, dst.rows, dst.cols);

}
