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

__global__ void denoising(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int kernelSize, int percent)
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

    const int k = (kernelSize - 1.0) / 2.0;

    if (dst_x < cols - k && dst_y < rows - k &&
        dst_x > k && dst_y > k)
    {
        float rNeighbors[144];
        float gNeighbors[144];
        float bNeighbors[144];

        float rValue = 0.0;
        float gValue = 0.0;
        float bValue = 0.0;

        int counter = 0;

        for (int i = -RADIUS; i <= RADIUS; i++)
        {
            for (int j = -RADIUS; j <= RADIUS; j++)
            {

                uchar3 cur = temp[lindex_Y + i][lindex_X + j];

                rNeighbors[counter] = (float)cur.z;
                gNeighbors[counter] = (float)cur.y;
                bNeighbors[counter] = (float)cur.x;
                counter++;
            }
        }

        int key, j;
        for (int i = 1; i < k * k; i++)
        {
            key = rNeighbors[i];
            j = i - 1;

            while (j >= 0 && rNeighbors[j] > key)
            {
                rNeighbors[j + 1] = rNeighbors[j];
                j = j - 1;
            }
            rNeighbors[j + 1] = key;
        }

        for (int i = 1; i < k * k; i++)
        {
            key = gNeighbors[i];
            j = i - 1;

            while (j >= 0 && gNeighbors[j] > key)
            {
                gNeighbors[j + 1] = gNeighbors[j];
                j = j - 1;
            }
            gNeighbors[j + 1] = key;
        }

        for (int i = 1; i < k * k; i++)
        {
            key = bNeighbors[i];
            j = i - 1;

            while (j >= 0 && bNeighbors[j] > key)
            {
                bNeighbors[j + 1] = bNeighbors[j];
                j = j - 1;
            }
            bNeighbors[j + 1] = key;
        }

        int medianIndx = 0;
        if (kernelSize % 2 == 0) {
            medianIndx = kernelSize * kernelSize / 2;
        }
        else {
            medianIndx = (kernelSize * kernelSize - 1) / 2;
        }

        int numEl = (kernelSize * kernelSize * int(percent) / 100) / 2;

        for (int w = (medianIndx - numEl); w <= (medianIndx + numEl); w++) {
            rValue += rNeighbors[w];
            gValue += gNeighbors[w];
            bValue += bNeighbors[w];
        }

        if (numEl >= 1) {
            rValue /= float(2 * numEl + 1);
            gValue /= float(2 * numEl + 1);
            bValue /= float(2 * numEl + 1);
        }

        dst(dst_y, dst_x).z = (unsigned char)(rValue);
        dst(dst_y, dst_x).y = (unsigned char)(gValue);
        dst(dst_y, dst_x).x = (unsigned char)(bValue);

    }
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void denoisingCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernelSize, int percent)
{

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    denoising << <grid, block >> > (src, dst, dst.rows, dst.cols, kernelSize, percent);

}
