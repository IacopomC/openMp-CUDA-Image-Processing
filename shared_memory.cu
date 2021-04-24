#include<stdio.h>
#include<stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#define BLOCK_SIZE 32
#define RADIUS 20

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

__global__ void gaussianSep(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, cv::cuda::PtrStep<uchar3> d_tmp_img,
    int rows, int cols, cv::cuda::PtrStep<float> d_kernelGaussConv, const int kernelSize, int sigma)
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

    if (dst_x < cols && dst_y < rows)
    {

        // Apply the kernel
        float tmp[3] = { 0,0,0 };

        for (int i = -RADIUS; i <= RADIUS; i++)
        {
            tmp[0] += temp[lindex_Y][lindex_X + i].x * d_kernelGaussConv(0, i + RADIUS);
            tmp[1] += temp[lindex_Y][lindex_X + i].y * d_kernelGaussConv(0, i + RADIUS);
            tmp[2] += temp[lindex_Y][lindex_X + i].z * d_kernelGaussConv(0, i + RADIUS);
        }

        d_tmp_img(dst_y, dst_x).x = (unsigned char)(tmp[0]);
        d_tmp_img(dst_y, dst_x).y = (unsigned char)(tmp[1]);
        d_tmp_img(dst_y, dst_x).z = (unsigned char)(tmp[2]);

    }
    
    __syncthreads();
    
    __shared__ uchar3 temp2[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE];

    // local indices
    int lindex_X2 = threadIdx.x;
    int lindex_Y2 = threadIdx.y + RADIUS;

    int dst_x2 = blockDim.x * blockIdx.x + lindex_X2;
    int dst_y2 = blockDim.y * blockIdx.y + lindex_Y2;

    if (dst_x2 < cols && dst_y2 < rows)
    {

        // Read input elements into shared memory
        temp2[lindex_Y2][lindex_X2] = d_tmp_img(dst_y2, dst_x2);
        if (threadIdx.y < RADIUS) {
            temp2[lindex_Y2 - RADIUS][lindex_X2] = d_tmp_img(dst_y2 - RADIUS, dst_x2);
            if (dst_y2 + BLOCK_SIZE < rows)
                temp2[lindex_Y2 + BLOCK_SIZE][lindex_X2] = d_tmp_img(dst_y2 + BLOCK_SIZE, dst_x2);
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
            tmp2[0] += temp2[lindex_Y2 + i][lindex_X2].x * d_kernelGaussConv(0, i + RADIUS);
            tmp2[1] += temp2[lindex_Y2 + i][lindex_X2].y * d_kernelGaussConv(0, i + RADIUS);
            tmp2[2] += temp2[lindex_Y2 + i][lindex_X2].z * d_kernelGaussConv(0, i + RADIUS);
        }


        dst(dst_y2, dst_x2).x = (unsigned char)(tmp2[0]);
        dst(dst_y2, dst_x2).y = (unsigned char)(tmp2[1]);
        dst(dst_y2, dst_x2).z = (unsigned char)(tmp2[2]);
    }

}

__global__ void gaussianConv(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, cv::cuda::PtrStep<float> d_kernelGaussConv, int kernelSize, int sigma)
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

    if (dst_x < cols - RADIUS && dst_y < rows -RADIUS && dst_x > RADIUS && dst_y > RADIUS)
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

void denoisingCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, int kernelSize, int percent)
{

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    denoising << <grid, block >> > (src, dst, dst.rows, dst.cols, kernelSize, percent);

}

void gaussianSepCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& tmp_img, int dimX, int dimY, cv::cuda::GpuMat& d_kernelGaussConv, int kernelSize, int sigma)
{

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    gaussianSep << <grid, block >> > (src, dst, tmp_img, dst.rows, dst.cols, d_kernelGaussConv, kernelSize, sigma);

}

void gaussianConvCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, cv::cuda::GpuMat& d_kernelGaussConv, int kernelSize, int sigma)
{

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    gaussianConv << <grid, block >> > (src, dst, dst.rows, dst.cols, d_kernelGaussConv, kernelSize, sigma);

}

void laplacianCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY)
{

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    laplacianFilter << <grid, block >> > (src, dst, dst.rows, dst.cols);

}
