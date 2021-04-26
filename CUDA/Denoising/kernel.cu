#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ void sortArray(float array[], int n) {
    float tmp;
    int key, j;
    for (int i = 1; i < n * n; i++)
    {
        key = array[i];
        j = i - 1;

        while (j >= 0 && array[j] > key)
        {
            array[j + 1] = array[j];
            j = j - 1;
        }
        array[j + 1] = key;
    }
}

__global__ void denoising(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int kernelSize, int percent)
{

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

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

        for (int u = dst_x - k; u <= dst_x + k; u++)
        {
            for (int v = dst_y - k; v <= dst_y + k; v++)
            {
                rNeighbors[counter] = (float)src(v, u).z;
                gNeighbors[counter] = (float)src(v, u).y;
                bNeighbors[counter] = (float)src(v, u).x;
                counter++;
            }
        }

        sortArray(rNeighbors, k);
        sortArray(gNeighbors, k);
        sortArray(bNeighbors, k);

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

void denoisingCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, int kernelSize, int percent)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    denoising << <grid, block >> > (src, dst, dst.rows, dst.cols, kernelSize, percent);

}