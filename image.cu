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
        /*
        dst(dst_y, dst_x).x = src(dst_y, dst_x).x;
        dst(dst_y, dst_x).y = src(dst_y, dst_x).y;
        dst(dst_y, dst_x).z = src(dst_y, dst_x).z;
        */
    }
}

__device__ float3 bgr2xyz(uchar3 src) {

    float scr_r = src.z / 255.0;
    float scr_g = src.y / 255.0;
    float scr_b = src.x / 255.0;

    float tmp[3];
    tmp[0] = 100.0 * ((scr_r > .04045) ? pow((scr_r + .055) / 1.055, 2.4) : scr_r / 12.92);
    tmp[1] = 100.0 * ((scr_g > .04045) ? pow((scr_g + .055) / 1.055, 2.4) : scr_g / 12.92);
    tmp[2] = 100.0 * ((scr_b > .04045) ? pow((scr_b + .055) / 1.055, 2.4) : scr_b / 12.92);

    float3 xyz;
    xyz.x = .4124 * tmp[0] + .3576 * tmp[1] + .1805 * tmp[2];
    xyz.y = .2126 * tmp[0] + .7152 * tmp[1] + .0722 * tmp[2];
    xyz.z = .0193 * tmp[0] + .1192 * tmp[1] + .9505 * tmp[2];
    
    return xyz;
}

__device__ float3 xyz2lab(float3 src, float angle) {

    float scr_z = src.z / 108.883;
    float scr_y = src.y / 100.;
    float scr_x = src.x / 95.047;

    float PI = 3.14159265358979323846;

    float v[3];
    v[0] = (scr_x > .008856) ? pow(scr_x, 1. / 3.) : (7.787 * scr_x) + (16. / 116.);
    v[1] = (scr_y > .008856) ? pow(scr_y, 1. / 3.) : (7.787 * scr_y) + (16. / 116.);
    v[2] = (scr_z > .008856) ? pow(scr_z, 1. / 3.) : (7.787 * scr_z) + (16. / 116.);

    float3 lab;
    lab.x = (116. * v[1]) - 16.;
    lab.y = 500. * (v[0] - v[1]);
    lab.z = 200. * (v[1] - v[2]);

    float C = sqrt(pow(lab.y, 2) + pow(lab.z, 2));
    float h = atan2(lab.z, lab.y);
    h += (angle * PI) / 180.0;
    lab.y = cos(h) * C;
    lab.z = sin(h) * C;

    return lab;
}

__device__ float3 bgr2lab(uchar3 c, float angle) {
    return xyz2lab(bgr2xyz(c), angle);
}

__device__ float3 lab2xyz(float3 src) {

    float fy = (src.x + 16.0) / 116.0;
    float fx = src.y / 500.0 + fy;
    float fz = fy - src.z / 200.0;

    float3 lab;
    lab.x = 95.047 * ((fx > 0.206897) ? fx * fx * fx : (fx - 16.0 / 116.0) / 7.787);
    lab.y = 100.000 * ((fy > 0.206897) ? fy * fy * fy : (fy - 16.0 / 116.0) / 7.787);
    lab.z = 108.883 * ((fz > 0.206897) ? fz * fz * fz : (fz - 16.0 / 116.0) / 7.787);

    return lab;
}

__device__ float3 xyz2bgr(float3 src) {

    src.x /= 100.0;
    src.y /= 100.0;
    src.z /= 100.0;

    
    float tmp[3];

    tmp[0] = 3.2406 * src.x - 1.5372 * src.y - 0.4986 * src.z;
    tmp[1] = -0.9689 * src.x + 1.8758 * src.y + 0.0415 * src.z;
    tmp[2] = 0.0557 * src.x - 0.2040 * src.y + 1.0570 * src.z;

    float3 bgr;
    bgr.z = 255.0 * ((tmp[0] > 0.0031308) ? ((1.055 * pow(tmp[0], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[0]));
    bgr.y = 255.0 * ((tmp[1] > 0.0031308) ? ((1.055 * pow(tmp[1], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[1]));
    bgr.x = 255.0 * ((tmp[2] > 0.0031308) ? ((1.055 * pow(tmp[2], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[2]));

    return bgr;
}

__device__ float3 lab2bgr(float3 src) {
    return xyz2bgr(lab2xyz(src));
}



__global__ void hueShift(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, float angle)
{
    
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows)
    {
        float3 bgr;
        bgr.x = lab2bgr(bgr2lab(src(dst_y, dst_x), angle)).x;
        bgr.y = lab2bgr(bgr2lab(src(dst_y, dst_x), angle)).y;
        bgr.z = lab2bgr(bgr2lab(src(dst_y, dst_x), angle)).z;

        dst(dst_y, dst_x).x = (unsigned char)(bgr.x < 0 ? 0 : (bgr.x > 255 ? 255 : bgr.x));
        dst(dst_y, dst_x).y = (unsigned char)(bgr.y < 0 ? 0 : (bgr.y > 255 ? 255 : bgr.y));
        dst(dst_y, dst_x).z = (unsigned char)(bgr.z < 0 ? 0 : (bgr.z > 255 ? 255 : bgr.z));
        
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

        int key, j;
        for (int i = 1; i < k*k; i++)
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
    int rows, int cols, cv::cuda::PtrStep<float> d_kernelGaussConv, int kernelSize, int sigma)
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
            tmp[0] += (float)src(dst_y, u).x * d_kernelGaussConv(0, u - dst_x + k);

            tmp[1] += (float)src(dst_y, u).y * d_kernelGaussConv(0, u - dst_x + k);

            tmp[2] += (float)src(dst_y, u).z * d_kernelGaussConv(0, u - dst_x + k);
        }
        d_tmp_img(dst_y, dst_x).x = tmp[0];
        d_tmp_img(dst_y, dst_x).y = tmp[1];
        d_tmp_img(dst_y, dst_x).z = tmp[2];

    }
    
    __syncthreads();

    if (dst_x < cols - k && dst_y < rows - k &&
        dst_x > k && dst_y > k)
    {
        float tmp[3] = { 0.0, 0.0, 0.0, };

        for (int v = dst_y - k; v <= dst_y + k; v++)
        {
            tmp[0] += (float)d_tmp_img(v, dst_x).x * d_kernelGaussConv(0, v - dst_y + k);

            tmp[1] += (float)d_tmp_img(v, dst_x).y * d_kernelGaussConv(0, v - dst_y + k);

            tmp[2] += (float)d_tmp_img(v, dst_x).z * d_kernelGaussConv(0, v - dst_y + k);
        }
        dst(dst_y, dst_x).x = (unsigned char)(tmp[0]);
        dst(dst_y, dst_x).y = (unsigned char)(tmp[1]);
        dst(dst_y, dst_x).z = (unsigned char)(tmp[2]);
        
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

void scalingCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, float scaleFactor)
{
    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    scaling << <grid, block >> > (src, dst, dst.rows, dst.cols, src.rows, src.cols, scaleFactor);
}

void colorTransfCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, float angle)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    hueShift << <grid, block >> > (src, dst, dst.rows, dst.cols, angle);

}

void denoisingCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, int kernelSize, int percent) 
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    denoising << <grid, block >> > (src, dst, dst.rows, dst.cols, kernelSize, percent);

}

void gaussianSepCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& tmp_img, int dimX, int dimY, cv::cuda::GpuMat& d_kernelGaussConv, int kernelSize, int sigma)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    gaussianSep << <grid, block >> > (src, dst, tmp_img, dst.rows, dst.cols, d_kernelGaussConv, kernelSize, sigma);

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

