#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

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

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void colorTransfCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, float angle)
{

    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    hueShift << <grid, block >> > (src, dst, dst.rows, dst.cols, angle);

}
