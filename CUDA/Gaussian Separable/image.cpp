#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>
#include <chrono>  // for high_resolution_clock

using namespace std;

void gaussianSepCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, cv::cuda::GpuMat& d_kernelGaussConv, int kernelSize, int sigma, int hor_pass);

cv::Mat_<float> generateGaussianKernel1D(int kernelSize, int sigma)
{
    float PI = 3.14159265358979323846;
    float constant = 1.0 / (2.0 * PI * pow(sigma, 2));
    int radius = (kernelSize - 1.0) / 2.0;
    cv::Mat_<float> h_kernel(kernelSize, 1);

    float sum = 0.0;
    for (int i = -radius; i < radius + 1; i++) {
        h_kernel[i + radius][0] = constant * (exp(-pow(i, 2) / (2 * pow(sigma, 2))));
        sum += h_kernel[i + radius][0];
    }

    for (int i = 0; i < kernelSize; ++i) {
        h_kernel[i][0] /= sum;
    }

    return h_kernel;
}

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat_<cv::Vec3b> h_img = cv::imread(argv[1]);
    cv::cuda::GpuMat d_result;
    cv::cuda::GpuMat d_img;
    cv::Mat_<cv::Vec3b> input_img;

    cv::cuda::GpuMat d_tmp_img;
    cv::cuda::GpuMat d_kernel_1D;
    cv::Mat_<float> h_kernel_1D;

    const int kernelSize = atof(argv[2]);
    int sigma = atof(argv[3]);

    int border = (int)(kernelSize - 1) / 2;

    cv::copyMakeBorder(h_img, input_img, border, border, border, border, cv::BORDER_REPLICATE);

    h_kernel_1D = generateGaussianKernel1D(kernelSize, sigma);
    d_kernel_1D.upload(h_kernel_1D);

    d_img.upload(input_img);
    d_result.upload(input_img);
    d_tmp_img.upload(input_img);

    gaussianSepCUDA(d_img, d_tmp_img, 2, 2, d_kernel_1D, kernelSize, sigma, 1);
    gaussianSepCUDA(d_tmp_img, d_result, 2, 2, d_kernel_1D, kernelSize, sigma, 0);

    // crop final image to original size
    d_result = d_result(cv::Range(border + 1, d_result.rows - border), cv::Range(border + 1, d_result.cols - border)).clone();

    /*
    for (int dim = 1; dim < 12; dim++)
    {
        auto begin = chrono::high_resolution_clock::now();
        const int iter = 100;


        for (int i = 0; i < iter; i++)
        {
            gaussianSepCUDA(d_img, d_tmp_img, dim, dim, d_kernel_1D, kernelSize, sigma, 1);
            gaussianSepCUDA(d_tmp_img, d_result, dim, dim, d_kernel_1D, kernelSize, sigma, 0);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - begin;

        cout << "Total time dim " << dim << ":" << diff.count() << endl;
        cout << "Time per iteration dim " << dim << ":" << diff.count() / iter << endl;
        cout << "Iterations per second dim " << dim << ":" << iter / diff.count() << endl;
    }
    */

    cv::imshow("Original Image", h_img);
    cv::imshow("Processed Image", d_result);

    cv::waitKey();
    return 0;
}