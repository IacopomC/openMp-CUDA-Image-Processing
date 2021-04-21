#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void denoisingCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, int kernelSize, int percent);

void gaussianSepCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& tmp_img, int dimX, int dimY, cv::cuda::GpuMat& d_kernelGaussConv, int kernelSize, int sigma);

void gaussianConvCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, cv::cuda::GpuMat& d_kernelGaussConv, int kernelSize, int sigma);

void laplacianCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY);

void imageCombCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& src2, int dimX, int dimY, int imageComb, float offSet, float scaleFactor);

void gaussianConvOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int kernelSize, int sigma);

void laplacianConvOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst);

void gaussianSepOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<cv::Vec3b>& tmp_img, int kernelSize, int sigma);

void colorTransfOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, float angle);

void imageCombOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<cv::Vec3b>& src2, int imageComb, float offSet, float scaleFactor);

void denoisingOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int kernelSize, int percent);

void scalingOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, float scaling);

cv::Mat_<float> generateGaussianKernel(int kernelSize, int sigma)
{
    float PI = 3.14159265358979323846;
    float constant = 1.0 / (2.0 * PI * pow(sigma, 2));
    int radius = (kernelSize - 1.0) / 2.0;
    cv::Mat_<float> h_kernel(kernelSize, kernelSize);

    float sum = 0.0;
    for (int i = -radius; i < radius + 1; i++) {
        for (int j = -radius; j < radius + 1; j++)
        {
            h_kernel[i + radius][j + radius] = constant * (exp(-(pow(i, 2) + pow(j, 2)) / (2 * pow(sigma, 2))));
            sum += h_kernel(i + radius, j + radius);
        }

    }

    for (int i = 0; i < kernelSize; ++i)
        for (int j = 0; j < kernelSize; ++j)
            h_kernel(i, j) /= sum;

    return h_kernel;
}

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
    bool cuda = false; // true only if using CUDA
    bool kernel = true; // true only if using a kernel

     // ========== OPENMP ========== //

    // gaussian separable parameters
    cv::Mat_<cv::Vec3b> tmp_img;

    // gaussian parameters
    int kernelSize = 1; // for laplacian and scaling set equal to 1
    int sigma = 30;

    // color transform parameters
    float angle = 360.0;

    // image combination parameters
    int imageComb = 0;
    float offSet = 0.5;
    float scaleFactor = 0.5;

    // denoising parameters
    int percent = 50;

    // scaling parameters
    float scaling = 4; // don't go higher than 4

    // =========== CUDA =========== //

    // CUDA gaussian convolution params
    const int kernelSizeGaussC = 5;
    int sigmaGaussC = 11;
    cv::cuda::GpuMat d_kernel;
    cv::Mat_<float> h_kernel;


    // CUDA gaussian separable params
    const int kernelSizeGaussS = 5;
    int sigmaGaussS = 11;
    cv::cuda::GpuMat d_tmp_img;
    cv::cuda::GpuMat d_kernel_1D;
    cv::Mat_<float> h_kernel_1D;

    // CUDA denoising params
    cv::cuda::GpuMat d_rNeighbor;
    cv::Mat_<float> h_rNeighbor;
    cv::cuda::GpuMat d_gNeighbor;
    cv::Mat_<float> h_gNeighbor;
    cv::cuda::GpuMat d_bNeighbor;
    cv::Mat_<float> h_bNeighbor;

    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat_<cv::Vec3b> h_img = cv::imread(argv[1]);
    cv::Mat_<cv::Vec3b> h_img2 = cv::imread(argv[2]);
    cv::cuda::GpuMat d_img, d_img2, d_result;
    cv::Mat_<cv::Vec3b> h_result;

    d_img2.upload(h_img2);
    d_result.upload(h_img);
    d_tmp_img.upload(h_img);

    // change kernel size with corresponding parameter of used filter
    int border = (int)(kernelSizeGaussS - 1) / 2;

    if (kernel) {
        cv::copyMakeBorder(h_img, h_img, border, border, border, border, cv::BORDER_REPLICATE);
    }

    d_img.upload(h_img);

    cv::imshow("Original Image", d_img);

    if (cuda)
    {
        /*for (int dim = 1; dim < 64; dim++)
        {
            auto begin = chrono::high_resolution_clock::now();
            const int iter = 10000;


            for (int i = 0; i < iter; i++)
            {
                gaussianCUDA(d_img, d_result, dim, dim);
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - begin;

            cv::imshow("Processed Image", d_result);

            cout << dim << ":" << diff.count() << endl;
            cout << dim << ":" << diff.count() / iter << endl;
            cout << dim << ":" << iter / diff.count() << endl;
        }*/

        // ======== GAUSSIAN SEPARABLE======== //
        /*
        h_kernel_1D = generateGaussianKernel1D(kernelSizeGaussS, sigmaGaussS);
        d_kernel_1D.upload(h_kernel_1D);
        gaussianSepCUDA(d_img, d_result, d_tmp_img, 32, 32, d_kernel_1D, kernelSizeGaussS, sigmaGaussS);
        */
        
        // ======== GAUSSIAN CONVOLUTION ======== //
        /*
        h_kernel = generateGaussianKernel(kernelSizeGaussC, sigmaGaussC);
        d_kernel.upload(h_kernel);
        gaussianConvCUDA(d_img, d_result, 32, 32, d_kernel, kernelSizeGaussC, sigmaGaussC);
        */

        // ======== IMAGE COMBINATION ======== //
        // imageCombCUDA(d_img, d_result, d_img2, 32, 32, imageComb, offSet, scaleFactor);
         
        // ======== LAPLACIAN ======== //
        // laplacianCUDA(d_img, d_result, 32, 32);

        // ======== DENOISING ======== //
        /*
        d_rNeighbor.upload(h_rNeighbor);
        d_gNeighbor.upload(h_gNeighbor);
        d_bNeighbor.upload(h_bNeighbor);*/
        denoisingCUDA(d_img, d_result, 32, 32, kernelSize, percent);

        // ======== SCALING ======== //
        

        /*d_result.download(h_result);
        std::cout << h_result;*/

        cv::imshow("Processed Image", d_result);
    }
    else
    {
        /*auto begin = chrono::high_resolution_clock::now();
        const int iter = 10000;


        for (int i = 0; i < iter; i++)
        {
            gaussianConvOpenmp(h_img, h_result, kernelSize, sigma);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - begin;

        cv::imshow("Processed Image", h_result);

        cout << "Total time" << diff.count() << endl;
        cout << "Time per iteration" << diff.count() / iter << endl;
        cout << "Iterations per second" << iter / diff.count() << endl;*/

        // ======== GAUSSIAN ======== //
        // gaussianConvOpenmp(h_img, h_result, kernelSize, sigma);
         
        // ======== LAPLACIAN ======== //
        // laplacianConvOpenmp(h_img, h_result);
        
        // ======== COLOR TRANSFORM ======== //
        // colorTransfOpenmp(h_img, h_result, angle);
        
        // ======== IMAGE COMBINATION ======== //
        // imageCombOpenmp(h_img, h_result, h_img2, imageComb, offSet, scaleFactor);

        // ======== GAUSSIAN SEPARABLE ======== //
        // gaussianSepOpenmp(h_img, h_result, tmp_img, kernelSize, sigma);

        // ======== DENOISING ======== //
        // denoisingOpenmp(h_img, h_result, kernelSize, percent);

        // ======== SCALING ======== //
        scalingOpenmp(h_img, h_result, scaling);

        cv::imshow("Processed Image", h_result);
        //std::cout << h_result;
    }

    cv::waitKey();
    return 0;
}
