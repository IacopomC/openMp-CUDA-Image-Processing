#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>
#include <chrono>  // for high_resolution_clock

using namespace std;

void scalingCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, float scaleFactor);

void colorTransfCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& tmp_img, int dimX, int dimY, float angle);

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
    bool cuda = true; // true only if using CUDA

    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat_<cv::Vec3b> h_img = cv::imread(argv[1]);
    cv::Mat_<cv::Vec3b> h_result;

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
        cv::cuda::GpuMat d_result;
        cv::cuda::GpuMat d_img;
        
        cv::cuda::GpuMat d_tmp_img;
        cv::cuda::GpuMat d_kernel_1D;
        cv::Mat_<float> h_kernel_1D;

        const int kernelSizeGaussS = 5;
        int sigmaGaussS = 11;

        d_img.upload(h_img);
        d_result.upload(h_img);
        d_tmp_img.upload(h_img);

        int border = (int)(kernelSizeGaussS - 1) / 2;

        cv::copyMakeBorder(h_img, h_img, border, border, border, border, cv::BORDER_REPLICATE);

        h_kernel_1D = generateGaussianKernel1D(kernelSizeGaussS, sigmaGaussS);
        d_kernel_1D.upload(h_kernel_1D);

        gaussianSepCUDA(d_img, d_result, d_tmp_img, 32, 32, d_kernel_1D, kernelSizeGaussS, sigmaGaussS);
        */
        
        // ======== GAUSSIAN CONVOLUTION ======== //
        /*
        cv::cuda::GpuMat d_result;
        cv::cuda::GpuMat d_img;
        cv::cuda::GpuMat d_kernel;
        cv::Mat_<float> h_kernel;

        d_img.upload(h_img);
        d_result.upload(h_img);

        const int kernelSizeGaussC = 5;
        int sigmaGaussC = 11;

        int border = (int)(kernelSizeGaussC - 1) / 2;

        cv::copyMakeBorder(h_img, h_img, border, border, border, border, cv::BORDER_REPLICATE);

        h_kernel = generateGaussianKernel(kernelSizeGaussC, sigmaGaussC);
        d_kernel.upload(h_kernel);
        gaussianConvCUDA(d_img, d_result, 32, 32, d_kernel, kernelSizeGaussC, sigmaGaussC);
        */

        // ======== IMAGE COMBINATION ======== //
        /*
        cv::cuda::GpuMat d_result;
        cv::cuda::GpuMat d_img;
        cv::Mat_<cv::Vec3b> h_img2 = cv::imread(argv[2]);
        cv::cuda::GpuMat d_img2;

        d_img.upload(h_img);
        d_result.upload(h_img);
        d_img2.upload(h_img2);

        int imageComb = 0; // 0, 1, 2, 3
        float offSet = 0.5;
        float scaleFactor = 0.5;

        imageCombCUDA(d_img, d_result, d_img2, 32, 32, imageComb, offSet, scaleFactor);
        */
         
        // ======== LAPLACIAN ======== //
        /*
        cv::cuda::GpuMat d_result;
        cv::cuda::GpuMat d_img;
        d_img.upload(h_img);
        d_result.upload(h_img);

        laplacianCUDA(d_img, d_result, 32, 32);
        */

        // ======== DENOISING ======== //
        /*
        cv::cuda::GpuMat d_result;
        cv::cuda::GpuMat d_img;

        d_img.upload(h_img);
        d_result.upload(h_img);

        int percent = 50;
        const int kernelSize = 5;
        
        denoisingCUDA(d_img, d_result, 32, 32, kernelSize, percent);        
        */

        // ======== SCALING ======== //

        /*
        cv::cuda::GpuMat d_result;
        cv::cuda::GpuMat d_img;

        float scaling = 2; // don't go higher than 4

        cv::Size orig_Size = h_img.size();
        cv::Size new_Size(orig_Size.width* scaling, orig_Size.height* scaling);
        //cv::Mat h_img_resized = cv::Mat(new_Size, CV_8UC3);
        cv::Mat_<cv::Vec3b> h_img_resized(new_Size);

        d_img.upload(h_img);
        d_result.upload(h_img_resized);

        scalingCUDA(d_img, d_result, 32, 32, scaling);
        */
        

        // ======== COLOR TRANSFORM ======== //
        
        cv::cuda::GpuMat d_result;
        cv::cuda::GpuMat d_img;
        cv::cuda::GpuMat d_tmp_img_c;

        d_img.upload(h_img);
        d_result.upload(h_img);
        d_tmp_img_c.upload(h_img);

        float angle = 40;

        colorTransfCUDA(d_img, d_result, d_tmp_img_c, 32, 32, angle);
        

        //d_result.download(h_result);
        //std::cout << h_result;

        cv::imshow("Original Image", d_img);
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
        
        /*
        const int kernelSize = 5;
        int sigma = 11;
        gaussianConvOpenmp(h_img, h_result, kernelSize, sigma);
        */
         
        // ======== LAPLACIAN ======== //
        
        /*
        laplacianConvOpenmp(h_img, h_result);
        */
        
        // ======== COLOR TRANSFORM ======== //
        /*
        float angle = 40;
        colorTransfOpenmp(h_img, h_result, angle);
        */

        // ======== IMAGE COMBINATION ======== //

        /*
        cv::Mat_<cv::Vec3b> h_img2 = cv::imread(argv[2]);
        
        int imageComb = 0; // 0, 1, 2, 3
        float offSet = 0.5;
        float scaleFactor = 0.5;
        
        imageCombOpenmp(h_img, h_result, h_img2, imageComb, offSet, scaleFactor);
        */

        // ======== GAUSSIAN SEPARABLE ======== //
        
        /*
        cv::Mat_<cv::Vec3b> tmp_img;

        const int kernelSize = 5;
        int sigma = 11;

        gaussianSepOpenmp(h_img, h_result, tmp_img, kernelSize, sigma);
        */

        // ======== DENOISING ======== //
        
        /*
        const int kernelSize = 5;
        int percent = 50;

        denoisingOpenmp(h_img, h_result, kernelSize, percent);
        */

        // ======== SCALING ======== //
        
        /*
        float scaling = 2; // don't go higher than 4

        scalingOpenmp(h_img, h_result, scaling);
        */

        cv::imshow("Original Image", h_img);
        cv::imshow("Processed Image", h_result);
    }

    cv::waitKey();
    return 0;
}
