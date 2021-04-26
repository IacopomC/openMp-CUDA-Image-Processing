#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>
#include <chrono>  // for high_resolution_clock

using namespace std;

void denoisingCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, int kernelSize, int percent);

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat_<cv::Vec3b> h_img = cv::imread(argv[1]);
    cv::cuda::GpuMat d_result;
    cv::cuda::GpuMat d_img;

    cv::Mat_<cv::Vec3b> input_img;

    int percent = atof(argv[2]);
    const int kernelSize = atof(argv[3]);

    int border = (int)(kernelSize - 1) / 2;

    cv::copyMakeBorder(h_img, input_img, border, border, border, border, cv::BORDER_REPLICATE);

    d_img.upload(input_img);
    d_result.upload(input_img);

    // ============= BEST DIM = 4. TESTED WITH CODE BELOW ============= //
    denoisingCUDA(d_img, d_result, 6, 6, kernelSize, percent);

    // crop final image to original size
    d_result = d_result(cv::Range(border + 1, d_result.rows - border), cv::Range(border + 1, d_result.cols - border)).clone();

    /*
    for (int dim = 1; dim < 12; dim++)
    {
        auto begin = chrono::high_resolution_clock::now();
        const int iter = 100;


        for (int i = 0; i < iter; i++)
        {
            imageCombCUDA(d_img, d_result, d_img2, dim, dim, imageComb, offSet, scaleFactor);
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