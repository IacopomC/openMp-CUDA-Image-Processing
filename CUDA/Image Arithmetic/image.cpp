#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>
#include <chrono>  // for high_resolution_clock

using namespace std;

void imageCombCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& src2, int dimX, int dimY, int imageComb, float offSet, float scaleFactor);

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat_<cv::Vec3b> h_img = cv::imread(argv[1]);
    cv::cuda::GpuMat d_result;
    cv::cuda::GpuMat d_img;
    cv::Mat_<cv::Vec3b> h_img2 = cv::imread(argv[2]);
    cv::cuda::GpuMat d_img2;

    d_img.upload(h_img);
    d_result.upload(h_img);
    d_img2.upload(h_img2);

    int imageComb = atof(argv[3]); // Sum:0, Sub:1, Mul:2, Div:3
    float offSet = atof(argv[4]);
    float scaleFactor = atof(argv[5]);

    // ============= SUM BEST DIM = 2. TESTED WITH CODE BELOW ============= //
    // ============= SUB BEST DIM = 2. TESTED WITH CODE BELOW ============= //
    // ============= MUL BEST DIM = 4. TESTED WITH CODE BELOW ============= //
    // ============= DIV BEST DIM = 2. TESTED WITH CODE BELOW ============= //
    imageCombCUDA(d_img, d_result, d_img2, 2, 2, imageComb, offSet, scaleFactor);

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