#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY);

void gaussianCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY);

void gaussianConvOpenmp(cv::Mat_<uchar>& src, cv::Mat_<uchar>& dst, int kernelSize, int sigma);

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat_<uchar> h_img = cv::imread(argv[1]);
    cv::cuda::GpuMat d_img, d_result;
    cv::Mat_<uchar> h_result;


    d_img.upload(h_img);
    d_result.upload(h_img);

    cv::imshow("Original Image", d_img);

    bool cuda = true;

    int kernelSize = 3;

    int sigma = 15;

    if (cuda)
    {
        for (int dim = 1; dim < 64; dim++)
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
        }
    }
    else
    {
        auto begin = chrono::high_resolution_clock::now();
        const int iter = 10000;


        for (int i = 0; i < iter; i++)
        {
            gaussianConvOpenmp(h_img, h_result, kernelSize, sigma);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - begin;

        cv::imshow("Processed Image", d_result);

        cout << "Total time" << diff.count() << endl;
        cout << "Time per iteration" << diff.count() / iter << endl;
        cout << "Iterations per second" << iter / diff.count() << endl;
    }

    cv::waitKey();
    return 0;

    return 0;
}
