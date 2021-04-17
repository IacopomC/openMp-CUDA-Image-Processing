#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY);

void gaussianCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY);

void gaussianConvOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int kernelSize, int sigma);

void laplacianConvOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int kernelSize);

int main(int argc, char** argv)
{
    bool cuda = false;

    int kernelSize = 9;

    int sigma = 30;

    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat_<cv::Vec3b> h_img = cv::imread(argv[1]);
    cv::cuda::GpuMat d_img, d_result;
    cv::Mat_<cv::Vec3b> h_result;

    int border = (int)(kernelSize - 1) / 2;

    cv::copyMakeBorder(h_img, h_img, border, border, border, border, cv::BORDER_REPLICATE);

    d_img.upload(h_img);
    d_result.upload(h_img);

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

        gaussianCUDA(d_img, d_result, 32, 32);

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

        laplacianConvOpenmp(h_img, h_result, kernelSize);

        cv::imshow("Processed Image", h_result);
    }

    cv::waitKey();
    return 0;

    return 0;
}
