#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void gaussianConvOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int kernelSize, int sigma);

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

int main(int argc, char** argv)
{
	cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

	cv::Mat_<cv::Vec3b> h_img = cv::imread(argv[1]);
	cv::Mat_<cv::Vec3b> h_result;
    cv::Mat_<cv::Vec3b> input_img;

    const int kernelSize = atof(argv[2]);
    int sigma = atof(argv[3]);

    int border = (int)(kernelSize - 1) / 2;

    cv::copyMakeBorder(h_img, input_img, border, border, border, border, cv::BORDER_REPLICATE);

    gaussianConvOpenmp(input_img, h_result, kernelSize, sigma);
	
    /*
    auto begin = chrono::high_resolution_clock::now();
    const int iter = 100;

    for (int i = 0; i < iter; i++)
    {
        gaussianConvOpenmp(h_img, h_result, kernelSize, sigma);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cout << "Total time " << diff.count() << endl;
    cout << "Time per iteration " << diff.count() / iter << endl;
    cout << "Iterations per second " << iter / diff.count() << endl;
    */

    cv::imshow("Original Image", h_img);
    cv::imshow("Processed Image", h_result);

    cv::waitKey();
    return 0;
}