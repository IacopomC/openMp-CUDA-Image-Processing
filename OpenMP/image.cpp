#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void scalingOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, float scaling);

int main(int argc, char** argv)
{
	cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

	cv::Mat_<cv::Vec3b> h_img = cv::imread(argv[1]);
	cv::Mat_<cv::Vec3b> h_result;

	float scaleFactor = atof(argv[2]);

	scalingOpenmp(h_img, h_result, scaleFactor);
	
    /*
    auto begin = chrono::high_resolution_clock::now();
    const int iter = 100;

    for (int i = 0; i < iter; i++)
    {
        scalingOpenmp(h_img, h_result, scaleFactor);
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