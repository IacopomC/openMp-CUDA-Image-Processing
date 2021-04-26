#include <opencv2/opencv.hpp>

void imageCombOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<cv::Vec3b>& src2, int imageComb, float offSet, float scaleFactor)
{

    dst.create(src.rows, src.cols);
    dst = cv::Vec3f(0, 0, 0);

    #pragma omp parallel for
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            for (int c = 0; c < 3; c++)
            {
                if (imageComb == 0) {
                    dst(i, j)[c] = src(i, j)[c] + src2(i, j)[c];
                }
                else if (imageComb == 1) {
                    dst(i, j)[c] = src(i, j)[c] - src2(i, j)[c];
                }
                else if (imageComb == 2) {
                    dst(i, j)[c] = src(i, j)[c] * src2(i, j)[c];
                }
                else if (imageComb == 3) {
                    dst(i, j)[c] = src2(i, j)[c] == 0 ? src(i, j)[c] : (src(i, j)[c] / src2(i, j)[c]);
                }
            }

            for (int c = 0; c < 3; c++)
            {
                dst(i, j)[c] *= scaleFactor;
                dst(i, j)[c] += offSet;
            }
        }
    }
}
