#include <opencv2/opencv.hpp>

void laplacianConvOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst)
{

    int k = 1;

    dst.create(src.rows - 2 * k, src.cols - 2 * k);
    dst = cv::Vec3b(0, 0, 0);

    const int laplace_mat[3][3] = {
    {-1, -1, -1},
    {-1,  8, -1},
    {-1, -1, -1}
    };

#pragma omp parallel for
    for (int i = k; i < src.rows - k; i++)
    {
        for (int j = k; j < src.cols - k; j++)
        {
            cv::Vec3f tmp = cv::Vec3f(0, 0, 0);
            for (int u = i - k; u <= i + k; u++)
            {
                for (int v = j - k; v <= j + k; v++)
                {
                    tmp += (cv::Vec3f)src.at<cv::Vec3b>(u, v) * laplace_mat[u - i + k][v - j + k];
                }
            }

            dst.at<cv::Vec3b>(i - k, j - k) = tmp;
        }
    }
}