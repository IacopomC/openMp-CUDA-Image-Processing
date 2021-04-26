#include <opencv2/opencv.hpp>

void gaussianSepOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<cv::Vec3b>& tmp_img, int kernelSize, int sigma)
{

    int k = (kernelSize - 1) / 2;

    dst.create(src.rows - 2 * k, src.cols - 2 * k);
    dst = cv::Vec3b(0, 0, 0);

    tmp_img.create(src.rows - 2 * k, src.cols);
    tmp_img = cv::Vec3b(0, 0, 0);

#pragma omp parallel for
    for (int i = k; i < src.rows - k; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            float tmp[3] = { 0.0, 0.0, 0.0, };
            float kernelSum[3] = { 0.0, 0.0, 0.0, };
            for (int u = i - k; u <= i + k; u++)
            {
                for (int c = 0; c < 3; c++)
                {
                    tmp[c] += (float)src(u, j)[c] * exp(-((u - i) * (u - i)) / (2.0 * sigma * sigma));
                    kernelSum[c] += exp(-((u - i) * (u - i)) / (2.0 * sigma * sigma));
                }
            }

            for (int c = 0; c < 3; c++)
            {
                tmp_img(i - k, j)[c] = (unsigned char)(tmp[c] / kernelSum[c]);

            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < tmp_img.rows; i++)
    {
        for (int j = k; j < tmp_img.cols - k; j++)
        {
            float tmp[3] = { 0.0, 0.0, 0.0, };
            float kernelSum[3] = { 0.0, 0.0, 0.0, };
            for (int u = j - k; u <= j + k; u++)
            {
                for (int c = 0; c < 3; c++)
                {
                    tmp[c] += (float)tmp_img(i, u)[c] * exp(-((u - j) * (u - j)) / (2.0 * sigma * sigma));
                    kernelSum[c] += exp(-((u - j) * (u - j)) / (2.0 * sigma * sigma));
                }
            }

            for (int c = 0; c < 3; c++)
            {
                dst(i, j - k)[c] = (unsigned char)(tmp[c] / kernelSum[c]);

            }
        }
    }

}
