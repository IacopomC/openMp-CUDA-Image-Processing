#include <opencv2/opencv.hpp>

void gaussianConvOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int kernelSize, int sigma)
{

    int k = (kernelSize - 1) / 2;

    dst.create(src.rows - 2 * k, src.cols - 2 * k);
    dst = cv::Vec3b(0, 0, 0);

    #pragma omp parallel for
    for (int i = k; i < src.rows - k; i++)
    {
        for (int j = k; j < src.cols - k; j++)
        {
            float tmp[3] = { 0.0, 0.0, 0.0, };
            float kernelSum[3] = { 0.0, 0.0, 0.0, };
            for (int u = i - k; u <= i + k; u++)
            {
                for (int v = j - k; v <= j + k; v++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        tmp[c] += (float)src(u, v)[c] * exp(-((u - i) * (u - i) + (v - j) * (v - j)) / (2.0 * sigma * sigma));
                        kernelSum[c] += exp(-((u - i) * (u - i) + (v - j) * (v - j)) / (2.0 * sigma * sigma));
                    }
                }
            }

            for (int c = 0; c < 3; c++)
            {
                dst(i - k, j - k)[c] = (unsigned char)(tmp[c] / kernelSum[c]);

            }
        }
    }
}

void laplacianConvOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst)
{

    int k = 1;

    dst.create(src.rows - 2 * k, src.cols - 2 * k);
    dst = cv::Vec3b(0, 0, 0);

    float laplacian[9] = { -1.0, -1.0, -1.0, -1.0 , 8.0, -1.0, -1.0, -1.0, -1.0 };

    #pragma omp parallel for
    for (int i = k; i < src.rows - k; i++)
    {
        for (int j = k; j < src.cols - k; j++)
        {
            int counter = 0;
            float tmp[3] = { 0.0, 0.0, 0.0, };
            for (int u = i - k; u <= i + k; u++)
            {
                for (int v = j - k; v <= j + k; v++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        tmp[c] += (float)src(u, v)[c] * laplacian[counter];
                    }
                    counter++;
                }
            }

            for (int c = 0; c < 3; c++)
            {
                tmp[c] /= 255.0;
                tmp[c] = pow(tmp[c], 1.0 / 0.2);
                tmp[c] *= 255.0;
                dst(i - k, j - k)[c] = (unsigned char)tmp[c];

            }
        }
    }
}
