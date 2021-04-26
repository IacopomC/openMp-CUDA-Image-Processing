#include <opencv2/opencv.hpp>

float lerp(float s, float e, float t) {
    return s + (e - s) * t;
}

cv::Vec3f bInterp(cv::Vec3f c00, cv::Vec3f c10, cv::Vec3f c01, cv::Vec3f c11, float tx, float ty) {

    float bchannel = lerp(lerp(c00[0], c10[0], tx), lerp(c01[0], c11[0], tx), ty);
    float gchannel = lerp(lerp(c00[1], c10[1], tx), lerp(c01[1], c11[1], tx), ty);
    float rchannel = lerp(lerp(c00[2], c10[2], tx), lerp(c01[2], c11[2], tx), ty);

    return cv::Vec3f(bchannel, gchannel, rchannel);
}

void scalingOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, float scaling)
{
    dst.create(scaling * src.rows, scaling * src.cols);
    dst = cv::Vec3b(0, 0, 0);

    float x_ratio = float(src.cols - 1) / dst.cols;
    float y_ratio = float(src.rows - 1) / dst.rows;

    #pragma omp parallel for
    for (int row = 0; row < dst.rows; row++)
    {
        for (int col = 0; col < dst.cols; col++)
        {
            float gx, gy, tx, ty;

            // Break into an integral and a fractional part
            gx, tx = std::modf(row * x_ratio, &gx);
            gy, ty = std::modf(col * y_ratio, &gy);
            cv::Vec3f c00 = (cv::Vec3f)src.at<cv::Vec3b>(gx, gy);
            cv::Vec3f c10 = (cv::Vec3f)src.at<cv::Vec3b>(gx + 1, gy);
            cv::Vec3f c01 = (cv::Vec3f)src.at<cv::Vec3b>(gx, gy + 1);
            cv::Vec3f c11 = (cv::Vec3f)src.at<cv::Vec3b>(gx + 1, gy + 1);
            dst.at<cv::Vec3b>(row, col) = bInterp(c00, c10, c01, c11, tx, ty);
        }
    }
}