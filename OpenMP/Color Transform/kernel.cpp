#include <opencv2/opencv.hpp>

cv::Vec3f bgr2xyz(const cv::Vec3f src) {

    float scr_r = src[2] / 255.0;
    float scr_g = src[1] / 255.0;
    float scr_b = src[0] / 255.0;

    cv::Vec3f tmp = cv::Vec3f(
        (scr_r > .04045) ? pow((scr_r + .055) / 1.055, 2.4) : scr_r / 12.92,
        (scr_g > .04045) ? pow((scr_g + .055) / 1.055, 2.4) : scr_g / 12.92,
        (scr_b > .04045) ? pow((scr_b + .055) / 1.055, 2.4) : scr_b / 12.92
    );

    float mat[3][3] = {
        .4124, .3576, .1805,
        .2126, .7152, .0722,
        .0193, .1192, .9505
    };

    tmp *= 100.0;

    cv::Vec3f xyz;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            xyz[i] += mat[i][j] * tmp[j];
        }
    }

    return xyz;
}
cv::Vec3f xyz2lab(const cv::Vec3f src) {

    float scr_z = src[2] / 108.883;
    float scr_y = src[1] / 100.;
    float scr_x = src[0] / 95.047;

    cv::Vec3f v = cv::Vec3f(
        (scr_x > .008856) ? pow(scr_x, 1. / 3.) : (7.787 * scr_x) + (16. / 116.),
        (scr_y > .008856) ? pow(scr_y, 1. / 3.) : (7.787 * scr_y) + (16. / 116.),
        (scr_z > .008856) ? pow(scr_z, 1. / 3.) : (7.787 * scr_z) + (16. / 116.)
    );

    cv::Vec3f lab = cv::Vec3f((116. * v[1]) - 16., 500. * (v[0] - v[1]), 200. * (v[1] - v[2]));

    return lab;
}

cv::Vec3f bgr2lab(cv::Vec3f c) {
    return xyz2lab(bgr2xyz(c));
}

cv::Vec3f lab2xyz(cv::Vec3f src) {

    float fy = (src[0] + 16.0) / 116.0;
    float fx = src[1] / 500.0 + fy;
    float fz = fy - src[2] / 200.0;

    return cv::Vec3f(
        95.047 * ((fx > 0.206897) ? fx * fx * fx : (fx - 16.0 / 116.0) / 7.787),
        100.000 * ((fy > 0.206897) ? fy * fy * fy : (fy - 16.0 / 116.0) / 7.787),
        108.883 * ((fz > 0.206897) ? fz * fz * fz : (fz - 16.0 / 116.0) / 7.787)
    );
}
cv::Vec3f xyz2bgr(cv::Vec3f src) {

    src /= 100.0;

    float mat[3][3] = {
        3.2406, -1.5372, -0.4986,
        -0.9689, 1.8758, 0.0415,
        0.0557, -0.2040, 1.0570
    };

    cv::Vec3f tmp;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            tmp[i] += mat[i][j] * src[j];
        }
    }

    cv::Vec3f bgr;
    bgr[2] = (tmp[0] > 0.0031308) ? ((1.055 * pow(tmp[0], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[0]);
    bgr[1] = (tmp[1] > 0.0031308) ? ((1.055 * pow(tmp[1], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[1]);
    bgr[0] = (tmp[2] > 0.0031308) ? ((1.055 * pow(tmp[2], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[2]);

    bgr *= 255.0;

    return bgr;
}

cv::Vec3f lab2bgr(const cv::Vec3f src) {
    return xyz2bgr(lab2xyz(src));
}

void colorTransfOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, float angle)
{

    float PI = 3.14159265358979323846;

    dst.create(src.rows, src.cols);
    dst = cv::Vec3b(0, 0, 0);

    cv::Mat_<cv::Vec3f> lab;
    lab.create(src.rows, src.cols);
    lab = cv::Vec3f(0, 0, 0);

    #pragma omp parallel for
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            lab(i, j) = bgr2lab((cv::Vec3f)src.at<cv::Vec3b>(i, j));

            float C = sqrt(lab(i, j)[1] * lab(i, j)[1] + lab(i, j)[2] * lab(i, j)[2]);
            float h = atan2(lab(i, j)[2], lab(i, j)[1]);
            h += (angle * PI) / 180.0;
            lab(i, j)[1] = cos(h) * C;
            lab(i, j)[2] = sin(h) * C;
            dst(i, j) = lab2bgr((cv::Vec3f)lab.at<cv::Vec3f>(i, j));
        }
    }
}