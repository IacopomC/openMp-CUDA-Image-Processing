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

//----------------------CIE Lch----------------------
cv::Vec3b HsvToRgb(cv::Vec3b hsv)
{
    cv::Vec3f rgb;
    int region;
    float h, s, v, p, q, t, ff;
    long i;

    if (hsv[1] == 0)
    {
        rgb[2] = hsv[2];
        rgb[1] = hsv[2];
        rgb[0]  = hsv[2];
        return rgb;
    }

    h = hsv[0];
    s = hsv[1];
    v = hsv[2];

    region = h / 60;
    i = (long)region;

    ff = region - i;

    p = v * (1.0 - s);
    q = v * (1.0 - (s * ff));
    t = v * (1.0 - (s * (1.0 - ff)));

    switch (region)
    {
    case 0:
        rgb[2] = v;
        rgb[1] = t;
        rgb[0]  = p;
        break;
    case 1:
        rgb[2] = q;
        rgb[1] = v;
        rgb[0]  = p;
        break;
    case 2:
        rgb[2] = p;
        rgb[1] = v;
        rgb[0]  = t;
        break;
    case 3:
        rgb[2] = p;
        rgb[1] = q;
        rgb[0]  = v;
        break;
    case 4:
        rgb[2] = t;
        rgb[1] = p;
        rgb[0]  = v;
        break;
    default:
        rgb[2] = v;
        rgb[1] = p;
        rgb[0]  = q;
        break;
    }

    return rgb;
}

cv::Vec3b RgbToHsv(cv::Vec3b rgb)
{
    cv::Vec3f hsv;
    float rgbMin, rgbMax;
    float r, g, b;

    r = rgb[2] / 255.0;
    g = rgb[1] / 255.0;
    b = rgb[0] / 255.0;

    rgbMin = r < g ? (r < b ? r : b) : (g < b ? g : b );
    rgbMax = r > g ? (r > b  ? r : b ) : (g > b  ? g : b );

    hsv[2] = rgbMax;

    if (hsv[2] == 0)
    {
        hsv[0] = 0;
        hsv[1] = 0;
        return hsv;
    }

    hsv[1] = ((rgbMax - rgbMin) / hsv[2]);

    if (hsv[1] == 0)
    {
        hsv[0] = 0;
        return hsv;
    }

    if (rgbMax == r)
        hsv[0] = (60 * (g - b) / (rgbMax - rgbMin));
    else if (rgbMax == g)
        hsv[0] = (60 * (2.0 + (b  - r) / (rgbMax - rgbMin)));
    else
        hsv[0] = (60 * (4.0 + (r - g) / (rgbMax - rgbMin)));

    return hsv;
}



void colorTransfOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, float angle)
{

    float PI = 3.14159265358979323846;

    dst.create(src.rows, src.cols);
    dst = cv::Vec3b(0, 0, 0);

    //#pragma omp parallel for
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            dst(i,j) = RgbToHsv((cv::Vec3b)src(i,j));
            //dst(i, j)[0] += angle;
            dst(i, j) = HsvToRgb(dst(i, j));
        }
    }
}