#include <opencv2/opencv.hpp>


void denoisingOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int kernelSize, int percent)
{

    int k = (kernelSize - 1) / 2;

    dst.create(src.rows - 2 * k, src.cols - 2 * k);
    dst = cv::Vec3f(0, 0, 0);

#pragma omp parallel for
    for (int i = k; i < src.rows - k; i++)
    {
        for (int j = k; j < src.cols - k; j++)
        {
            std::vector<float> rNeighbors;
            std::vector<float> gNeighbors;
            std::vector<float> bNeighbors;

            float rValue = 0.0;
            float gValue = 0.0;
            float bValue = 0.0;

            for (int u = i - k; u <= i + k; u++)
            {
                for (int v = j - k; v <= j + k; v++)
                {
                    rNeighbors.push_back(src.at<cv::Vec3b>(u, v)[2]);
                    gNeighbors.push_back(src.at<cv::Vec3b>(u, v)[1]);
                    bNeighbors.push_back(src.at<cv::Vec3b>(u, v)[0]);
                }
            }

            std::sort(rNeighbors.begin(), rNeighbors.end());
            std::sort(gNeighbors.begin(), gNeighbors.end());
            std::sort(bNeighbors.begin(), bNeighbors.end());

            int medianIndx = 0;
            if (kernelSize % 2 == 0) {
                medianIndx = kernelSize * kernelSize / 2;
            }
            else {
                medianIndx = (kernelSize * kernelSize - 1) / 2;
            }

            int numEl = (kernelSize * kernelSize * int(percent) / 100) / 2;

            for (int w = (medianIndx - numEl); w <= (medianIndx + numEl); w++) {
                rValue += rNeighbors[w];
                gValue += gNeighbors[w];
                bValue += bNeighbors[w];
            }

            if (numEl >= 1) {
                rValue /= float(2 * numEl + 1);
                gValue /= float(2 * numEl + 1);
                bValue /= float(2 * numEl + 1);
            }

            dst(i - k, j - k) = cv::Vec3f(bValue, gValue, rValue);

        }
    }
}