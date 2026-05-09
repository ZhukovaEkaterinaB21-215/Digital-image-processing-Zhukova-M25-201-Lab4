#include "threshold_avg.h"
#include "convolution.h"
#include "kernels.h"

cv::Mat applyAveragingWithThreshold(const cv::Mat& src, int ksize, double T) {
    auto box = createBoxKernel(ksize);
    cv::Mat avg_float = convolve2D(src, box, false);
    cv::Mat dst(src.size(), CV_8UC1);

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            const double avg = avg_float.at<double>(i, j);
            const double diff = std::abs(static_cast<double>(src.at<uchar>(i, j)) - avg);

            double out = (diff > T) ? avg : src.at<uchar>(i, j);

            if (out < 0.0) {
                out = 0.0;
            }
            if (out > 255.0) {
                out = 255.0;
            }
            dst.at<uchar>(i, j) = static_cast<uchar>(out);
        }
    }
    return dst;
}