#include "sharpening.h"
#include "convolution.h"
#include "kernels.h"



cv::Mat applySharpening(const cv::Mat& src, double alpha, int ksize, double sigma, double T = 0.0) {
    auto gauss = createGaussianKernel(ksize, sigma);
    cv::Mat blurred = convolve2D(src, gauss, false);

    cv::Mat dst(src.size(), CV_8UC1);
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            const double original = static_cast<double>(src.at<uchar>(i, j));
            const double blur_val = blurred.at<double>(i, j);

            const double diff = original - blur_val;

            double sharp;

            if (std::abs(diff) > T) {
                sharp = original + alpha * diff;
            }
            else {
                sharp = original;
            }

            if (sharp < 0.0) {
                sharp = 0.0;
            }
            if (sharp > 255.0) {
                sharp = 255.0;
            }

            dst.at<uchar>(i, j) = static_cast<uchar>(sharp);
        }
    }

    return dst;
}