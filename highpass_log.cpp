#include "highpass_log.h"
#include "convolution.h"
#include "kernels.h"
#include <cmath>

cv::Mat applyLaplacian(const cv::Mat& src) {
    std::vector<std::vector<double>> kernel = {
        {0,  1, 0},
        {1, -4, 1},
        {0,  1, 0}
    };
    return convolve2D(src, kernel, false);
}

cv::Mat applyLoG(const cv::Mat& src, double sigma) {
    int ksize = static_cast<int>(std::ceil(6.0 * sigma));
    if (ksize % 2 == 0) {
        ksize++;
    }
    if (ksize < 3) {
        ksize = 3;
    }
    auto kernel = createLoGKernel(ksize, sigma);
    return convolve2D(src, kernel, false);
}