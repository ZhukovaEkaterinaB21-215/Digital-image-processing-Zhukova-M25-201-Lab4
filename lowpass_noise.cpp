#include "lowpass_noise.h"
#include "convolution.h"
#include "kernels.h"

constexpr double PI = 3.14159265358979323846;

static double rand_uniform() {
    return static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0);
}


static double rand_gaussian() {

    double u1 = rand_uniform();
    double u2 = rand_uniform();

    if (u1 < 1e-10) {
        u1 = 1e-10;
    }

    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2);
}


cv::Mat addGaussianNoise(const cv::Mat& src, double sigma) {
    if (src.empty() || src.channels() != 1 || src.depth() != CV_8U) {
        return cv::Mat();
    }

    cv::Mat dst(src.size(), CV_8UC1);

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {

            double val = src.at<uchar>(i, j) + sigma * rand_gaussian();

            if (val < 0.0) {
                val = 0.0;
            }
            if (val > 255.0) {
                val = 255.0;
            }

            dst.at<uchar>(i, j) = static_cast<uchar>(val);
        }
    }
    return dst;
}


cv::Mat addSaltPepperNoise(const cv::Mat& src, double density) {
    if (src.empty() || src.channels() != 1 || src.depth() != CV_8U) {
        return cv::Mat();
    }
    cv::Mat dst(src.size(), CV_8UC1);

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            dst.at<uchar>(i, j) = src.at<uchar>(i, j);
        }
    }

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            if (rand_uniform() < density) {
                dst.at<uchar>(i, j) = (rand() & 1) ? 255 : 0;
            }
        }
    }
    return dst;
}

cv::Mat applyLowPass(const cv::Mat& src, int ksize, double sigma) {
    return convolve2D(src, createGaussianKernel(ksize, sigma), true);
}

cv::Mat applyGaussianLPF(const cv::Mat& src, int ksize, double sigma) {
    return convolve2D(src, createGaussianKernel(ksize, sigma), true);
}

cv::Mat applyAveragingLPF(const cv::Mat& src, int ksize) {
    return convolve2D(src, createBoxKernel(ksize), true);
}

double calcVariance(const cv::Mat& img) {
    double sum = 0, sq = 0; 
    int n = img.rows * img.cols;

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            double p = img.at<uchar>(i, j);
            sum += p; sq += p * p;
        }
    }
    return (sq / n) - (sum / n) * (sum / n);
}




