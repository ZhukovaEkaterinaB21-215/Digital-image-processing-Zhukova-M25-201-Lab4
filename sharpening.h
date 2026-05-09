#pragma once
#include <opencv2/core.hpp>

cv::Mat applySharpening(const cv::Mat& src, double alpha, int ksize, double sigma, double T);