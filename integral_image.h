#pragma once
#include <opencv2/core.hpp>

cv::Mat computeIntegralImage(const cv::Mat& src);
cv::Mat applyAveragingIntegral(const cv::Mat& src, int ksize);