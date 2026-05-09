#pragma once
#include <opencv2/core.hpp>

cv::Mat applyAveragingWithThreshold(const cv::Mat& src, int ksize, double threshold);
