#pragma once
#include <opencv2/core.hpp>

cv::Mat applyLaplacian(const cv::Mat& src);
cv::Mat applyLoG(const cv::Mat& src, double sigma);