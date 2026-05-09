#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>

cv::Mat convolve2D(const cv::Mat& src, const std::vector<std::vector<double>>& kernel, bool clamp_to_8u = false);