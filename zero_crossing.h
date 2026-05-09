#pragma once
#include <opencv2/core.hpp>
#include <stdexcept>

cv::Mat detectZeroCrossings(const cv::Mat& float_response_map);