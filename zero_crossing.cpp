#include "zero_crossing.h"

cv::Mat detectZeroCrossings(const cv::Mat& float_map) {
    if (float_map.type() != CV_64F)
        throw std::invalid_argument("CV_64F для обработки отрицательных и дробных значений");

    double sum_abs = 0.0;
    for (int i = 0; i < float_map.rows; ++i) {
        for (int j = 0; j < float_map.cols; ++j) {
            sum_abs += std::abs(float_map.at<double>(i, j));
        }
    }
    double T = (3.0 / (4.0 * float_map.rows * float_map.cols)) * sum_abs;

    cv::Mat dst(float_map.size(), CV_8UC1, cv::Scalar(0));

    for (int i = 1; i < float_map.rows - 1; ++i) {
        for (int j = 1; j < float_map.cols - 1; ++j) {
            double c = float_map.at<double>(i, j);

            const std::pair<int, int> neighbors[4] = {
                {-1, 0}, {1, 0}, {0, -1}, {0, 1}
            };

            for (const auto& [di, dj] : neighbors) {
                double neighbor = float_map.at<double>(i + di, j + dj);

                bool sign_change = (c < 0 && neighbor > 0) || (c > 0 && neighbor < 0);

                double diff = std::abs(c - neighbor);
                bool magnitude_ok = (diff >= T);

                if (sign_change && magnitude_ok) {
                    dst.at<uchar>(i, j) = 255;
                    break;
                }
            }
        }
    }
    return dst;
}