#include "integral_image.h"
#include <stdexcept>

cv::Mat computeIntegralImage(const cv::Mat& src) {
    if (src.empty() || src.channels() != 1 || src.depth() != CV_8U) {
        throw std::invalid_argument("Требуется grayscale изображение и 8 бит");
    }

    const int H = src.rows;
    const int W = src.cols;

    cv::Mat integral(H + 1, W + 1, CV_64F, cv::Scalar(0.0));

  
    for (int i = 1; i <= H; ++i) {
        double row_sum = 0.0;
        for (int j = 1; j <= W; ++j) {
            row_sum += static_cast<double>(src.at<uchar>(i - 1, j - 1));
            integral.at<double>(i, j) = integral.at<double>(i - 1, j) + row_sum;
        }
    }

    return integral;
}

cv::Mat applyAveragingIntegral(const cv::Mat& src, int ksize) {

    if (src.empty() || src.channels() != 1 || src.depth() != CV_8U) {
        throw std::invalid_argument("Требуется grayscale изображение и 8 битp");
    }
    if (ksize <= 0 || ksize % 2 == 0) {
        throw std::invalid_argument("ksize должен быть нечётным");
    }

    const int H = src.rows;
    const int W = src.cols;
    const int half = ksize / 2;

   
    cv::Mat integral = computeIntegralImage(src);

    cv::Mat dst(H, W, CV_8UC1);

   
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            
            int x1 = j - half;
            int y1 = i - half;
            int x2 = j + half;
            int y2 = i + half;

            
            if (x1 < 0) {
                x1 = 0;
            }
            if (y1 < 0) {
                y1 = 0;
            }
            if (x2 >= W) {
                x2 = W - 1;
            }
            if (y2 >= H) {
                y2 = H - 1;
            }

            
            x1++;
            y1++;
            x2++;
            y2++;

            
            double sum = integral.at<double>(y2, x2) - integral.at<double>(y2, x1 - 1) - integral.at<double>(y1 - 1, x2) + integral.at<double>(y1 - 1, x1 - 1);

            int area = (x2 - x1 + 1) * (y2 - y1 + 1);
            double avg = sum / static_cast<double>(area);

            if (avg < 0.0) {
                avg = 0.0;
            }
            if (avg > 255.0) {
                avg = 255.0;
            }
            dst.at<uchar>(i, j) = static_cast<uchar>(avg);
        }
    }

    return dst;
}