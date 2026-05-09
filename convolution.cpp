#include "convolution.h"

cv::Mat convolve2D(const cv::Mat& src, const std::vector<std::vector<double>>& kernel, bool clamp_to_8u)
{
    if (src.empty()) {
        throw std::invalid_argument("Входное изображение пусто");
    }
    if (src.channels() != 1 || src.depth() != CV_8U) {
        throw std::invalid_argument("Поддерживается только grayscale и глубина 8 бит");
    }
    


    const int kh = static_cast<int>(kernel.size());
    const int kw = static_cast<int>(kernel[0].size());
    if (kh == 0 || kw == 0) {
        throw std::invalid_argument("Ядро не должно быть пустым");
    }
    if (kh % 2 == 0 || kw % 2 == 0) {
        throw std::invalid_argument("Ядро должно иметь нечётные размеры");
    }

    const int padH = kh / 2;
    const int padW = kw / 2;

    cv::Mat dst;
    if (clamp_to_8u) {
        dst = cv::Mat(src.size(), CV_8UC1);
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                dst.at<uchar>(i, j) = src.at<uchar>(i, j);
            }
        }
    }
    else {
        dst = cv::Mat(src.size(), CV_64F);
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                dst.at<double>(i, j) = static_cast<double>(src.at<uchar>(i, j));
            }
        }
    }

 
    for (int i = padH; i < src.rows - padH; ++i) {
        for (int j = padW; j < src.cols - padW; ++j) {
            double sum = 0.0;

            for (int ki = 0; ki < kh; ++ki) {
                for (int kj = 0; kj < kw; ++kj) {
                    const int si = i + ki - padH;
                    const int sj = j + kj - padW;
                    sum += src.at<uchar>(si, sj) * kernel[ki][kj];
                }
            }

            if (clamp_to_8u) {
                if (sum < 0.0) sum = 0.0;
                if (sum > 255.0) sum = 255.0;
                dst.at<uchar>(i, j) = static_cast<uchar>(sum);
            }
            else {
                dst.at<double>(i, j) = sum;
            }
        }
    }
    return dst;
}