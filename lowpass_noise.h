#pragma once
#include <opencv2/core.hpp>
#include <cstdlib>  
#include <cmath>

cv::Mat applyGaussianLPF(const cv::Mat& src, int ksize, double sigma);

cv::Mat applyAveragingLPF(const cv::Mat& src, int ksize);

cv::Mat addGaussianNoise(const cv::Mat& src, double sigma);

cv::Mat addSaltPepperNoise(const cv::Mat& src, double density);

double calcVariance(const cv::Mat& img);