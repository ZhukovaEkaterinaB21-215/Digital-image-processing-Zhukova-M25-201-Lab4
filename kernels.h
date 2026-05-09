#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>

std::vector<std::vector<double>> createGaussianKernel(int ksize, double sigma);

std::vector<std::vector<double>> createBoxKernel(int ksize);

std::vector<std::vector<double>> createLoGKernel(int ksize, double sigma);