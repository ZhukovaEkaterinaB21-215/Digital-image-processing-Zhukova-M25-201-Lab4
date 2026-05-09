#include "kernels.h"


constexpr double PI = 3.14159265358979323846;

std::vector<std::vector<double>> createGaussianKernel(int ksize, double sigma) {
    if (ksize <= 0 || ksize % 2 == 0) {
        throw std::invalid_argument("Размер ядра должен быть положительным нечётным числом");
    }
    std::vector<std::vector<double>> k(ksize, std::vector<double>(ksize));
    double sum = 0.0;
    const int half = ksize / 2;
    const double sigma2 = sigma * sigma;

    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            double val = std::exp(-(x * x + y * y) / (2.0 * sigma2));
            k[y + half][x + half] = val;
            sum += val;
        }
    }
    for (auto& row : k) {
        for (auto& v : row) {
            v /= sum;
        }
    }
    return k;
}

std::vector<std::vector<double>> createBoxKernel(int ksize) {
    const double val = 1.0 / (ksize * ksize);
    return std::vector<std::vector<double>>(ksize, std::vector<double>(ksize, val));
}

std::vector<std::vector<double>> createLoGKernel(int ksize, double sigma) {
    if (ksize <= 0 || ksize % 2 == 0) {
        throw std::invalid_argument("Размер ядра должен быть положительным нечётным числом");
    }
    std::vector<std::vector<double>> k(ksize, std::vector<double>(ksize));
    double sum = 0.0;
    const int half = ksize / 2;
    const double sigma2 = sigma * sigma;

    const double factor = -1.0 / (PI * sigma2 * sigma2);

    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            const double r2 = x * x + y * y;
            
            double val = factor * (1.0 - r2 / (2.0 * sigma2)) * std::exp(-r2 / (2.0 * sigma2));
            k[y + half][x + half] = val;
            sum += val;
        }
    }
   
    const double mean = sum / (ksize * ksize);
    for (auto& row : k) {
        for (auto& v : row) {
            v -= mean;
        }
    }
    return k;
}