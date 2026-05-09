#include "convolution.h"
#include "highpass_log.h"
#include "lowpass_noise.h"
#include "sharpening.h"
#include "threshold_avg.h"
#include "zero_crossing.h"
#include "integral_image.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#include "tests.h"

int main(int argc, char** argv) {
    #ifdef _WIN32
        system("chcp 65001 >nul");
    #endif
    
    runTests();

    srand(42);

    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение.\n";
        return 1;
    }

    
    cv::Mat gauss_noise = addGaussianNoise(src, 25.0);
    cv::Mat sp_noise = addSaltPepperNoise(src, 0.05);


    
    cv::Mat gauss_lpf_gauss = applyGaussianLPF(gauss_noise, 5, 1.2);
    cv::Mat gauss_lpf_sp = applyGaussianLPF(sp_noise, 5, 1.2);
    
    cv::Mat box_lpf_gauss = applyAveragingLPF(gauss_noise, 5);
    cv::Mat box_lpf_sp = applyAveragingLPF(sp_noise, 5);

    cv::Mat thresh_gauss = applyAveragingWithThreshold(gauss_noise, 5, 15.0);
    cv::Mat thresh_sp = applyAveragingWithThreshold(sp_noise, 5, 40.0);

    
    cv::Mat lap_map = applyLaplacian(src);
    cv::Mat log_map = applyLoG(src, 1.5); 

    cv::Mat zc_lap = detectZeroCrossings(lap_map);
    cv::Mat zc_log = detectZeroCrossings(log_map);

    cv::Mat sharpened = applySharpening(src, 1.5, 5, 1.2, 3.0);

    cv::Mat avg_integral = applyAveragingIntegral(src, 5);
  
    cv::imwrite("01_original.png", src);
    cv::imwrite("02_noisy_gauss.png", gauss_noise);
    cv::imwrite("03_noisy_sp.png", sp_noise);

    cv::imwrite("04_gauss_lpf_gauss.png", gauss_lpf_gauss);
    cv::imwrite("05_box_lpf_gauss.png", box_lpf_gauss);
    cv::imwrite("06_gauss_lpf_sp.png", gauss_lpf_sp);
    cv::imwrite("07_box_lpf_sp.png", box_lpf_sp);

    cv::imwrite("08_thresh_gauss.png", thresh_gauss);
    cv::imwrite("09_thresh_sp.png", thresh_sp);

    cv::Mat lap_vis(lap_map.size(), CV_8UC1);
    
    for (int i = 0; i < lap_map.rows; ++i) {
        for (int j = 0; j < lap_map.cols; ++j) {
            double val = lap_map.at<double>(i, j) + 128.0;
            if (val < 0.0) val = 0.0;
            if (val > 255.0) val = 255.0;
            lap_vis.at<uchar>(i, j) = static_cast<uchar>(val);
        }
    }

    cv::Mat log_vis(log_map.size(), CV_8UC1);
    for (int i = 0; i < log_map.rows; ++i) {
        for (int j = 0; j < log_map.cols; ++j) {
            double val = log_map.at<double>(i, j) + 128.0;
            if (val < 0.0) val = 0.0;
            if (val > 255.0) val = 255.0;
            log_vis.at<uchar>(i, j) = static_cast<uchar>(val);
        }
    }
    cv::imwrite("10_laplacian_response.png", lap_vis);
    cv::imwrite("11_log_response.png", log_vis);

    cv::imwrite("12_zc_laplacian.png", zc_lap);
    cv::imwrite("13_zc_log.png", zc_log);
    cv::imwrite("14_sharpened.png", sharpened);

    cv::imwrite("15_avg_integral.png", avg_integral);
    

    cv::waitKey(0);
    return 0;
}