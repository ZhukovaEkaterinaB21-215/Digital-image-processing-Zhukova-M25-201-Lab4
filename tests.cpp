#include "tests.h"
#include "convolution.h"
#include "kernels.h"
#include "lowpass_noise.h"
#include "threshold_avg.h"
#include "highpass_log.h"
#include "zero_crossing.h"
#include "sharpening.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <opencv2/opencv.hpp>


static void saveTestImage(const cv::Mat& img, const std::string& filename) {
    if (!img.empty()) {
        if (cv::imwrite(filename, img)) {
            std::cout << "  [SAVE] " << filename << "\n";
        }
        else {
            std::cerr << "  [WARN] Не удалось сохранить " << filename << "\n";
        }
    }
}


void runTests() {
    {
        auto gk = createGaussianKernel(5, 1.0);
        double g_sum = 0.0;
        for (const auto& row : gk) {
            for (double v : row) {
                g_sum += v;
            }
        }

        if (std::abs(g_sum - 1.0) >= 1e-6) {
            throw std::runtime_error("Сумма ядра Гаусса != 1.0\n");
        }
        if (std::abs(g_sum - 1.0) < 1e-6) {
            std::cout << "Сумма ядра Гаусса = 1.0, тест пройден\n";
        }


        auto lk = createLoGKernel(7, 1.5);
        double l_sum = 0.0;
        for (const auto& row : lk) {
            for (double v : row) {
                l_sum += v;
            }
        }
        if (std::abs(l_sum) >= 1e-5) {
            throw std::runtime_error("Сумма ядра LoG != 0\n");
        }
        if (std::abs(l_sum) < 1e-5) {
            std::cout << "Сумма ядра LoG = 0, тест пройден\n";
        }

    }


    {
        std::vector<std::vector<double>> delta = { {0,0,0},{0,1,0},{0,0,0} };
        cv::Mat test_img(128, 128, CV_8UC1);
        for (int i = 0; i < test_img.rows; ++i) {
            for (int j = 0; j < test_img.cols; ++j) {
                test_img.at<uchar>(i, j) = static_cast<uchar>(j * 255 / (test_img.cols - 1));
            }
        }
        cv::Mat id_res = convolve2D(test_img, delta, true);

        if (cv::countNonZero(test_img != id_res) != 0) {
            throw std::runtime_error("Свёртка с единичным ядром изменила изображение\n");
        }

        if (cv::countNonZero(test_img != id_res) == 0) {
            std::cout << "Свёртка с единичным ядром работает корректно.\n";
        }
    }
    
    
    {
        cv::Mat flat(128, 128, CV_8UC1, cv::Scalar(100));
        cv::Mat lap = applyLaplacian(flat);
        cv::Mat lap_vis(lap.size(), CV_8UC1);
        const double alpha = 1.0;
        const double beta = 128.0;

        for (int i = 0; i < lap.rows; ++i) {
            for (int j = 0; j < lap.cols; ++j) {
                double val = lap.at<double>(i, j);
                val = val * alpha + beta;

                if (val < 0.0) {
                    val = 0.0;
                }
                if (val > 255.0) {
                    val = 255.0;
                }
                lap_vis.at<uchar>(i, j) = static_cast<uchar>(val);
            }
        }

        cv::Mat zc = detectZeroCrossings(lap);
        

        if (cv::countNonZero(zc) != 0) {
            throw std::runtime_error("Есть ложные границы на сплошном изображении");
        }
        if (cv::countNonZero(zc) == 0) {
            std::cout << "Ложные границы на сплошном изображении отсутствуют\n";
        }
    }

    
    {
        std::cout << "Подавление адитивного гауссового шума Гауссовым ФНЧ...\n";
        cv::Mat clean(512, 512, CV_8UC1, cv::Scalar(120));
        saveTestImage(clean, "test4_clean_gauss.png");

        cv::Mat noisy = addGaussianNoise(clean, 35.0);
        saveTestImage(noisy, "test4_noisy_gauss_gauss.png");

        double var_noisy = calcVariance(noisy);
        std::cout << "  Дисперсия зашумлённого: " << var_noisy << "\n";

        cv::Mat denoised = applyGaussianLPF(noisy, 5, 1.5);
        saveTestImage(denoised, "test4_gauss_gauss_lpf_result.png");

        double var_denoised = calcVariance(denoised);
        std::cout << "  Дисперсия после Гауссова ФНЧ: " << var_denoised << "\n";

        if (var_denoised >= var_noisy * 0.5) {
            throw std::runtime_error("Гауссов ФНЧ не снизил дисперсию");
        }
        if (var_denoised < var_noisy * 0.5) {
            std::cout << "Гауссов ФНЧ эффективно подавляет гауссовый шум.\n";
        }
    }

    {
        std::cout << "Подавление импульсного шума Гауссовым ФНЧ...\n";
        cv::Mat clean(512, 512, CV_8UC1, cv::Scalar(120));

        cv::Mat noisy = addSaltPepperNoise(clean, 0.1);
        saveTestImage(noisy, "test4_noisy_gauss_sp.png");

        double var_noisy = calcVariance(noisy);
        std::cout << "  Дисперсия зашумлённого: " << var_noisy << "\n";

        cv::Mat denoised = applyGaussianLPF(noisy, 5, 1.5);
        saveTestImage(denoised, "test4_gauss_sp_lpf_result.png");

        double var_denoised = calcVariance(denoised);
        std::cout << "  Дисперсия после Гауссова ФНЧ: " << var_denoised << "\n";

        if (var_denoised >= var_noisy * 0.5) {
            throw std::runtime_error("Гауссов ФНЧ не снизил дисперсию");
        }
        if (var_denoised < var_noisy * 0.5) {
            std::cout << "Гауссов ФНЧ эффективно подавляет импульсный шум.\n";
        }
    }
    
    {
        std::cout << "\nПодавление гауссового шума Усредняющим ФНЧ...\n";
        cv::Mat clean(512, 512, CV_8UC1, cv::Scalar(120));
        saveTestImage(clean, "test5_clean_box.png");

        cv::Mat noisy = addGaussianNoise(clean, 35.0);
        saveTestImage(noisy, "test5_noisy_box_gauss.png");

        double var_noisy = calcVariance(noisy);
        std::cout << "  Дисперсия зашумлённого: " << var_noisy << "\n";

        cv::Mat denoised = applyAveragingLPF(noisy, 5);
        saveTestImage(denoised, "test5_box_gauss_lpf_result.png");

        double var_denoised = calcVariance(denoised);
        std::cout << "  Дисперсия после Усредняющего ФНЧ: " << var_denoised << "\n";

        if (var_denoised >= var_noisy * 0.6) {
            throw std::runtime_error("Усредняющий ФНЧ не снизил дисперсию");
        }
        if (var_denoised < var_noisy * 0.6) {
            std::cout << "Усредняющий ФНЧ эффективно подавляет гауссов шум.\n";
        }
    }

    {
        std::cout << "\nПодавление импульсного шума Усредняющим ФНЧ...\n";
        cv::Mat clean(512, 512, CV_8UC1, cv::Scalar(120));

        cv::Mat noisy = addSaltPepperNoise(clean, 0.1);
        saveTestImage(noisy, "test5_noisy_box_sp.png");

        double var_noisy = calcVariance(noisy);
        std::cout << "  Дисперсия зашумлённого: " << var_noisy << "\n";

        cv::Mat denoised = applyAveragingLPF(noisy, 5);
        saveTestImage(denoised, "test5_box_sp_lpf_result.png");

        double var_denoised = calcVariance(denoised);
        std::cout << "  Дисперсия после Усредняющего ФНЧ: " << var_denoised << "\n";

        if (var_denoised >= var_noisy * 0.6) {
            throw std::runtime_error("Усредняющий ФНЧ не снизил дисперсию");
        }
        if (var_denoised < var_noisy * 0.6) {
            std::cout << "Усредняющий ФНЧ эффективно подавляет импульсный шум.\n";
        }
    }
    
    {
        std::cout << "\nПроверка усредняющего фильтра с порогом на импульсном шуме...\n";
        cv::Mat clean = cv::Mat(512, 512, CV_8UC1, cv::Scalar(150));
        saveTestImage(clean, "test6_clean_sp.png");

        cv::Mat sp_noise = addSaltPepperNoise(clean, 0.1);
        saveTestImage(sp_noise, "test6_noisy_sp.png");

        cv::Mat filtered = applyAveragingWithThreshold(sp_noise, 5, 50.0);
        saveTestImage(filtered, "test6_filtered_sp.png");


        int outliers_before = 0;
        for (int i = 0; i < sp_noise.rows; ++i) {
            for (int j = 0; j < sp_noise.cols; ++j) {
                uchar val = sp_noise.at<uchar>(i, j);
                if (val == 0 || val == 255) {
                    outliers_before++;
                }
            }
        }

        int outliers_after= 0;
        for (int i = 0; i < filtered.rows; ++i) {
            for (int j = 0; j < filtered.cols; ++j) {
                uchar val = filtered.at<uchar>(i, j);
                if (val == 0 || val == 255) {
                    outliers_after++;
                }
            }
        }

        std::cout << "Кол-во выбросов до: " << outliers_before << ", после: " << outliers_after << "\n";

        if (outliers_after >= outliers_before * 0.3) {
            throw std::runtime_error("Адаптивный фильтр не убрал выбросы");
        }
        if (outliers_after < outliers_before * 0.3) {
            std::cout << "Адаптивный фильтр подавляет импульсный шум.\n";
        }
    }
    
    
    {
        cv::Mat gradient(512, 512, CV_8UC1);

        for (int i = 0; i < gradient.rows; ++i) {
            uchar val = static_cast<uchar>(i * 8);
            for (int j = 0; j < gradient.cols; ++j) {
                gradient.at<uchar>(i, j) = val;
            }
        }

        saveTestImage(gradient, "test7_gradient.png");

        double var_orig = calcVariance(gradient);
        std::cout << "  Дисперсия зашумлённого: " << var_orig << "\n";

        cv::Mat sharpened = applySharpening(gradient, 2.5, 5, 1.2, 3.0);
        saveTestImage(sharpened, "test7_sharpened.png");

        
        double var_sharp = calcVariance(sharpened);
        std::cout << "  Дисперсия после повышения резкости: " << var_sharp << "\n";

        if (sharpened.empty() || sharpened.type() != CV_8UC1 || var_sharp < var_orig * 0.9) {
            throw std::runtime_error("Sharpening работает некорректно");
        }
        if (var_sharp >= var_orig * 0.9) {
            std::cout << "Повышение резкости работает корректно.\n";
        }
    }
    

    {        
        cv::Mat edge_img(512, 512, CV_8UC1);
        for (int i = 0; i < edge_img.rows; ++i) {
            for (int j = 0; j < edge_img.cols; ++j) {
                edge_img.at<uchar>(i, j) = (j >= 256) ? 255 : 0;
            }
        }
        saveTestImage(edge_img, "test8_edge_input.png");

        cv::Mat log_response = applyLoG(edge_img, 1.0);
        
        cv::Mat log_vis(log_response.size(), CV_8UC1);
        for (int i = 0; i < log_response.rows; ++i) {
            for (int j = 0; j < log_response.cols; ++j) {
                double val = log_response.at<double>(i, j) + 128.0; 
                if (val < 0.0) {
                    val = 0.0;
                }
                if (val > 255.0) {
                    val = 255.0;
                }
                log_vis.at<uchar>(i, j) = static_cast<uchar>(val);
            }
        }
        saveTestImage(log_vis, "test8_log_response.png");

        cv::Mat zc = detectZeroCrossings(log_response);
        saveTestImage(zc, "test8_zc_log.png");

        int boundary_pixels = 0;
        for (int i = 1; i < 511; ++i) {
            if (zc.at<uchar>(i, 256) == 255) boundary_pixels++;
        }
        std::cout << "  Найдено пикселей границы: " << boundary_pixels << "\n";

        if (boundary_pixels < 10) {
            throw std::runtime_error("Граница не детектирована");
        }
        if (boundary_pixels >= 10) {
            std::cout << "LoG + Zero-Crossings детектирует границы.\n";
        }
    }
    
    {
        cv::Mat clean(512, 512, CV_8UC1, cv::Scalar(120));
        cv::Mat noisy = addGaussianNoise(clean, 30.0);

        cv::Mat gauss_result = applyGaussianLPF(noisy, 5, 1.2);
        cv::Mat box_result = applyAveragingLPF(noisy, 5);

        saveTestImage(gauss_result, "test_gauss_lpf_compare.png");
        saveTestImage(box_result, "test_box_lpf_compare.png");

        double var_gauss = calcVariance(gauss_result);
        double var_box = calcVariance(box_result);
        double var_noisy = calcVariance(noisy);

        std::cout << "Дисперсия: шум=" << var_noisy
            << ", Гаусс=" << var_gauss
            << ", Box=" << var_box << "\n";

        if (var_gauss >= var_box) {
            std::cout << "Дисперсия Гауссова ФНЧ не меньше усредняющего (допустимо)\n";
        }
    }

}