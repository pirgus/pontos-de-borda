#include <iostream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

cv::Mat localProcessing(cv::Mat original_image, double Tm, double A, double Ta){
    cv::Mat sobel_x, sobel_y, direction, mag, angle;

    cv::Sobel(original_image, sobel_x, CV_8U, 1, 0);
    cv::Sobel(original_image, sobel_y, CV_8U, 0, 1);

    cv::Mat sobel_x_float, sobel_y_float;
    sobel_x.convertTo(sobel_x_float, CV_64F);
    sobel_y.convertTo(sobel_y_float, CV_64F);

    cv::cartToPolar(sobel_x_float, sobel_y_float, mag, angle, true);   

    cv::Mat imagem_g(original_image.rows, original_image.cols, CV_8U);
    cv::Mat angle_8u(original_image.rows, original_image.cols, CV_8U), mag_8u(original_image.rows, original_image.cols, CV_8U);

    double valorMaximo;
    cv::Point posicaoMaxima;
    cv::minMaxLoc(mag, nullptr, &valorMaximo, nullptr, &posicaoMaxima);

    // std::cout << "valormaximo = " << valorMaximo << std::endl;

    mag.convertTo(mag_8u, CV_8U);
    angle.convertTo(angle_8u, CV_8U);

    Tm *= valorMaximo;

    for(int i = 0; i < original_image.rows; ++i){
        for(int j = 0; j < original_image.cols; ++j){
            if(mag.at<double>(i,j) > Tm){
                double angulo = angle.at<double>(i, j);
                // std::cout << "angulo[" << i << ", " << j << "] = " << angulo << std::endl;
                // double diff_angle = std::abs(angulo - A);
                if(std::abs(A - angulo) <= Ta || std::abs(A - angulo - 180) <= Ta)
                    imagem_g.at<uchar>(i, j) = 255;
                else
                    imagem_g.at<uchar>(i,j) = 0;
            }
            else{
                imagem_g.at<uchar>(i,j) = 0;
            }
        }
    }

    // cv::imshow("Imagem Original", original_image);
    // cv::imshow("Derivada em X", sobel_x);
    // cv::imshow("Derivada em Y", sobel_y);
    // cv::imshow("Magnitude das Derivadas", mag_8u);
    // cv::imshow("Ângulo de Fase", angle);
    cv::imshow("g(x,y)", imagem_g);
    // std::cout << imagem_g << std::endl;

    cv::waitKey(0);

    return imagem_g;

}

cv::Mat correction(cv::Mat image, int K){

    cv::Mat corrected_image = image;

    for(int i = 1; i < image.rows - 1; i++){
        for(int j = 1; j < image.cols - 1; j++){
            if(image.at<uchar>(i, j) == 255){
                int count = 0;
                int k = j + 1;
                while(k < image.cols -1 && image.at<uchar>(i, k) == 0){
                    count++;
                    k++;
                }
                if(k < image.cols -1 && image.at<uchar>(i, k) == 255 && count <= K){
                    for(int fill = j + 1; fill < k; fill++){
                        corrected_image.at<uchar>(i, fill) = 255;
                    }
                }
                j = k - 1;
            }
        }
    }
    return corrected_image;
}


int main(int argc, char *argv[]){

    cv::Mat imagem, sobel_x, sobel_y, direction, mag, angle, correcao;
    imagem = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    int K = 5;

    double Tm = 0.3, Ta = 45, A = 90;

    cv::imwrite("piramidetc.jpg", imagem);

    if(imagem.empty()){
        std::cerr << "Erro ao carregar a imagem.\n";
        return -1;
    }

    cv::Mat result_h(imagem.rows, imagem.cols, CV_8U), result_v(imagem.rows, imagem.cols, CV_8U);
    result_h = localProcessing(imagem, Tm, A, Ta);
    result_v = localProcessing(imagem, Tm, 180, Ta);

    cv::Mat result_or;
    cv::bitwise_or(result_h, result_v, result_or);
    cv::imshow("Operação de OR dos dois resultados", result_or);
    // std::cout << result_or << std::endl;

    correcao = correction(result_or, K);
    cv::imshow("correcao", correcao);

    cv::rotate(result_or, result_or, cv::ROTATE_90_CLOCKWISE);
    cv::Mat correcao2 = correction(result_or, K);

    cv::rotate(correcao2, correcao2, cv::ROTATE_90_COUNTERCLOCKWISE);

    cv::imshow("Correcao de lado", correcao2);

    cv::waitKey(0);

    return 0;
}