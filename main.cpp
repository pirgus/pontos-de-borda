#include <iostream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "image.hpp"

int main(int argc, char *argv[]){

    cv::Mat imagem, sobel_x, sobel_y, direction, mag, angle;
    imagem = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    double Tm = 0.3, Ta = 45, A = 90;

    cv::imwrite("piramidetc.jpg", imagem);

    if(imagem.empty()){
        std::cerr << "Erro ao carregar a imagem.\n";
        return -1;
    }


    cv::Sobel(imagem, sobel_x, CV_8U, 1, 0);
    cv::Sobel(imagem, sobel_y, CV_8U, 0, 1);

    cv::Mat sobel_x_float, sobel_y_float;
    sobel_x.convertTo(sobel_x_float, CV_64F);
    sobel_y.convertTo(sobel_y_float, CV_64F);

    cv::cartToPolar(sobel_x_float, sobel_y_float, mag, angle);   

    cv::Mat imagem_g(imagem.rows, imagem.cols, CV_64F);

    double valorMaximo;
    cv::Point posicaoMaxima;
    cv::minMaxLoc(mag, nullptr, &valorMaximo, nullptr, &posicaoMaxima);

    Tm *= valorMaximo;

    for(int i = 0; i < imagem.rows; i++){
        for(int j = 0; j < imagem.cols; j++){
            if(mag.at<uchar>(i,j) > Tm && (angle.at<uchar>(i,j) == A + Ta || angle.at<uchar>(i,j) == A - Ta))
                imagem_g.at<uchar>(i, j) = 255;
            else
                imagem_g.at<uchar>(i, j) = 0;
        }
    }

    mag.convertTo(mag, CV_8U);
    angle.convertTo(angle, CV_8U);

    cv::imshow("Imagem Original", imagem);
    cv::imshow("Derivada em X", sobel_x);
    cv::imshow("Derivada em Y", sobel_y);
    cv::imshow("Magnitude das Derivadas", mag);
    cv::imshow("Ângulo de Fase", angle);
    cv::imshow("g(x,y)", imagem_g);
    std::cout << imagem_g << std::endl;

    cv::waitKey(0);

    return 0;
}