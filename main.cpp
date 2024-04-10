#include <iostream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "image.hpp"

int main(int argc, char *argv[]){

    image img(argv[1]);
    cv::Mat imagem, imagem_tc, imagem_x, imagem_y, mag;
    cv::Mat imagem_xs, imagem_ys;
    // cv::Mat gradient_angle_degrees;
    // bool angleInDegrees = true;

    // int fator = atoi(argv[2]);

    if(img.getPixels().empty()){
        std::cout << "Não foi possível ler o arquivo: " << argv[1] << std::endl;
        return 1;
    }

    imagem = img.getPixels();

    cv::cvtColor(imagem, imagem_tc, cv::COLOR_BGR2GRAY);
    // std::cout << "converteu o arquivo pra tons de cinza\n";
    // //     Altere os pixels da imagem
    // for (int y = 0; y < imagem.rows; ++y) {
    //     for (int x = 0; x < imagem.cols; ++x) {
    //         // Acesse o pixel na posição (x, y)
    //         cv::Vec3b& pixel = imagem.at<cv::Vec3b>(y, x);
            
    // //         // Modifique os valores dos canais de cor (BGR)
    // //         // Por exemplo, faça o pixel ser azul
    //         pixel[0] *= fator;
    //         pixel[1] *= fator;
    //         pixel[2] *= fator;
    //     }
    // }

    cv::imshow("Janela de visualização", imagem);
    cv::Sobel(imagem_tc, imagem_x, imagem_tc.depth(), 1, 0, 3);
    cv::resize(imagem_x, imagem_xs, cv::Size(imagem_x.rows, imagem_x.cols), 0, 0);
    cv::imshow("Janela de visualização - Sobel x", imagem_xs);

    cv::Sobel(imagem_tc, imagem_y, imagem_tc.depth(), 0, 1, 3);
    cv::resize(imagem_y, imagem_ys, cv::Size(imagem_y.rows, imagem_y.cols), 0, 0);
    cv::imshow("Janela de visualização - Sobel y", imagem_ys);

    // cv::phase(imagem_x, imagem_y, gradient_angle_degrees, angleInDegrees);
    // cv::imshow("Janela de visualização - gradiente", gradient_angle_degrees);

    std::cout << "imagem x size == " << imagem_x.size << std::endl;
    std::cout << "imagem y size == " << imagem_y.size << std::endl;

    // cv::Mat mag(imagem_x.size(), imagem_x.type());

    imagem_tc = abs(imagem_xs) + abs(imagem_ys);
    cv::imshow("Janela de visualização - Sobel |x + y|", imagem_tc);

    cv::magnitude(imagem_xs, imagem_ys, mag);
    cv::Mat res;
    cv::hconcat(imagem_xs,imagem_ys, res);
    cv::hconcat(res, mag, res);
    
    cv::imshow("Janela de visualização - magnitude de sobelx e sobely", res);

    // std::cout << imagem_tc << std::endl;
    // std::cout << imagem << std::endl;
    int k = cv::waitKey(0);

    return 0;
}