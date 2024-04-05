#include <iostream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "image.hpp"

int main(int argc, char *argv[]){

    image img(argv[1]);
    cv::Mat imagem, imagem_tc, imagem_x, imagem_y;
    // cv::Mat gradient_angle_degrees;
    // bool angleInDegrees = true;

    // int fator = atoi(argv[2]);

    if(img.getPixels().empty()){
        std::cout << "Não foi possível ler o arquivo: " << argv[1] << std::endl;
        return 1;
    }

    imagem = img.getPixels();

    cv::cvtColor(imagem, imagem_tc, cv::COLOR_BGR2GRAY);
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
    cv::Sobel(imagem_tc, imagem_x, 0, 1, 0);
    cv::imshow("Janela de visualização - Sobel x", imagem_x);

    cv::Sobel(imagem_tc, imagem_y, 0, 0, 1);
    cv::imshow("Janela de visualização - Sobel y", imagem_y);

    // cv::phase(imagem_x, imagem_y, gradient_angle_degrees, angleInDegrees);
    // cv::imshow("Janela de visualização - gradiente", gradient_angle_degrees);

    imagem_tc = abs(imagem_x) + abs(imagem_y);
    cv::imshow("Janela de visualização - Sobel |x + y|", imagem_tc);

    // std::cout << imagem_tc << std::endl;
    // std::cout << imagem << std::endl;
    int k = cv::waitKey(0);

    return 0;
}