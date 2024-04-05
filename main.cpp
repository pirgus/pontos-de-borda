#include <iostream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "image.hpp"

int main(int argc, char *argv[]){

    image img(argv[1]);
    cv::Mat imagem;
    int fator = atoi(argv[2]);

    if(img.getPixels().empty()){
        std::cout << "Não foi possível ler o arquivo: " << argv[1] << std::endl;
        return 1;
    }

    imagem = img.getPixels();

        // Altere os pixels da imagem
    for (int y = 0; y < imagem.rows; ++y) {
        for (int x = 0; x < imagem.cols; ++x) {
            // Acesse o pixel na posição (x, y)
            cv::Vec3b& pixel = imagem.at<cv::Vec3b>(y, x);
            
            // Modifique os valores dos canais de cor (BGR)
            // Por exemplo, faça o pixel ser azul
            pixel[0] *= fator;
            pixel[1] *= fator;
            pixel[2] *= fator;
        }
    }

    cv::imshow("Janela de visualização", imagem);
    std::cout << imagem << std::endl;
    int k = cv::waitKey(0);

    return 0;
}