#include "image.hpp"

image::image(std::string file_name)
{
    // pixels = (int**)malloc(height * sizeof(int*));
    // for(int i = 0; i < height; i++){
    //     pixels[i] = (int*)malloc(width * sizeof(int));
    // }

    pixels = cv::imread(file_name, 1);
}

cv::Mat image::getPixels(){
    return this->pixels;
}

image::~image()
{
    // desalocando a matriz de pixels
    // for(int i = 0; i < height; i++){
    //     free(pixels[i]);
    // }
    // free(pixels);
    pixels.release();
}