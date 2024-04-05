#ifndef image_hpp
#define image_hpp

#include <stdint.h>
#include <cstdlib>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

class image
{
private:
    // estruturas de leitura dos valores BMP
    // mas vai ser sempre BMP?
    cv::Mat pixels; // matriz de pixels
    int width;      // largura da img
    int height;     // altura da img
    
public:
    image(std::string file_name);
    cv::Mat getPixels();
    ~image();
};

// image::image(/* args */)
// {
//     pixels = (int**)malloc(height * sizeof(int*));
//     for(int i = 0; i < height; i++){
//         pixels[i] = (int*)malloc(width * sizeof(int));
//     }
// }


// image::~image()
// {
//     // desalocando a matriz de pixels
//     for(int i = 0; i < height; i++){
//         free(pixels[i]);
//     }
//     free(pixels);
// }

#endif