#include <iostream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <stack>
#include <list>
#include <vector>

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

struct point{
    double x, y;
};


// funcao p calcular a distancia entre os pontos
double distance(point p1, double inclinicacao, double interceptacao){
    return std::abs(inclinicacao * p1.x - p1.y + interceptacao) / 
            std::sqrt(inclinicacao * inclinicacao + 1);
}

// calcular parametros de reta
void calcLineParams(point topFe, point topAb, double& inclinacao, double &interceptacao){
    inclinacao = (topAb.y - topFe.y) / (topAb.x - topFe.x);
    interceptacao = topFe.y - inclinacao * topFe.x;
}

// 1) P é um conjunto ordenado de pontos. A e B são os pontos de partida
// 2) definimos o limiar T e duas pilhas vazias (Ab e Fe)
// 3) se P define uma curva fechada, empilhamos B em Ab e Fe e A em Ab.
//  se a curva for aberta colocamos A em Ab e B em Fe.
// 4) calculamos os parâmetros da reta que passa pelos vértices no topo de Fe
// e no topo de Ab
// 5) para todos os pontos, entre aqueles vértices obtidos em (4), calculamos
// suas distâncias em relação à reta obtida em (4). Selecionamos o ponto Vmax,
// com distância Dmax
// 6) se Dmax > T, empilhamos o vertice Vmax em Ab e retornamos a (4)
// 7) senão, removemos o vértice no topo de Ab empilhando-o em Fe
// 8) se a pilha Ab não estiver vaiz, retornamos a (4)
// 9) caso contrário, saímos. Os vértices em Fe definem a aproximação poligonal 
// do conjunto de pontos P.
void regionalProcessing(std::vector<point> points, double T, bool closed){
    std::stack<point> aberta, fechada;
    point A, B;

    // escolher onde empilhar A e B, de acordo com o tipo de curva (como saber?)
    if(closed){ // p curva fechada
        aberta.push(B);
        fechada.push(B);
        aberta.push(A);
    }
    else{ // p curva aberta
        aberta.push(A);
        fechada.push(B);
    }

    while(!aberta.empty()){ // enquanto a lista de abertos não estiver vazia
        point topFe(fechada.top()), topAb(aberta.top());
        double incl, intercep;
        calcLineParams(topFe, topAb, incl, intercep);
        point vmax;
        double dmax = 0.0;

        for(const auto& it : points){
            if(dmax < distance(it, incl, intercep)){
                vmax = it;
                dmax = distance(it, incl, intercep);
            }
        }

        if(dmax > T){
            aberta.push(vmax);
            continue;
        }
        else{
            aberta.pop();
            fechada.push(vmax);
        }
    }
    while(!fechada.empty()){
        std::cout << fechada.top().x << ", " << fechada.top().y << std::endl;;
        fechada.pop();
    }
}


int main(int argc, char *argv[]){

    cv::Mat imagem, correcao;
    int opc = -1;
    std::string nome_imagem;

    std::cout << "Digite o nome do arquivo que você gostaria de processar: \n";
    std::cin >> nome_imagem;
    imagem = cv::imread(nome_imagem, cv::IMREAD_GRAYSCALE);
    if(imagem.empty()){
        std::cerr << "Erro ao carregar a imagem.\n";
        return -1;
    }

    std::cout << "Qual tipo de processamento você quer realizar?\n 1 - Processamento Local\n 2 - Processamento Regional\n 3 - Processamento Global\n";
    std::cin >> opc;
    std::system("clear");

    switch(opc){
        case 1:{
            double Tm = 0.3, Ta = 45, A = 90;
            int K = 5;

            // obtendo os parâmetros pela entrada do usuário
            std::cout << "Digite o limiar Tm: \n";
            std::cin >> Tm;
            std::cout << "Digite a direção angular (A): \n";
            std::cin >> A;
            std::cout << "Digite a faixa de direções aceitáveis (Ta) ao redor A: \n";
            std::cin >> Ta;
            std::cout << "Digite o limiar K para realizar a correção de falhas: \n";
            std::cin >> K;

            // aplicando as funções
            cv::Mat result_h(imagem.rows, imagem.cols, CV_8U), result_v(imagem.rows, imagem.cols, CV_8U);
            result_h = localProcessing(imagem, Tm, A, Ta);
            result_v = localProcessing(imagem, Tm, 180, Ta);

            cv::Mat result_or;
            cv::bitwise_or(result_h, result_v, result_or);
            cv::imshow("Operação de OR dos dois resultados", result_or);

            // realizando a correcao na vertical
            correcao = correction(result_or, K);
            cv::imshow("correcao", correcao);

            // realizando a correcao na vertical
            cv::Mat correcao2;
            cv::rotate(result_or, correcao2, cv::ROTATE_90_CLOCKWISE);
            correcao2 = correction(correcao2, K);
            cv::rotate(correcao2, correcao2, cv::ROTATE_90_COUNTERCLOCKWISE);
            cv::imshow("Correcao de lado", correcao2);

            cv::waitKey(0);
            break;
        }
        
        case 2:{
            int T = -1;

            // obtendo parâmetros do usuário
            std::cout << "Informe o limiar T (quanto menor o limiar, maior a precisão e vice-versa): \n";
            std::cin >> T;

            std::vector<point> P;

            // P.push_back(std::make_tuple(1, 4));
            // P.push_back(std::make_tuple(4, 8));
            
            // std::cout << "lista de pontos P\n";
            // for(const auto& tupla : P){
            //     std::cout << std::get<0>(tupla) << ", " << std::get<1>(tupla) << std::endl;
            // }


            break;
        }
        
        case 3:{
            // 1) obtenha uma imagem binária com as bordas da imagem
            // 2) defina como o plano p-theta será dividido (estrutura da matriz acumuladora)
            // 3) aplique a parametrização aos pontos da imagem das bordas, atualizando
            // a matriz acumuladora
            // 4) examine a matriz acumuladora em busca de células com valores elevados
            // 5) examine a relação (principalmente as de continuidade) entre os pixels
            // oriundos das células escolhidas em (4)

            std::cout << "Operação em desenvolvimento, tente novamente mais tarde.\n";
            break;
        }

        default:{
            std::cout << "Operação inválida\n";
            return -1;
        }
    }


    return 0;
}