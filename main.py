import cv2 as cv
import numpy as np
import math as m
from scipy import spatial


img_path = input("Informe o nome do arquivo a ser utilizado: ")
path = "./imagens/" + img_path
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
    cv.imshow('imagem', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print('Erro ao carregar a imagem.')
    exit(-1)

dimensions = img.shape
rows, cols = dimensions

def localProcessing(original_image, Tm, A, Ta):
    sobel_x = cv.Sobel(original_image, cv.CV_64F, 1, 0)
    sobel_y = cv.Sobel(original_image, cv.CV_64F, 0, 1)


    magnitude, angle = cv.cartToPolar(sobel_x, sobel_y, angleInDegrees = True)


    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(magnitude)


    Tm = Tm * maxVal
    img_g = np.zeros((rows, cols))

    for i in range(0, rows):
        for j in range(0, cols):
            if(magnitude[i][j] > Tm):
                angulo = angle[i, j]
                if(abs(A - angulo) <= Ta or abs(A - angulo - 180) <= Ta):
                    img_g[i][j] = 255
                else:
                    img_g[i][j] = 0
            else:
                img_g[i][j] = 0

    cv.imshow("Imagem resultante", img_g)
    cv.waitKey(0)

    return img_g

def correction(original_image, K):
    dimensions = original_image.shape
    rows, cols = dimensions
    img_corrected = original_image

    for i in range(0, rows):
        for j in range(0, cols):
            if(original_image[i][j] == 255):
                count = 0
                k = j + 1
                while(k < cols - 1 and original_image[i][k] == 0):
                    count = count + 1
                    k = k + 1
                if(k < cols - 1 and original_image[i][k] == 255 and count <= K):
                    for fill in range(j + 1, k):
                        img_corrected[i][fill] = 255
                    
                j = k - 1

    return img_corrected


def distance(p1, p2, p3):

    if p1[0] == p2[0] and p1[1] == p2[1]:
        return 0

    x0, y0 = p1
    x1, y1 = p2
    x2, y2 = p3

    m = (y2 - y1) / (x2 - x1)

    c = -m * x1 + y1
    a = -m
    b = 1

    distancia = abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)
    return distancia
    # return abs(inclinicacao * p1[0] - p1[1] + interceptacao) / m.sqrt(inclinicacao * inclinicacao + 1)


# def calcLineParams(topFe, topAb):
#     if(topAb[0] - topFe[0] != 0):
#         inclinacao = (topAb[1] - topFe[1]) / (topAb[0] - topFe[0])
#         interceptacao = topFe[1] - inclinacao * topFe[0]
#     else:
#         inclinacao = float('inf')
#         interceptacao = float('inf')
    

    return inclinacao, interceptacao

def limiarize(original_image, limiar):
    sobel_x = cv.Sobel(original_image, cv.CV_64F, 1, 0)
    sobel_y = cv.Sobel(original_image, cv.CV_64F, 0, 1)

    magnitude, angle = cv.cartToPolar(sobel_x, sobel_y, angleInDegrees = True)

    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(magnitude)

    ret, bin_img = cv.threshold(magnitude, limiar * maxVal, 255, cv.THRESH_BINARY)

    bin_img = cv.morphologyEx(bin_img, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return bin_img

def ordenarPontos(points):
    cx, cy = points.mean(0)
    x, y = points.T
    angles = np.arctan2(x - cx, y - cy)
    indices = np.argsort(-angles)
    return points[indices]

# 1) P é um conjunto ordenado de pontos. A e B são os pontos de partida
# 2) definimos o limiar T e duas pilhas vazias (Ab e Fe)
# 3) se P define uma curva fechada, empilhamos B em Ab e Fe e A em Ab.
#  se a curva for aberta colocamos A em Ab e B em Fe.
# 4) calculamos os parâmetros da reta que passa pelos vértices no topo de Fe
# e no topo de Ab
# 5) para todos os pontos, entre aqueles vértices obtidos em (4), calculamos
# suas distâncias em relação à reta obtida em (4). Selecionamos o ponto Vmax,
# com distância Dmax
# 6) se Dmax > T, empilhamos o vertice Vmax em Ab e retornamos a (4)
# 7) senão, removemos o vértice no topo de Ab empilhando-o em Fe
# 8) se a pilha Ab não estiver vaiz, retornamos a (4)
# 9) caso contrário, saímos. Os vértices em Fe definem a aproximação poligonal 
#  do conjunto de pontos P.
def regionalProcessing(points, T, closed):
    final_img = np.ndarray((rows, cols))
    aberta = []
    fechada = []
    A = points[0]
    B = points[-1]
    
    if(closed):
        aberta.append(B)
        fechada.append(B)
        aberta.append(A)
    else:
        aberta.append(A)
        fechada.append(B)
    
    while(len(aberta) > 0):
        topFe = fechada[0]
        print("topfe = ", topFe)
        topAb = aberta[0]
        print("topab = ", topAb)
        # incl, intercep = calcLineParams(topFe, topAb)
        vmax = 0
        dmax = 0.0
        for i in points:
            if(dmax < distance(i, topFe, topAb)):
                vmax = i
                dmax = distance(i, topFe, topAb)
        
        if(dmax > T):
            aberta.append(vmax)
            continue
        else:
            aberta.pop()
            fechada.append(vmax)
    

    first_point = tuple(fechada[0])
    while(len(fechada) > 0):
        if(len(fechada) > 1):
            p1 = tuple(fechada[0])
            p2 = tuple(fechada[1])
            print("p2 = ", fechada[1])
        else:
            p1 = tuple(fechada[0])
            p2 = first_point
        final_img = cv.line(final_img, p1, p2, (255, 255, 255), 10)
        # print(fechada[0][0], ", ", fechada[0][1])
        fechada.pop()

    cv.imshow("Imagem contornada", final_img)

    

def globalProcessing():
    pass


# cv.imshow("Imagem escolhida (tons de cinza)", img)
# cv.waitKey(0)

print("Qual processamento você quer realizar? \n 1 - Processamento Local \n 2 - Processamento Regional \n 3 - Processamento Global")
choice = int(input())

if(choice == 1):
    Tm = float(input("Informe o limiar Tm: "))
    A = float(input("Informe a direção angular (A): "))
    Ta = float(input("Informe a faixa de direções aceitáveis (Ta): "))
    K = int(input("Informe o limiar (K) para correção de falhas: "))

    img_h = localProcessing(img, Tm, A, Ta)
    img_r = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    img_v = localProcessing(img_r, Tm, A, Ta)
    img_v = cv.rotate(img_v, cv.ROTATE_90_COUNTERCLOCKWISE)
    cv.imshow("img_v rotacionada", img_v)
    cv.waitKey(0)

    img_or = cv.bitwise_or(img_h, img_v)
    cv.imshow("imagem depois do OR", img_or)
    cv.waitKey(0)

    corrected = correction(img_or, K)
    cv.imshow("correcao", corrected)
    cv.waitKey(0)

    img_vert = cv.rotate(img_or, cv.ROTATE_90_CLOCKWISE)
    corrected2 = correction(img_vert, K)
    corrected2 = cv.rotate(corrected2, cv.ROTATE_90_COUNTERCLOCKWISE)

    result = cv.bitwise_or(corrected, corrected2)
    cv.imshow("correcao", result)
    cv.waitKey(0)


elif(choice == 2):
    binary = float(input("Digite a % a ser limiarizada da magnitude da imagem: "))
    T = float(input("Informe o limiar T: "))
    limit = float(input("Informe o limite de distancia para curvas fechadas: "))
    points = []
    bin_img = limiarize(img, binary)

    for i in range(0, rows):
        for j in range(0, cols):
            if(bin_img[i][j] == 255):
                points.append((i, j))
            
    points = np.array(points)

    points_ord = ordenarPontos(points)
    dist_mat = spatial.distance_matrix(points_ord, points_ord)
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    dist = m.dist(points[i], points[j])
    if(dist <= limit):
        closed = True
    else:
        closed = False

    regionalProcessing(points, T, closed)

elif(choice == 3):
    globalProcessing()
else:
    print("Operação inválida.")
