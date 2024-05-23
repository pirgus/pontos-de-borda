import cv2 as cv
import numpy as np
import math as m
from scipy import spatial
import matplotlib.pyplot as plt
from functools import cmp_to_key

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

def localProcessing(original_image, Tm, A, Ta, K, rows, cols):
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


    img_g = correction(img_g, K)

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

def calcLineParams(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    
    if x2 - x1 != 0:
        inclinacao = (y2 - y1) / (x2 - x1)
    else:
        inclinacao = np.inf
    
    interceptacao = y1 - inclinacao * x1
    return inclinacao, interceptacao

def distance(point, inclinacao, interceptacao):
    x0, y0 = point
    
    if np.isinf(inclinacao):
        return abs(x0 - interceptacao)
    
    try:
        distancia = abs(inclinacao * x0 - y0 + interceptacao) / np.sqrt(inclinacao**2 + 1)
    except ZeroDivisionError:
        distancia = abs(y0 - interceptacao)
    
    return distancia

def is_between(point, A, B):
    # Verifica se o ponto está entre os vértices A e B no sentido horário
    # Calcula os vetores AB e AP
    vector_AB = [B[0] - A[0], B[1] - A[1]]
    vector_AP = [point[0] - A[0], point[1] - A[1]]

    # Calcula o produto vetorial entre AB e AP
    cross_product = vector_AB[0] * vector_AP[1] - vector_AB[1] * vector_AP[0]

    # Se o produto vetorial for positivo, o ponto está no sentido horário
    return cross_product > 0

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
# 8) se a pilha Ab não estiver vazia, retornamos a (4)
# 9) caso contrário, saímos. Os vértices em Fe definem a aproximação poligonal 
#  do conjunto de pontos P.

def regionalProcessing(points, A, B, closed): 
    ABERTA = []
    FECHADA = []

    if(closed):
        ABERTA.insert(0, tuple(B))
        FECHADA.insert(0, tuple(B))
        ABERTA.insert(0, tuple(A))
    else:
        ABERTA.insert(0, tuple(A))
        FECHADA.insert(0, tuple(B))

    while(True):
        inclinacao, interceptacao = calcLineParams(FECHADA[0], ABERTA[0])
        dmax = 0.0
        vmax = None
        for point in points:
            if is_between(point, FECHADA[0], ABERTA[0]):
                # print("point = ", point)
                d = distance(point, inclinacao, interceptacao)
                if d > dmax:
                    dmax = d
                    vmax = point
        # print("dmax = ", dmax)
        if(dmax > T):
            ABERTA.insert(0, tuple(vmax))
        else:
            FECHADA.insert(0, ABERTA.pop(0))
            if not ABERTA:
                return FECHADA
    
def limiarize(original_image, limiar):
    sobel_x = cv.Sobel(original_image, cv.CV_64F, 1, 0)
    sobel_y = cv.Sobel(original_image, cv.CV_64F, 0, 1)

    magnitude, angle = cv.cartToPolar(sobel_x, sobel_y, angleInDegrees = True)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(magnitude)
    ret, bin_img = cv.threshold(magnitude, limiar * maxVal, 255, cv.THRESH_BINARY)

    bin_img = cv.convertScaleAbs(bin_img)
    bin_img = cv.ximgproc.thinning(bin_img)

    cv.imshow("limiarizada e afinada", bin_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return bin_img

def get_angle(point, lowest_point):
    return m.atan2(point[1] - lowest_point[1], point[0] - lowest_point[0])

def get_distance(point1, point2):
    return m.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def ordenarPontos(points):
    centro = [0,0]
    for point in points:
        centro[0] += point[0]
        centro[1] += point[1]

    n = len(points)

    centro[0] /= n
    centro[1] /= n

    for point in points:
        point[0] -= centro[0]
        point[1] -= centro[1]

    # sort com funcao customizada
    sorted(points, key=cmp_to_key(comparePoints))

    for point in points:
        point[0] += centro[0]
        point[1] += centro[1]

    return points

def comparePoints(pt1, pt2):
    angle1 = getAngle((0, 0), pt1)
    angle2 = getAngle((0, 0), pt2)

    if(angle1 < angle2):
        return True
    
    d1 = getDistance((0, 0), pt1)
    d2 = getDistance((0, 0), pt2)

    if(angle1 == angle2) and (d1 < d2):
        return True
    
    return False

def getAngle(centro, pt):
    x = pt[0] - centro[0]
    y = pt[1] - centro[1]

    angle = m.atan2(x, y)

    if(angle <= 0):
        angle = 2 * m.pi + angle

    return angle

def getDistance(pt1, pt2):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]

    return m.sqrt(x**2 + y**2)

def globalProcessing(edge_image):
    # Definindo os parâmetros de Hough
    height, width = edge_image.shape
    max_dist = int(np.ceil(np.sqrt(height**2 + width**2)))
    rhos = np.linspace(-max_dist, max_dist, 2 * max_dist)
    thetas = np.deg2rad(np.arange(-90, 90))

    # Inicializando a acumuladora
    accumulator = np.zeros((2 * max_dist, len(thetas)), dtype=np.int32)

    # Povoando a matriz acumuladora
    y_indices, x_indices = np.nonzero(edge_image)  # Índices dos pixels de borda

    for i in range(len(x_indices)):
        x = x_indices[i]
        y = y_indices[i]
        for theta_index in range(len(thetas)):
            theta = thetas[theta_index]
            rho = int(round(x * np.cos(theta) + y * np.sin(theta))) + max_dist
            accumulator[rho, theta_index] += 1
            # print("theta index = ", theta_index)

    print("matriz acc = ", accumulator)

    return accumulator, rhos, thetas

def plot_hough_accumulator(accumulator, thetas, rhos):
    plt.imshow(accumulator, aspect='auto', cmap='binary', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[0], rhos[-1]])
    plt.title('Hough Transform Accumulator')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')
    plt.colorbar()
    plt.show()

def drawHoughLines(image, accumulator, rhos, thetas, threshold, color):
    # Copiar a imagem original para não modificar a original
    output_image = image.copy()

    # Encontrar picos na acumuladora
    peak_indices = np.argwhere(accumulator > threshold)
    
    print("y ind = ", peak_indices[:, 0])
    print("x ind = ", peak_indices[:, 1])
    
    for peak in peak_indices:
        rho_index = peak[0]
        theta_index = peak[1]
        
        rho = rhos[rho_index]
        theta = thetas[theta_index]
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        cv.line(output_image, (x1, y1), (x2, y2), color, 2)
    
    return output_image


print("Qual processamento você quer realizar? \n 1 - Processamento Local \n 2 - Processamento Regional \n 3 - Processamento Global")
choice = int(input())


# ------------------- PROCESSAMENTO LOCAL - BASEADO EM GRADIENTE ---------------------
if(choice == 1):
    Tm = float(input("Informe o limiar Tm: "))
    A = float(input("Informe a direção angular (A): "))
    Ta = float(input("Informe a faixa de direções aceitáveis (Ta): "))
    K = int(input("Informe o limiar (K) para correção de falhas: "))

    dimensions = img.shape
    rows1, cols1 = dimensions

    img_h = localProcessing(img, Tm, A, Ta, K, rows1, cols1)
    img_r = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    dimensions = img_r.shape
    rows2, cols2 = dimensions
    img_v = localProcessing(img_r, Tm, A, Ta, K, rows2, cols2)
    img_v = cv.rotate(img_v, cv.ROTATE_90_COUNTERCLOCKWISE)
    # cv.imshow("img_v rotacionada", img_v)
    # cv.waitKey(0)

    img_or = cv.bitwise_or(img_h, img_v)
    cv.imshow("imagem final", img_or)
    cv.waitKey(0)



# ------------------- PROCESSAMENTO REGIONAL - APROXIMAÇÃO POLIGONAL ---------------------
elif(choice == 2):
    binary = float(input("Digite a % a ser limiarizada da magnitude da imagem: "))
    T = float(input("Informe o limiar T: "))
    limit = float(input("Informe o limite de distancia para curvas fechadas: "))
    points = []
    bin_img = limiarize(img, binary)
    cv.imshow("imagem limiarizada", bin_img)

    for i in range(0, rows):
        for j in range(0, cols):
            if(bin_img[i][j] == 255):
                points.append((j, rows - i))
            
    points = np.array(points)

    points_ord = ordenarPontos(points)
    dist_mat = spatial.distance_matrix(points_ord, points_ord)
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    dist = m.dist(points[i], points[j])
    if(dist <= limit):
        closed = True
    else:
        closed = False

    # print("curva fechada? ", closed)

    aprox = regionalProcessing(points_ord, points_ord[0], points_ord[-1], closed)
    print(aprox)
    aprox = np.array(aprox)
    # aprox = aprox.reshape((-1, 1, 2))
    f_img = np.zeros((rows, cols))

    # if(closed):
    #     print("curva fechada")
    #     cv.line(f_img, aprox[0], aprox[-1], (255, 255, 255), 2)
    for i in range(len(aprox) - 1):
        cv.line(f_img, aprox[i], aprox[i + 1], (255, 255, 255), 2)
    # cv.polylines(f_img, [aprox], closed, (255, 255, 255), 1)

    cv.imshow("contorno", f_img)
    cv.waitKey(0)
    cv.destroyAllWindows()



# ------------------- PROCESSAMENTO GLOBAL - TRANSFORMADA DE HOUGH ---------------------
elif(choice == 3):
    lower = float(input("Limite inferior para o Canny: "))
    upper = float(input("Limite superior para o Canny: "))
    opc = int(input("Suavizar imagem? \n0 - Sim \n1 - Não\n"))
    color_opc = int(input("Desenhar linhas: \n0 - Pretas \n1 - Brancas\n"))
    if(not color_opc):
        color = (0, 0, 0)
    else:
        color = (255, 255, 255)

    if(not opc):
        img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)

    img_b = cv.Canny(img, lower, upper)
    cv.imshow("contorno", img_b)
    cv.waitKey(0)
    cv.destroyAllWindows()

    accumulator, rhos, thetas = globalProcessing(img_b)
    plot_hough_accumulator(accumulator, rhos, thetas)

    qtd_lines = float(input("Informe o limiar de linhas: "))

    img_c_linhas = drawHoughLines(img, accumulator, rhos, thetas, qtd_lines, color)

    cv.imshow("imagem com linhas geradas", img_c_linhas)
    cv.waitKey(0)
    # cv.destroyAllWindows(0)

else:
    print("Operação inválida.")
