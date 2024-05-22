import cv2 as cv
import numpy as np
import math as m
from scipy import spatial
import matplotlib.pyplot as plt

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
        print("dmax = ", dmax)
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
    # Encontre o ponto mais baixo
    lowest_point = min(points, key=lambda point: (point[1], point[0]))
    
    # Ordene os pontos com base em seus ângulos em relação ao ponto mais baixo
    sorted_points = sorted(points, key=lambda point: (get_angle(point, lowest_point), -get_distance(point, lowest_point)))
    
    return sorted_points

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

def polarCoord(radius, angle ):
    pass

def globalProcessing(img):
    height, width = img.shape
    max_dist = int(np.sqrt(height**2 + width**2))
    rhos = np.linspace(-max_dist, max_dist, 2 * max_dist)
    thetas = np.deg2rad(np.arange(-90, 90))

    accumulator = np.zeros((2 * max_dist, len(thetas)))

    y_ind, x_ind = np.nonzero(img) # pegando pixels de borda
    for i in range(len(x_ind)):
        print("gerando matriz acumuladora")
        x = x_ind[i]
        y = y_ind[i]
        for theta_ind in range(len(thetas)):
            theta = thetas[theta_ind]
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            accumulator[rho + max_dist, theta_ind] += 1

    return accumulator, rhos, thetas

def plot_hough_accumulator(accumulator, thetas, rhos):
    plt.imshow(accumulator, aspect='auto', cmap='hot', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[0], rhos[-1]])
    plt.title('Hough Transform Accumulator')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')
    plt.colorbar()
    plt.show()

def draw_hough_lines(image, accumulator, rhos, thetas, qtd_lines):
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # threshold = np.max(accumulator) - qtd_lines
    acc_copy = accumulator.copy()
    
    vet_x = []
    vet_y = []

    while qtd_lines > 0:
        # Encontrar picos na acumuladora
        index_linear = np.argmax(acc_copy)
        y_idxs, x_idxs = np.unravel_index(index_linear, acc_copy.shape)
        print("shape do x_idxs = ", x_idxs)
        print("shape do y_idxs = ", y_idxs)

        vet_x.append(x_idxs)
        vet_y.append(y_idxs)

        acc_copy[y_idxs, x_idxs] = -1

        qtd_lines -= 1
    
# Encontrar picos na acumuladora
    # y_idxs, x_idxs = np.where(accumulator > threshold)
    y_idxs, x_idxs = vet_y, vet_x
    
    for i in range(len(x_idxs)):
        rho = rhos[y_idxs[i]]
        theta = thetas[x_idxs[i]]
        
        a = cos_t[x_idxs[i]]
        b = sin_t[x_idxs[i]]
        
        x0 = a * rho
        y0 = b * rho
        
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    

    return image


print("Qual processamento você quer realizar? \n 1 - Processamento Local \n 2 - Processamento Regional \n 3 - Processamento Global")
choice = int(input())

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
    cv.imshow("img_v rotacionada", img_v)
    cv.waitKey(0)

    img_or = cv.bitwise_or(img_h, img_v)
    cv.imshow("imagem depois do OR", img_or)
    cv.waitKey(0)


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

    aprox = regionalProcessing(points_ord, points_ord[0], points_ord[-1], closed)
    print(aprox)
    aprox = np.array(aprox)
    aprox = aprox.reshape((-1, 1, 2))
    f_img = np.zeros((rows, cols))
    cv.polylines(f_img, [aprox], closed, (255, 255, 255), 1)

    cv.imshow("contorno", f_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


elif(choice == 3):
    lower = float(input("Limite inferior para o Canny: "))
    upper = float(input("Limite superior para o Canny: "))
    qtd_lines = float(input("Informe a qtd de linhas para desenhar: "))
    opc = int(input("Suavizar imagem? \n0 - Sim \n1 - Não\n"))
    if(not opc):
        img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)

    img_b = cv.Canny(img, 20, 100)
    cv.imshow("contorno", img_b)
    cv.waitKey(0)
    cv.destroyAllWindows()

    accumulator, rhos, thetas = globalProcessing(img)
    plot_hough_accumulator(accumulator, rhos, thetas)

    img_c_linhas = draw_hough_lines(img, accumulator, rhos, thetas, qtd_lines)

    cv.imshow("imagem com linhas geradas", img_c_linhas)
    cv.waitKey(0)
    cv.destroyAllWindows(0)

else:
    print("Operação inválida.")
