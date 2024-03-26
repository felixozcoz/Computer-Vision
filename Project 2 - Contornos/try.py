import cv2 as cv
import numpy as np
import array

# Leer imagen 
def hough(src):
    img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    orientation = np.arctan2(sobel_y, sobel_x)

    threshold = 130 # Umbral de thresholding

    # Aplicar filtro de thresholding
    _, thresh = cv.threshold(gradient, threshold, 255,cv.THRESH_BINARY)
    
    # cv.imshow("Threshold_before", thresh)
    
    # Aplicar threshold para quitar líneas verticales
    h, w = img.shape
    # Indices de píxeles blancos
    y_idx, x_idx = np.nonzero(thresh) # filas, columnas
    diff = np.radians(15)
    tosee = src.copy()

    for ind in range(len(y_idx)):
        # Obtener índices de keypoint en la imagen
        i = y_idx[ind]
        j = x_idx[ind]

            # verticales        
        a = (-diff < orientation[i, j] < diff)
        b = (-np.pi-diff < orientation[i, j] < -np.pi+diff)
        c = (np.pi-diff < orientation[i, j] < np.pi+diff)
            # horizontales
        d = (np.pi/2-diff < orientation[i, j] < np.pi/2+diff)
        e = (-np.pi/2-diff < orientation[i, j] < -np.pi/2+diff)

        if a | b | c | d | e:
            # print(i,j, orientation[i,j], np.degrees(orientation[i,j]))
            cv.circle(tosee, (j,i), 1, (0, 0, 255), 1)
            thresh[i,j] = 0

    # cv.imshow("Threshold", thresh)
    # cv.imshow("src", tosee)
    # exit() 

    y_idx, x_idx = np.nonzero(thresh) # filas, columnas
    accumulator = np.zeros(w, dtype=int)
    diag = int(np.sqrt((h//2)**2 + (w//2)**2))
    # accumulator = np.zeros(2*diag+1, dtype=int)
    print(accumulator.shape)
    pts = [set() for _ in range(w)]
    # Rectas que votan por puntos
    for ind in range(len(y_idx)):
        # Obtener índices de keypoint en la imagen
        i = y_idx[ind]
        j = x_idx[ind]

        # Extraer orientación del gradiente 
        theta = orientation[i, j]    # indica la pendiente de la recta perpendicular a la recta que buscamos en el espacio de Hough
        
        # Centrar imagen ((0,0) en centro de la imagen)
        x = j - w/2
        y = h/2 - i

        # Ecuación de la recta en espacio de Hough
        th_sin, th_cos = np.sin(theta),np.cos(theta)
        rho = x * th_cos + y * th_sin

        # Buscar coordenada x en la intersección con eje y en el espacio de Hough ( y = 0 )
        x_to_find = rho/th_cos # if th_cos != 0 else 0
        j_aprox = int(x_to_find + w/2)
        # print( "j=", j, "i=", i, "j_real=",j_real, "rho=", rho)#, "theta=", theta)
        # continue
        cv.circle(src, (j, i), 1, (0, 255, 0), 1)
        if 0 <= j_aprox < w:
            # Recta vota por punto j
            accumulator[j_aprox] += 1
            pts[j_aprox].add((j,i))

    indmax, valmax = k_maximos_con_indices(accumulator, 3)
    print(indmax, valmax)
    iter = 0
    colors= [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for ind in indmax:
        for p in pts[ind]:
            cv.circle(src, (p[0], p[1]), 1, colors[iter], 1)

        iter = iter + 1
    cv.circle(src, (indmax[0], h//2), 2, (0, 0, 0), 2)
    return src

def k_maximos_con_indices(arreglo, k):
    # Obtener los índices que ordenarían el arreglo de forma descendente
    indices_descendentes = np.argsort(arreglo)[::-1]
    
    # Obtener los K primeros índices (que corresponden a los K valores máximos)
    k_indices_maximos = indices_descendentes[:k]
    
    # Obtener los K valores máximos y sus índices
    k_valores_maximos = arreglo[k_indices_maximos]
    
    return k_indices_maximos, k_valores_maximos

src1 = cv.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo2.pgm")
src1 = hough(src1)

# src2 = cv.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo2.pgm")
# src2 = hough(src2)

# src3 = cv.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo3.pgm")
# src3 = hough(src3)

cv.imshow("Pasillo 1", src1)
# cv.imshow("Pasillo 2", src2)
# cv.imshow("Pasillo 3", src3)
cv.waitKey(0)