import cv2 as cv
import numpy as np
import math

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

    # Quitar las líneas verticales, porque contribuyen a la detección de bordes
    # Utilizar orientation, quitar los puntos cuyo gradiente sea próximo a 0 0 180 que son ya que son perpendiculares a las líneas verticales
    # Aplicar threshold para quitar líneas verticales
    cv.imshow("Threshold", thresh)
    cv.waitKey(0)

    h, w = img.shape
    
    # Indices de píxeles blancos
    y_idx, x_idx = np.nonzero(thresh) # filas, columnas
    accumulator = np.zeros((w, 1), dtype=int)

    # Rectas que votan por puntos
    for ind in range(len(y_idx)):
        # Obtener índices de keypoint en la imagen
        i = y_idx[ind]
        j = x_idx[ind]

        # Extraer orientación del gradiente 
        theta = orientation[i, j]    # indica la pendiente de la recta perpendicular a la recta que buscamos en el espacio de Hough
        
        # Centrar imagen ((0,0) en centro de la imagen)
        x = j - w//2
        y = h//2 - i

        # Ecuación de la recta en espacio de Hough
        rho = x * np.cos(theta) + y * np.sin(theta)

        # Buscar coordenada x en la intersección con eje y en el espacio de Hough ( y = 0 )
        x_to_find = rho/np.cos(theta)

        # Devolver a coordenadas de la imagen
        j_buscado = int(np.round(x_to_find + w//2))

        if 0 <= j_buscado < w:
            # Recta vota por punto j
            accumulator[j_buscado] += 1

    # print(accumulator)
    median = np.median(accumulator)
    ind = np.where(accumulator > median+max(accumulator//2))
    for ind in ind[0]:
        cv.circle(src, (ind, 256), 2, (0, 0, 255), 2)
    
    return src


src1 = cv.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo1.pgm")
src1 = hough(src1)

src2 = cv.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo2.pgm")
src2 = hough(src2)

src3 = cv.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo3.pgm")
src3 = hough(src3)

cv.imshow("Pasillo 1", src1)
cv.imshow("Pasillo 2", src2)
cv.imshow("Pasillo 3", src3)
cv.waitKey(0)