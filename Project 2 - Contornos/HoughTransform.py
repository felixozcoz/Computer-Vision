import cv2 as cv
import numpy as np
import array
import operators as op

# Leer imagen 
def hough(src):
    img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    h, w = img.shape


    Gx, Gy, gradient, orientation = op.Sobel_filter(img)

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

    # cv.imshow("Threshold_new", thresh)
    # cv.imshow("src", tosee)
    # cv.waitKey(0)
    # exit() 

    y_idx, x_idx = np.nonzero(thresh) # filas, columnas

    accumulator = np.zeros(w, dtype=int)
    # diag = int(np.sqrt((h//2)**2 + (w//2)**2))
    # accumulator = np.zeros(2*diag+1, dtype=int)
    # print(accumulator.shape)
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
        th_sin, th_cos = np.sin(theta), np.cos(theta)
        rho = x * th_cos + y * th_sin
        # Buscar coordenada x en la intersección con eje y en el espacio de Hough ( y = 0 )
        
            
        x_to_find = rho/th_cos # if th_cos != 0 else 0
        j_aprox = int(x_to_find + w/2)

      
        
        # print( "j=", j, "i=", i, "j_real=",j_real, "rho=", rho)#, "theta=", theta)
        # continue
        # aux = src.copy()
        # cv.circle(src, (j, i), 1, (0, 0, 255), 1)
        if 0 <= j_aprox < w:
            # print("i=", i, "j=", j, "j_aprox=", j_aprox, "x_to_find=", x_to_find, "rho=", rho, "cos=", th_cos, "sin=", th_sin, "x=", x, "y=", y, "theta=", theta, "theta_deg=", np.degrees(theta))
            # aux = src.copy()
            # for j_var in range(w):
            #     x_var = j_var - w/2
            #     y_var = int((rho - (x_var * th_cos)) / th_sin)
            #     i_var = int(h/2 - y_var)
            #     if 0 <= i_var < h: 
            #         cv.circle(aux, (j_var, i_var), 1, (0, 255, 0), 1)

            # cv.circle(src, (j, i), 1, (0, 255, 0), 1)
            # cv.circle(aux, (j, i), 2, (0, 0, 255), 2)
            # cv.circle(aux, (j_aprox, h//2), 2, (255, 0, 0), 2)
            # cv.imshow("aux", aux)
            # cv.waitKey(0)
            # Recta vota por punto j
            accumulator[j_aprox] += 1
            pts[j_aprox].add((j,i))
    

    ind, val = k_maximos_con_indices(accumulator, 5)
    print(ind, val)

    for i in ind:
        cv.circle(src, (i, h//2), 2, (0, 255, 0), 2)    # Puntos más votados

    # median = np.median(accumulator)
    # max = np.argmax(accumulator)
    # for i in range(len(accumulator)):
    #     if accumulator[i] > median:
    #         cv.circle(src, (i, h//2), 1, (255, 0, 0), 1)    # Puntos más votados
    max_ind = ind[0]
    for p in pts[max_ind]:
        cv.line(src, (p[0], p[1]), (max_ind, h//2), (255, 0, 0), 1)
        cv.circle(src, p, 2, (0, 0, 255), 2)    # puntos que votaron por la recta más votada

    cv.circle(src, (max_ind, h//2), 2, (0, 0, 0), 2)  # Punto más votado

    return src

def k_maximos_con_indices(arreglo, k):
    # Obtener los índices que ordenarían el arreglo de forma descendente
    indices_descendentes = np.argsort(arreglo)[::-1]
    
    # Obtener los K primeros índices (que corresponden a los K valores máximos)
    k_indices_maximos = indices_descendentes[:k]
    
    # Obtener los K valores máximos y sus índices
    k_valores_maximos = arreglo[k_indices_maximos]
    
    return k_indices_maximos, k_valores_maximos

src2 = cv.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo2.pgm")
# src1 = cv.imread(r"C:\Users\felix\OneDrive\Escritorio\oblicua.png")
src2 = hough(src2)

src3 = cv.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo3.pgm")
src3 = hough(src3)

cv.imshow("Pasillo 2", src2)
cv.imshow("Pasillo 3", src3)
cv.waitKey(0)