import cv2
import numpy as np

def transformada_hough(imagen, umbral):
    # Detección de bordes (Canny)
    bordes = cv2.Canny(imagen, 50, 150)
    
    # Dimensiones de la imagen
    altura, ancho = imagen.shape[:2]
    
    # Rango de theta: de -90 a 90 grados (en radianes)
    thetas = np.deg2rad(np.arange(-90, 90))
    
    # Rango de rho: de -diagonal a diagonal (diagonal de la imagen)
    diagonal = np.sqrt(altura**2 + ancho**2)
    rhos = np.linspace(-diagonal, diagonal, int(diagonal * 2))
    
    # Matriz de acumulación de votos
    acumulacion = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    
    # Obtener índices de píxeles de borde
    y_idx, x_idx = np.nonzero(bordes)
    
    # Para cada píxel de borde, calcular rho y theta y votar en la matriz de acumulación
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        
        for t_idx, theta in enumerate(thetas):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = np.argmin(np.abs(rhos - rho))
            acumulacion[rho_idx, t_idx] += 1
    
    # Aplicar umbral para obtener líneas detectadas
    lineas = []
    for r_idx, rho in enumerate(rhos):
        for t_idx, theta in enumerate(thetas):
            if acumulacion[r_idx, t_idx] > umbral:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                lineas.append(((x1, y1), (x2, y2)))
    
    return lineas

# Leer la imagen en escala de grises
imagen = cv2.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo1.pgm", cv2.IMREAD_GRAYSCALE)

# Aplicar la Transformada de Hough
lineas_detectadas = transformada_hough(imagen, umbral=100)

# Dibujar las líneas detectadas en la imagen original
imagen_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
for linea in lineas_detectadas:
    cv2.line(imagen_color, linea[0], linea[1], (0, 0, 255), 2)

# Mostrar la imagen con las líneas detectadas
cv2.imshow('Lineas Detectadas', imagen_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
