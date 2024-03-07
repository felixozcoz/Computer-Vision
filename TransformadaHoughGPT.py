import numpy as np
import cv2
import math

def hough_transform(image, threshold):
    # Dimensiones de la imagen
    height, width = image.shape

    # Rango de valores para rho y theta
    max_rho = int(math.sqrt(height**2 + width**2))
    max_theta = 180

    # Matriz acumuladora Hough
    accumulator = np.zeros((max_rho, max_theta), dtype=np.uint8)

    # Calcular senos y cosenos de los ángulos
    cos_theta = np.cos(np.deg2rad(np.arange(max_theta)))
    sin_theta = np.sin(np.deg2rad(np.arange(max_theta)))

    # Búsqueda de píxeles de borde
    edge_points = np.argwhere(image > 0)

    # Loop a través de los píxeles de borde
    for y, x in edge_points:
        # Loop a través de los valores de theta
        for t_index in range(max_theta):
            # Calcular rho
            rho = int(x * cos_theta[t_index] + y * sin_theta[t_index])

            # Incrementar el valor en la matriz acumuladora
            accumulator[rho, t_index] += 1

    # Encontrar las coordenadas de los píxeles con valores acumulados por encima del umbral
    lines = np.argwhere(accumulator >= threshold)

    return lines

# Leer la imagen
image = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar un umbral para obtener los bordes
_, edges = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Realizar la transformada de Hough
threshold = 100
hough_lines = hough_transform(edges, threshold)

# Dibujar las líneas detectadas en la imagen original
for rho, theta in hough_lines:
    a = np.cos(np.deg2rad(theta))
    b = np.sin(np.deg2rad(theta))
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Mostrar la imagen con las líneas detectadas
cv2.imshow('Hough Transform', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
