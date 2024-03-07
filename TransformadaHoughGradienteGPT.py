import numpy as np
import cv2
import math

def hough_transform_with_gradient(image, threshold):
    # Calcular gradientes utilizando Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)    

    # Calcular la magnitud y la dirección del gradiente
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    # Definir los rangos de rho y theta
    height, width = image.shape
    max_rho = int(math.sqrt(height**2 + width**2))
    max_theta = 180

    # Crear matriz acumuladora Hough
    accumulator = np.zeros((max_rho, max_theta), dtype=np.uint8)

    # Búsqueda de píxeles de borde
    edge_points = np.argwhere(image > 0)

    # Loop a través de los píxeles de borde
    for y, x in edge_points:
        gradient_theta = gradient_direction[y, x]
        for t_index in range(max_theta):
            theta = np.deg2rad(t_index)
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            accumulator[rho, t_index] += 1

    # Encontrar las coordenadas de los píxeles con valores acumulados por encima del umbral
    lines = np.argwhere(accumulator >= threshold)

    return lines

# Leer la imagen
image = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar un umbral para obtener los bordes
_, edges = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Realizar la transformada de Hough con gradiente
threshold = 100
hough_lines = hough_transform_with_gradient(edges, threshold)

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
cv2.imshow('Hough Transform with Gradient', image)
cv2.waitKey(0)
cv2.destroyAllWindows()