# ---------------------------------------------
# Fichero: contrastAndBright.py
# ---------------------------------------------
# Escuela de Ingeniería y Arquitectura de Zaragoza
# Visión por Computador	
# 2023- 2024
#
# Félix Ozcoz Eraso     801108
# Victor Marcuello Baquero  
#
# Descripción:
#   Programa que aplica un filtro de contraste y brillo
# ---------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Función que aplica un filtro de contraste y brillo a una imagen
# Parámetros:
#   image: imagen de entrada
#   alpha: factor de contraste
#   beta: factor de brillo
# Salida:
#   new_image: imagen resultante
def contrastNbright(image, alpha, beta):
    new_image = np.zeros(image.shape, image.dtype)
    for idx,x in np.ndenumerate(image):
        new_image[idx] = np.clip(alpha * image[idx] + beta, 0, 255)
    return new_image




# ejemplo de uso y comparativa
image = cv2.imread(r'C:\Users\Lenovo\OneDrive\Escritorio\lena_color.jpg', cv2.IMREAD_GRAYSCALE)
newImage = np.zeros(image.shape, image.dtype)

cv2.convertScaleAbs(image, newImage, 1, 40)

cv2.imshow('Original', image)
cv2.imshow('Mine', contrastNbright(image, 10, 40))
cv2.imshow('OpenCV', newImage)

cv2.waitKey(0)


