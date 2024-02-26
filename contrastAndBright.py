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
    for idx, x in np.ndenumerate(image):
        new_image[idx] = np.clip(alpha * image[idx] + beta, 0, 255)
    return new_image


# ejemplo de uso y comparativa
# image = cv2.imread(r'C:\Users\Lenovo\OneDrive\Escritorio\lena_color.jpg', cv2.IMREAD_GRAYSCALE)
#image = cv2.imread(r'C:\Users\usuario\Desktop\lena_color.jpg', cv2.IMREAD_GRAYSCALE)
#newImage = np.zeros(image.dtype, image.shape)

#cv2.convertScaleAbs(image, newImage, 1, 40)

#cv2.imshow('Original', image)
#cv2.imshow('Mine', contrastNbright(image, 10, 40))
#cv2.imshow('OpenCV', newImage)

#cv2.waitKey(0)


# Iniciar la captura de la cámara
cap = cv2.VideoCapture(0)

# Esperar a que la cámara se abra
while not cap.isOpened():
    pass

# Ciclo principal para procesar los fotogramas de la cámara
while True:
    # Capturar fotograma de la cámara
    ret, frame = cap.read()

    # Verificar si se capturó correctamente un fotograma
    if not ret:
        print("Error al capturar fotograma")
        break

    # Mostrar el fotograma resultante
    #cv2.imshow('PosterFilter', colorReduce(frame))
    cv2.imshow('ConstrastAndBrightFilter',contrastNbright(frame,10,40))

    # Detener la ejecución si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()