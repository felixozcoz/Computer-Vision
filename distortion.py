# ---------------------------------------------
# Fichero: distortion.py
# ---------------------------------------------
# Escuela de Ingeniería y Arquitectura de Zaragoza
# Visión por Computador	
# 2023- 2024
#
# Félix Ozcoz Eraso     801108
# Victor Marcuello Baquero  
#
# Descripción:
#   Programa que aplica un filtro de distorsión 
#   a la imagen capturada por la cámara.
# ---------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Apply the barrel distortion filter
# Parameters:
#   - image: source image (numpy array format)
#   - k1: coefficiente of distortion 1
#   - k2: coefficiente de distortion 2
# k = 0 --> not effect
# k < 0 --> pincushion distortion
# k < 0 --> barrel distortion
def geometric_distortion(image, k1, k2=0.0):

    [ydim, xdim] = image.shape[:2]
    cx, cy = xdim/2, ydim/2  # define distortion centre

    # define the grid to manipulate the image
    x, y = np.meshgrid(np.arange(xdim), np.arange(ydim))
    x, y = x - cx, y - cy   # centering the image
    
    # calculate radius
    r = np.sqrt(x**2 + y**2)

    # calculate distorted radius
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_distorted = r * (1 + k1 * r**2 + k2 * r**4)
    
    print(r_distorted)
    # calculate distorte coordinates
    # nota: esta fórmula es igual de válida que la que nos dan, pero esta 
    # se centra en la distorsión basada en la relación entre las distancias radiales
    # a diferencia de la que nos dan que se centra en el desplazamiento y la distorsión
    # basada en la distancia 'r' y la posición 'x' con respecto al centro
    x_dis = x * ( r_distorted / r ) + cx
    y_dis = y * ( r_distorted / r ) + cy

    # remap image
    image_distorted = cv2.remap(image, x_dis.astype(np.float32), y_dis.astype(np.float32), cv2.INTER_LINEAR)

    return image_distorted

