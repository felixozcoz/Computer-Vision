# ---------------------------------------------
# Fichero: operators.py
# ---------------------------------------------
# Escuela de Ingeniería y Arquitectura de Zaragoza
# Visión por Computador	
# 2023- 2024
#
# Félix Ozcoz Eraso             801108
# Victor Marcuello Baquero      741278
#
# Descripción:
#   Módulo que contiene implementaciones de operadores
#   de gradiente, módulo y orientación
# ---------------------------------------------

import cv2
import numpy as np

def Gaussian_filter(x, sigma):
    return np.sqrt(1 / (2 * np.pi * sigma)) * np.exp( -(x**2) / (2 * sigma**2))  # one-dimensional Gaussian filter


def Gaussian_first_derivative(x, sigma):
    return ((-x) / (sigma**2)) * np.exp((-x**2) / (2 * sigma**2)) # First derivative of the 1D Gaussian filter formula


def Canny_operator(image, sigma, kernel_size=5):
    '''
        Implement the Canny operator

        Parameters:
            sigma (float): Standard deviation of the Gaussian filter
            kernel_size (int): Size of the kernel

        Returns:
            Gx (numpy array): Gradient in the x direction
            Gy (numpy array): Gradient in the y direction
            G (numpy array): Gradient module
            theta (numpy array): Gradient orientation
    '''
    # Construir kernel 1D 

    

    x1 = np.zeros((1, kernel_size), dtype=np.float32)
    x2 = np.zeros((kernel_size, 1), dtype=np.float32)

    y1 = np.zeros((1, kernel_size), dtype=np.float32)
    y2 = np.zeros((kernel_size, 1), dtype=np.float32)

    center = kernel_size // 2


    

    # Generar kernel 2D Gaussian Filter
    for x in range(kernel_size):
        for y in range(kernel_size):
            dx = x - center
            dy = y - center
            x1[0, x] = -1 * Gaussian_first_derivative(dx, sigma)
            x2[y, 0] = Gaussian_filter(dy, sigma)

            y1[0, x] = Gaussian_filter(dx, sigma)
            y2[y, 0] = Gaussian_first_derivative(dy, sigma)

    
    print(x1)
    print(x2)
    print(y1)
    print(y2)

    kernelX = np.dot(x2, x1)
    kernelY = np.dot(y2, y1)

    print(kernelX)
    print(kernelY)

    # aplicar los kernels a la imagen con filter 2D

    return grad_x, grad_y


    


img = cv2.imread(r"C:\Users\felix\OneDrive\Escritorio\chess_table.jpg", cv2.IMREAD_GRAYSCALE)
Gx, Gy = Canny_operator(img, 1, 5)
cv2.imshow("Gx", Gx)
cv2.imshow("Gy", Gy)
cv2.waitKey(0)