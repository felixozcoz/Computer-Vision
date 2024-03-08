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

def Gaussian_filter(kernel_size, sigma):
    '''
        Implement the Gaussian filter

        Parameters:
            kernel_size (int): Size of the kernel
            sigma (float): Standard deviation of the Gaussian filter
        
        Returns:
            gfx (numpy array): Gaussian filter in the x direction
    '''
    # create an array from -kernel_size/2 to kernel_size/2
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)

    gaussian_filter_1D_equation = lambda x, sigma : (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp( -x**2 / (2 * sigma**2))

    # calculate the Gaussian filter
    gfx = ( -1 * gaussian_filter_1D_equation(x, sigma) ).reshape(1, kernel_size)  # x direction
    gfy = gfx.reshape(kernel_size, 1)                                        # y direction       

    return gfx, gfy


def Gaussian_first_derivative(kernel_size, sigma):
    '''
        Implement the first derivative of the Gaussian filter

        Parameters:
            kernel_size (int): Size of the kernel
            sigma (float): Standard deviation of the Gaussian filter
        
        Returns:
            gfdx (numpy array): First derivative of the Gaussian filter in the x direction
            gfdy (numpy array): First derivative of the Gaussian filter in the y direction
    '''
    # create an array from -kernel_size/2 to kernel_size/2
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)

    gaussian_first_derivative_eq = lambda x, sigma : ((-x) / (sigma**2)) * np.exp((-x**2) / (2 * sigma**2))

    # calculate the first derivative of the Gaussian filter
    gfdx = ( -1 * gaussian_first_derivative_eq(x, sigma) ).reshape(1, kernel_size)  # x direction
    gfdy = gfdx.reshape(kernel_size, 1)[::-1]                                             # y direction                 

    return gfdx, gfdy



def Canny_operator(image, kernel_size=3, sigma=1):
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
    gfx, gfy = Gaussian_filter(kernel_size, sigma)    # Gaussian filter in the x direction
    gfdx, gfdy = Gaussian_first_derivative(kernel_size, sigma) # First derivative of the Gaussian filter in the x direction

    print(gfdx)
    print(gfdy)


    kernelX = np.dot(gfy, gfdx) # kernel in the x direction
    kernelY = np.dot(gfdy, gfx) # kernel in the y direction

    print(kernelX)
    print(kernelY)

    # normalize the kernels
    K = np.sum(kernelX[kernelX > 0]) 
    kernelX = kernelX / K
    kernelY = kernelY / K

    # apply the kernels to the image
    Gx = cv2.filter2D(image, -1, kernelX)
    Gy = cv2.filter2D(image, -1, kernelY)

    # TODO: posible ajuste de rango de color
    print(np.min(Gx), np.max(Gx))
    print(np.min(Gy), np.max(Gy))

    return Gx, Gy





img = cv2.imread(r"C:\Users\felix\OneDrive\Escritorio\poster.pgm", cv2.IMREAD_GRAYSCALE)

kernel_size = 5
sigma = 1

Gx, Gy = Canny_operator(img, kernel_size, sigma)
cv2.imshow("Original", img)
cv2.imshow("Gx", Gx)
cv2.imshow("Gy", Gy)
cv2.waitKey(0)
    