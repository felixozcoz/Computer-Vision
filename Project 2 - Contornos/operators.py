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
    gfx = ( gaussian_filter_1D_equation(x, sigma) ).reshape(1, kernel_size)  # x direction
    gfy = gfx.reshape(kernel_size, 1)                                        # y direction (from top to bottom)       

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
    gfdy = gfdx.reshape(kernel_size, 1)                                             # y direction  (from top to bottom)  

    return gfdx, gfdy


def Sobel_operator(image):
    '''
        Implement the Sobel operator

        Parameters:
            image (numpy array): Image to apply the operator

        Returns:
            Gx (numpy array): Gradient in the x direction
            Gy (numpy array): Gradient in the y direction
            G (numpy array): Gradient module
            theta (numpy array): Gradient orientation
    '''
    # create the Sobel kernels
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int8) # kernel in the x direction
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.int8) # kernel in the y direction
    # apply the kernels to the image
    # TODO: ajustar tipo de dato
    Gx = cv2.filter2D(image, cv2.CV_64F, kernelX)  
    Gy = cv2.filter2D(image, cv2.CV_64F, kernelY)   

    # [-255, 255]
    Gx = np.int8(Gx) 
    Gy = np.int8(Gy)
    
    # para plotear como en las diapos
    Gx = Gx//2 + 128
    Gy = Gy//2 + 128

    # devolver a uint8 para imprimir en imshow
    Gx = np.uint8(Gx)
    Gy = np.uint8(Gy)

    print(np.min(Gx), np.max(Gx))
    print(np.min(Gy), np.max(Gy))
    
    # llevar a range 0-255
    # Gx = np.uint8( np.absolute(Gx) ) # 0 - 255
    # Gy = np.uint8( np.absolute(Gy) ) # 0 - 255

    # calculate the gradient module and orientation
    # G = np.sqrt(Gx**2 + Gy**2)
    # theta = np.arctan2(Gy, Gx)

    # G = np.uint8( np.absolute(G) )

    # theta = np.uint8( theta/(np.pi * 128) )

    cv2.imshow("Gx", Gx)
    cv2.waitKey(0)

    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1,3,2),plt.imshow(Gx,cmap = 'gray')
    # plt.title('Gx'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1,3,3),plt.imshow(Gy,cmap = 'gray')
    # plt.title('Gy'), plt.xticks([]), plt.yticks([])
    # plt.show()

    return Gx, Gy



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
    # create 1D vectors for the Gaussian filter and its first derivative
    gfx, gfy = Gaussian_filter(kernel_size, sigma)    
    gfdx, gfdy = Gaussian_first_derivative(kernel_size, sigma)

    # create kernels
    kernelX = np.outer(gfy, gfdx) # kernel in the x direction
    kernelY = np.outer(gfdy, gfx) # kernel in the y direction (top to bottom)

    print(kernelX)
    print(kernelY)

    K = np.sum(np.maximum(0, kernelX))

    # normalization of the kernels (positive sum normalization)
    kernelX = kernelX / K
    kernelY = kernelY / K

    # apply the kernels to the image
    Gx = cv2.filter2D(img, -1, kernelX)
    Gy = cv2.filter2D(img, -1, kernelY)

    Gx += 128
    Gx = np.uint8(Gx/2) + 128

 
    # TODO: posible ajuste de rango de color depende de la imagen
    print(np.min(Gx), np.max(Gx))
    print(np.min(Gy), np.max(Gy))

    cv2.imshow("Original", image)
    cv2.imshow("Gx", Gx)
    cv2.imshow("Gy", Gy)
    cv2.waitKey(0)
    
    #return Gx, Gy



# ---------------------------------------------

from matplotlib import pyplot as plt

img = cv2.imread(r"C:\Users\felix\OneDrive\Escritorio\poster.pgm", cv2.IMREAD_GRAYSCALE)
Sobel_operator(img)

exit(0)
kernel_size = 5
sigma = 1

Canny_operator(img, kernel_size, sigma)
    