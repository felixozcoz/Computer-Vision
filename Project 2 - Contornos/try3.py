import cv2
import numpy as np

def Hough_transform(gradient, orientation, threshold):

    height, width = gradient.shape
    central_y = int(height / 2)
    central_x = int(width / 2)
    horizon = np.zeros(width)
    for i in range(height-1):
        for j in range(width-1):
            theta = orientation[i,j]
            # Además del umbral(threshold), comprobamos que la orientación no sea aproximadamente vertical ni horizontal
            if ((gradient[i,j] >= threshold) and 
                ((np.abs(theta) > np.radians(5)) and (np.abs(theta - np.pi/2) > np.radians(5)) and
                 (np.abs(theta - np.pi) > np.radians(5)) and (np.abs(theta - (np.pi * (2/3)) > np.radians(5))) and
                 (np.abs(theta - 2*np.pi) > np.radians(5)))):
                x = j - central_x
                y = central_y - i
                # Calculamos la ecuación de la recta con las coordenadas polares
                rho = x*np.cos(theta) + y*np.sin(theta)
                # Calculamos la coordenada x de la línea central donde intersecciona
                vote_x = int((rho / np.cos(theta) + central_x))
                # Si la x está dentro de la imagen se vota el píxel
                if((vote_x >= 0) and (vote_x < width)) :
                    horizon[vote_x] += 1

    return [np.argmax(horizon), central_y]



img = cv2.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo1.pgm", cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
gradient = np.sqrt(sobel_x**2 + sobel_y**2)
orientation = np.arctan2(sobel_y, sobel_x)
Hough_transform