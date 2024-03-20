import operators as op
import numpy as np
import cv2

#Función de la transformada de Hough (errores por corregir)
def Hough_transform_gradient(image,threshold):
    '''
        Implement the Hough transform with orientation gradient applying the Sobel operator
        Parameters:
            image (int): original image
            threshold (int): orientation gradient magnitude min limit (umbral)
        Returns:
            image: Image with the edges delimited after Hough transform
    '''
    # Calcular el gradiente utilizando el operador Sobel
    # grad_x, grad_y = op.Sobel_operator(image)
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular la magnitud del gradiente
    magnitude = np.sqrt(grad_x.astype(np.int32)** 2 + grad_y.astype(np.int32) ** 2)

    # Definir los rangos de rho y theta
    rho_max = int(np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2))
    theta_max = np.pi

    # Crear la matriz de acumulación
    accumulator = np.zeros((2 * rho_max, int(np.ceil(theta_max))), dtype=np.uint64)

    # Crear una cuadrícula de coordenadas x y y
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    # Calcular todas las magnitudes y orientaciones de gradiente simultáneamente
    magnitudes = magnitude > threshold
    orientations = np.arctan2(grad_y, grad_x)

    # Crear una cuadrícula de theta para todos los píxeles
    thetas = np.linspace(0, np.pi, num=orientations.shape[1], endpoint=False)

    # Calcular rho para todos los píxeles y thetas simultáneamente
    rhos = (xx[:, :, np.newaxis] * np.cos(thetas) + yy[:, :, np.newaxis] * np.sin(thetas)).astype(int)
    orientations = orientations[:, :, np.newaxis]  # Añadir una dimensión adicional

    # Incrementar los contadores en el acumulador
    accumulator[rhos + rho_max, np.round(orientations * (accumulator.shape[1] - 1)).astype(int)] += magnitudes

    hough_lines = np.argwhere(accumulator > 0)

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

    return image


# --- Verificación de la transformada de Hough ---

img1 = cv2.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo1.pgm", cv2.IMREAD_GRAYSCALE)

# img1 = cv2.imread(r"C:\Users\usuario\Desktop\Contornos\pasillo1.pgm", cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread(r"C:\Users\usuario\Desktop\Contornos\pasillo2.pgm", cv2.IMREAD_GRAYSCALE)
#img3 = cv2.imread(r"C:\Users\usuario\Desktop\Contornos\pasillo3.pgm", cv2.IMREAD_GRAYSCALE)

hough_img1 = Hough_transform_gradient(img1,100)
#hough_img2 = Hough_transform_gradient(img2,100)
#hough_img3 = Hough_transform_gradient(img3,100)

cv2.imshow('Original Image 1',img1)
#cv2.imshow('Original Image 2',img2)
#cv2.imshow('Original Image 3',img3)

cv2.imshow('Hough Image 1', hough_img1)
#cv2.imshow('Hough Image 2', hough_img2)
#cv2.imshow('Hough Image 3', hough_img3)
cv2.waitKey(0)