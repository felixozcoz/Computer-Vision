import operators as op
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_central_points(image, num_points):
    # Obteniendo la fila central
    central_row = image[image.shape[0] // 2, :]
    
    # Calculando la discretización
    discretization = len(central_row) // num_points
    
    # Tomando los puntos centrales con la discretización adecuada
    central_points = []
    for i in range(0, len(central_row), discretization):
        central_points.append((i, central_row[i]))
    
    return central_points

def find_vanishing_point(accumulator):
    # Encontrar el punto de fuga como el punto de intersección de las líneas detectadas en la transformada de Hough
    max_coord = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    rho = max_coord[0]
    theta = max_coord[1]

    # Calcular las coordenadas x e y del punto de fuga
    x_vanishing = int(rho * np.cos(theta))
    y_vanishing = int(rho * np.sin(theta))

    return x_vanishing, y_vanishing

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
    _, _, gradient, orientation = op.Sobel_filter(image)

    # Seleccionar la fila central horizontal de la imagen
    central_row = image[gradient.shape[0] // 2, :]
    central_x = gradient.shape[0] // 2
    central_y = gradient.shape[1] // 2
    
    # Calcular la transformada de Hough con orientación de gradiente para los puntos de la fila central
    accumulator = np.zeros(gradient.shape[1])
    
    
    for i in range(gradient.shape[0]-1):
        for j in range(gradient.shape[1]-1):
            theta = orientation[i,j]
            
            gradient_round_x = np.abs(theta) > np.radians(5) and np.abs(theta - np.pi/2) > np.radians(5) and np.abs(theta - np.pi) > np.radians(5)
            gradient_round_y = np.abs(theta - np.pi*(2/3)) > np.radians(5) and np.abs(theta - 2*np.pi) > np.radians(5) 
            
            if gradient[i,j] >= threshold and (gradient_round_x and gradient_round_y):            
                x = j - central_x
                y = central_y - i
                rho = x*np.cos(theta) + y*np.sin(theta)
                vote = int((rho / np.cos(theta) + central_x))
                
                if vote >= 0 and vote < gradient.shape[1]:
                    accumulator[vote] += 1
    
    x_vanishing = np.argmax(accumulator)
    y_vanishing = central_y
    
    
    
    #thetas = np.deg2rad(np.arange(0, 180))
    #for x, _ in enumerate(central_row):
    #    if image_thresholded[central_row_index, x] != 0:
    #        gradient_orientation = orientation[central_row_index, x]
    #        for theta_idx, theta in enumerate(thetas):
    #            rho = int(round(x * np.cos(theta - gradient_orientation) + central_row_index * np.sin(theta - gradient_orientation)))
    #            accumulator[rho, theta_idx] += 1

    # Encontrar el punto de fuga
    #x_vanishing, y_vanishing = find_vanishing_point(accumulator)
    
    plt.imshow(image)
    plt.scatter(x_vanishing, y_vanishing, c='red', marker='x', s=100)
    plt.axhline(image.shape[0] // 2, color='r', linestyle='--', linewidth=1)  # Línea horizontal que representa la fila central
    plt.plot(central_row)
    plt.title('Punto de Fuga Detectado')
    plt.show()

    return image

                
                
                 
            
    # Dibujar las líneas detectadas en la imagen original
    #for rho, theta in hough_lines:
    #    a = np.cos(np.deg2rad(theta))
    #    b = np.sin(np.deg2rad(theta))
    #    x0 = a * rho
    #    y0 = b * rho
    #    x1 = int(x0 + 1000 * (-b))
    #    y1 = int(y0 + 1000 * (a))
    #    x2 = int(x0 - 1000 * (-b))
    #    y2 = int(y0 - 1000 * (a))
    #    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# --- Verificación de la transformada de Hough ---

#img1 = cv2.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo1.pgm", cv2.IMREAD_GRAYSCALE)

img1 = cv2.imread(r"C:\Users\usuario\Desktop\Contornos\pasillo1.pgm", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"C:\Users\usuario\Desktop\Contornos\pasillo2.pgm", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(r"C:\Users\usuario\Desktop\Contornos\pasillo3.pgm", cv2.IMREAD_GRAYSCALE)
sunset = cv2.imread(r"C:\Users\usuario\Desktop\Contornos\sunset.png", cv2.IMREAD_GRAYSCALE)

hough_img1 = Hough_transform_gradient(img1,100)
hough_img2 = Hough_transform_gradient(img2,100)
hough_img3 = Hough_transform_gradient(img3,100)
sunset_img = Hough_transform_gradient(sunset,100)


#cv2.imshow('Original Image 1',img1)
#cv2.imshow('Original Image 2',img2)
#cv2.imshow('Original Image 3',img3)

#cv2.imshow('Hough Image 1', hough_img1)
#cv2.imshow('Hough Image 2', hough_img2)
#cv2.imshow('Hough Image 3', hough_img3)
cv2.waitKey(0)