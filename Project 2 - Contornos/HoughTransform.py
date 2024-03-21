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
            threshold (int): orientation gradient magnitude minimum limit (umbral)
        Returns:
            vanishing_point: coordinates in which the vanishing point is present at the image
    '''
    # Calcular el gradiente utilizando el operador Sobel
    _, _, gradient, orientation = op.Sobel_filter(image)

    # Seleccionar el conjunto de coordenadas de la línea central
    # gradient.shape[0] = altura (x)
    # gradient.shape[1] = base (y)
    central_y = np.uint32(gradient.shape[0] / 2)
    central_x = np.uint32(gradient.shape[1] / 2)
    
    # Calcular la transformada de Hough con orientación de gradiente para los puntos de la fila central
    accumulator = np.zeros(gradient.shape[1])
    
    # Comprobar que el punto de fuga no se aproxima ni a nivel horizonal ni a nivel vertical
    #gradient_round_x = (np.abs(theta) > np.radians(5)) and (np.abs(theta - np.pi/2) > np.radians(5)) and (np.abs(theta - np.pi) > np.radians(5))
    #gradient_round_y = (np.abs(theta - np.pi*2/3) > np.radians(5)) and (np.abs(theta - 2*np.pi) > np.radians(5)) 
    
    # Comprobar el punto de fuga en la imagen a través del sistema de votación
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            # En caso de que el módulo del gradiente supera el umbral, el píxel realiza su voto 
            if gradient[i,j] >= threshold:   
                # Coordenada x en base a la coordenada <x> central y a la posición <j>
                x = j - central_x
                # Coordenada y en base a la coordenada <y> central y a la posición <i>
                y = central_y - i
                # Representar en <theta> el grado de orientación
                theta = orientation[i,j]
                # Calcular la ecuación de la recta con las coordenadas polares (p)
                rho = x*np.cos(theta) + y*np.sin(theta)
                # Calcular la coordenada x de la línea central donde intersecta con la ecuación p.  
                x_vote = int((rho / np.cos(theta) + central_x))
                
                # Aceptar el voto si el valor está ubicado en la línea central
                if x_vote >= 0 and x_vote < gradient.shape[1]:
                    accumulator[x_vote] += 1
    
    x_vanishing = np.argmax(accumulator)
    y_vanishing = central_y
    vanishing_point = [x_vanishing, y_vanishing]
    
    plt.imshow(image)
    plt.scatter(vanishing_point[0], vanishing_point[1], c='red', marker='x', s=100)
    plt.axhline(image.shape[0] // 2, color='r', linestyle='--', linewidth=1)  # Línea horizontal que representa la fila central
    plt.title('Punto de Fuga Detectado')
    plt.show()

    return vanishing_point

                
                
                 
            
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

Hough_transform_gradient(img1,100)
Hough_transform_gradient(img2,100)
Hough_transform_gradient(img3,100)
Hough_transform_gradient(sunset,100)


#cv2.imshow('Original Image 1',img1)
#cv2.imshow('Original Image 2',img2)
#cv2.imshow('Original Image 3',img3)

#cv2.imshow('Hough Image 1', hough_img1)
#cv2.imshow('Hough Image 2', hough_img2)
#cv2.imshow('Hough Image 3', hough_img3)
cv2.waitKey(0)