import cv2 as cv
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    
    vanishing_lines = []
    
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
                # Aceptar el voto si el valor de la coordenada x está en el rango [0,width_gradient]
                if x_vote >= 0 and x_vote < gradient.shape[1]:
                    accumulator[x_vote] += 1
    
    x_vanishing = np.argmax(accumulator)
    vanishing_point = [x_vanishing, central_y]
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if accumulator[j] > 0:
                x = j - central_x
                y = central_y - i
                theta = orientation[i,j]
                rho = x*np.cos(theta) + y*np.sin(theta)
                x_vote = int((rho / np.cos(theta) + central_x))
                if x_vote == vanishing_point[1]:
                    vanishing_lines.append([(i, j), (vanishing_point[0],vanishing_point[1])])

    return vanishing_point, vanishing_lines


def plot_vanishing_point(image,vanishing_point,vanishing_lines):
    hough_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for line in vanishing_lines:
        p1, p2 = line
        x1, y1 = p1
        x2, y2 = p2
        cv2.line(hough_image,(x1,y1),(x2,y2),(0,255,0),1)
    
    
    # Dibujar el punto de fuga mediante una cruz
    cv2.line(hough_image, [vanishing_point[0]-5,vanishing_point[1]], [vanishing_point[0]+5,vanishing_point[1]], color=[0, 0, 250], thickness=2)
    cv2.line(hough_image, [vanishing_point[0],vanishing_point[1]-5], [vanishing_point[0],vanishing_point[1]+5], color=[0, 0, 250], thickness=2)

    return hough_image

#img1 = cv2.imread(r"C:\Users\felix\OneDrive\Escritorio\Contornos\pasillo1.pgm", cv2.IMREAD_GRAYSCALE)

img1 = cv2.imread(r"C:\Users\usuario\Desktop\Contornos\pasillo1.pgm", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"C:\Users\usuario\Desktop\Contornos\pasillo2.pgm", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(r"C:\Users\usuario\Desktop\Contornos\pasillo3.pgm", cv2.IMREAD_GRAYSCALE)
sunset = cv2.imread(r"C:\Users\usuario\Desktop\Contornos\sunset.png", cv2.IMREAD_GRAYSCALE)

vanishing_point1, vanishing_lines1 = Hough_transform_gradient(img1,100)
vanishing_point2, vanishing_lines2 = Hough_transform_gradient(img2,100)
vanishing_point3, vanishing_lines3 = Hough_transform_gradient(img3,100)
vanishing_point4, vanishing_lines4 = Hough_transform_gradient(sunset,100)

hough_img1 = plot_vanishing_point(img1,vanishing_point1,vanishing_lines1)
hough_img2 = plot_vanishing_point(img2,vanishing_point2,vanishing_lines2)
hough_img3 = plot_vanishing_point(img3,vanishing_point3,vanishing_lines3)
hough_sunset = plot_vanishing_point(sunset,vanishing_point4,vanishing_lines4)

cv2.imshow('Original Image 1',img1)
cv2.imshow('Original Image 2',img2)
cv2.imshow('Original Image 3',img3)
cv2.imshow('Original Sunset', sunset)

cv2.imshow('Hough Image 1', hough_img1)
cv2.imshow('Hough Image 2', hough_img2)
cv2.imshow('Hough Image 3', hough_img3)
cv2.imshow('Hough Sunset', hough_sunset)

cv2.waitKey(0)
cv2.destroyAllWindows()