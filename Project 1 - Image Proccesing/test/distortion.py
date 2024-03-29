# ---------------------------------------------
# Fichero: distortion.py
# ---------------------------------------------
# Escuela de Ingeniería y Arquitectura de Zaragoza
# Visión por Computador	
# 2023- 2024
#
# Félix Ozcoz Eraso             801108
# Victor Marcuello Baquero      741278
#
# Descripción:
#   Programa que aplica un filtro de distorsión 
#   a la imagen capturada por la cámara.
# ---------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings


def geometric_distortion(image, k1, k2=0.0):
    '''
        Apply the barrel distortion filter

        Parameters:
            - image: source image (numpy array format)
            - k1: coefficient of distortion 1
            - k2: coefficient de distortion 2
            k = 0 --> not effect
            k < 0 --> pincushion distortion
            k < 0 --> barrel distortion
    '''

    [ydim, xdim] = image.shape[:2]
    cx, cy = xdim // 2, ydim // 2  # define distortion centre

    x, y = np.meshgrid(np.arange(xdim), np.arange(ydim))  # define the grid to manipulate the image

    x, y = x - cx, y - cy   # centering the image
    
    # regularize the origin coordinates (0, 0) to the center of the image (cx, cy) 
    # in order to avoid division by zero. This value maintains the original position of the pixels
    x[cx, cy], y[cx, cy] = 1, 1  

    # calculate radius
    r = np.sqrt(x ** 2 + y ** 2)

    # calculate distorted radius
    r_distorted = r * (1 + k1 * r ** 2 + k2 * r ** 4)

    # calculate distorte coordinates
    x_dis = x * (r_distorted / r) + cx
    y_dis = y * (r_distorted / r) + cy

    # remap image
    return cv2.remap(image, x_dis.astype(np.float32), y_dis.astype(np.float32), cv2.INTER_LINEAR)



# ----- Test the function -----


def plot_comparision(image, k1_barrel, k2_barrel, k1_pincushion, k2_pincushion):
    # Crear una figura con dos subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Mostrar la primera imagen en el primer subplot
    axs[0].imshow(cv2.cvtColor(geometric_distortion(image, k1_barrel, k2_barrel), cv2.COLOR_BGR2RGB))
    axs[0].set_title('Barrel distortion')

    # Mostrar la segunda imagen en el segundo subplot
    axs[1].imshow(cv2.cvtColor(geometric_distortion(image, k1_pincushion, k2_pincushion), cv2.COLOR_BGR2RGB))
    axs[1].set_title('Pincushion distortion')

    # Ajustar espacio entre subplots
    plt.tight_layout()

    # Mostrar la figura
    plt.show()


# Test the function with a sample image
def test_geometric_distortion_sample_image(image, k1, k2):
    cv2.imshow('DistortionFilter', geometric_distortion(image, k1, k2))
    cv2.waitKey(0)


# Test the function with video capture
def test_geometric_distortion_video_capture(k1, k2):
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
        cv2.imshow('DistortionFilter', geometric_distortion(frame, 0.0, 0.0))

        # Detener la ejecución si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()


# ---- Main ----

image = cv2.imread(r'C:\Users\Lenovo\OneDrive\Escritorio\lena_color.jpg', cv2.IMREAD_COLOR)

# visualización de distorsiones aisladas
#test_geometric_distortion_sample_image(image, 0.00001, 0.0)
#test_geometric_distortion_sample_image(image, -0.000005, 0.0)

# comparación de efectors de distorsión
plot_comparision(image, 0.00001, 0.0, -0.000005, 0.0)