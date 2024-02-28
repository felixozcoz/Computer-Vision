# ---------------------------------------------
# Fichero: posterFilter.py
# ---------------------------------------------
# Escuela de Ingeniería y Arquitectura de Zaragoza
# Visión por Computador	
# 2023- 2024
#
# Félix Ozcoz Eraso     801108
# Victor Marcuello Baquero      741278
#
# Descripción:
#   Programa que aplica un filtro de posterización 
#   a la imagen capturada por la cámara.
# ---------------------------------------------

import numpy as np
import cv2
import matplotlib.pyplot as plt

def posterization_filter(img_in, div=np.uint8(64)):
    '''
        Apply a posterization filter to an image
        
        Parameters:
            image: source image
            div: divisor for color reduction
        
        Output:
            img_result: result image
    '''
    img_result = img_in.copy()

    for idx, x in np.ndenumerate(img_in): 
        img_result[idx] = np.uint8(np.uint8(x // div)*div)

    return img_result


def plot_comparision(img, *args):
    num_figures = len(args)

    # Crear una figura con dos subplots
    fig, axs = plt.subplots(1, num_figures, figsize=(10, 5))

    for i in range(0, num_figures):
        # Mostrar la imagen i en el subplot i
        axs[i].imshow(cv2.cvtColor(posterization_filter(img, args[i]), cv2.COLOR_BGR2RGB))
        axs[i].set_title('Poster effect ' + str(i))
   
    # Ajustar espacio entre subplots
    plt.tight_layout()

    # Mostrar la figura
    plt.show()




plot_comparision(cv2.imread(r'C:\Users\Lenovo\OneDrive\Escritorio\lena_color.jpg', cv2.IMREAD_COLOR), 32, 64, 100)
exit(0)

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
    cv2.imshow('PosterFilter', posterization_filter(frame))

    # Detener la ejecución si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
