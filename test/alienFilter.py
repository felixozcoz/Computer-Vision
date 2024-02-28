# ---------------------------------------------
# Fichero: alienFilter.py
# ---------------------------------------------
# Escuela de Ingeniería y Arquitectura de Zaragoza
# Visión por Computador	
# 2023- 2024
#
# Félix Ozcoz Eraso     801108
# Victor Marcuello Baquero      741278
#
# Descripción:
#   Programa que aplica un filtro de color a la piel (alien filter)
#   y al fondo de la imagen capturada por la cámara.
# ---------------------------------------------

import cv2
import numpy as np

def alien_filter(frame, color):
    '''
        Apply a filter of color to the skin and the background of an image
        
        Parameters:
            frame: source image
            colour: RGB color for the skin 
        Output:
            result: result image
    '''
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define skin range color in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Define a mask for the skin
    skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Invert the skin mask to get the background
    background_mask = cv2.bitwise_not(skin_mask)

    # Apply the background mask to get the background
    background = cv2.bitwise_and(frame, frame, mask=background_mask)

    # Create an image with the skin
    skin_color = np.full(frame.shape, color, dtype=np.uint8)
    colored_skin = cv2.bitwise_and(skin_color, skin_color, mask=skin_mask)

    # Combine the colored skin and the background
    return cv2.add(colored_skin, background)



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

    # Cambiar el color de la piel a un color RGB específico (por ejemplo, verde)
    green_color = (220, 100, 100)
    frame_colored_skin_and_background = alien_filter(frame, green_color)

    # Mostrar el fotograma resultante
    cv2.imshow('Colored Skin and Background', frame_colored_skin_and_background)

    # Detener la ejecución si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
