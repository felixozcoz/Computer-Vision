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

# Función que aplica un filtro de color a la piel y al fondo de una imagen
# Parámetros:
#   frame: imagen de entrada
#   color: color RGB para la piel
# Salida:
#   result: imagen resultante
def alien_filter(frame, color):
    # Convertir el fotograma a espacio de color HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir rangos de color de piel en HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Crear una máscara para la piel
    skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Invertir la máscara de piel para obtener el fondo
    background_mask = cv2.bitwise_not(skin_mask)

    # Aplicar la máscara de fondo para obtener el fondo
    background = cv2.bitwise_and(frame, frame, mask=background_mask)

    # Crear una imagen con el color de piel
    skin_color = np.full(frame.shape, color, dtype=np.uint8)
    colored_skin = cv2.bitwise_and(skin_color, skin_color, mask=skin_mask)

    # Combinar la piel coloreada y el fondo
    result = cv2.add(colored_skin, background)

    return result



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
