# ---------------------------------------------
# Fichero: filterMenu.py
# ---------------------------------------------
# Escuela de Ingeniería y Arquitectura de Zaragoza
# Visión por Computador
# 2023- 2024
#
# Félix Ozcoz Eraso     801108
# Victor Marcuello Baquero      741278
#
# Descripción:
#   Programa interactivo que permite la
#   implementación de múltiples filtros a
#   cualquier captura de imagen proveniente
#   de una webcam
# ---------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
#from alienFilter import alien_filter
#from distortion import geometric_distortion
#from contrastAndBright import contrastNbright
#from posterFilter import colorReduce

def test_filter_menu():
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

        # if cv2.waitKey(1) & 0xFF == ord('c'):
        #    cv2.imshow('ConstrastAndBrightFilter',contrastNbright(frame,10,40))
        # if cv2.waitKey(1) & 0xFF == ord('a'):
        #    green_color = (220, 100, 100)
        #    frame_colored_skin_and_background = alien_filter(frame, green_color)
        #    cv2.imshow('Colored Skin and Background', frame_colored_skin_and_background)
        # if cv2.waitKey(1) & 0xFF == ord('d'):
        #    cv2.imshow('DistortionFilter', geometric_distortion(frame, 0.0, 0.0))
        # if cv2.waitKey(1) & 0xFF == ord('p'):
        #    cv2.imshow('PosterFilter', colorReduce(frame))
        # Detener la ejecución si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()
    return 0

test_filter_menu()



