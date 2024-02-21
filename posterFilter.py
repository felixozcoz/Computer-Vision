import numpy as np
import cv2

def colorReduce2(img_in, div=np.uint8(64)):
    img2 = img_in.copy()
    for idx, x in np.ndenumerate(img_in):  # one or three channels
        img2[idx] = np.uint8(np.uint8(x // div)*div)
    return img2


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
    cv2.imshow('PosterFilter', colorReduce2(frame))

    # Detener la ejecución si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
