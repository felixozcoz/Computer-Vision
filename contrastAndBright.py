import cv2
import numpy as np
import matplotlib.pyplot as plt


def contrastNbright(image, alpha, beta):
    new_image = np.zeros(image.shape, image.dtype)
    for idx,x in np.ndenumerate(image):
        new_image[idx] = np.clip(alpha * image[idx] + beta, 0, 255)
    return new_image


image = cv2.imread(r'C:\Users\Lenovo\OneDrive\Escritorio\lena_color.jpg', cv2.IMREAD_GRAYSCALE)
newImage = np.zeros(image.shape, image.dtype)

cv2.convertScaleAbs(image, newImage, 1, 40)

cv2.imshow('Original', image)
cv2.imshow('Mine', contrastNbright(image, 10, 40))
cv2.imshow('OpenCV', newImage)

cv2.waitKey(0)


