# ---------------------------------------------
# Fichero: filters.py
# ---------------------------------------------
# Escuela de Ingeniería y Arquitectura de Zaragoza
# Visión por Computador	
# 2023- 2024
#
# Félix Ozcoz Eraso             801108
# Victor Marcuello Baquero      741278
#
# Descripción:
#   Módulo que contiene las funciones de 
#   procesamiento de imágenes (filtros)
# ---------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt



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

    # calculate radius
    r = np.sqrt(x ** 2 + y ** 2)
    r[cx, cy] = 1  # avoid division by zero

    # calculate distorted radius
    r_distorted = r * (1 + k1 * r ** 2 + k2 * r ** 4)
    
    # calculate distorte coordinates
    x_dis = x * (r_distorted / r) + cx
    y_dis = y * (r_distorted / r) + cy

    # remap image
    return cv2.remap(image, x_dis.astype(np.float32), y_dis.astype(np.float32), cv2.INTER_LINEAR)



def contrastNbright(image, alpha, beta):
    '''
        Apply a filter of contrast and brightness to an image

        Parámeters:
            image: source image
            alpha: contrast factor
            beta: brightness factor

        Output:
            new_image: retult image
    '''
    # define destination image
    new_image = np.zeros(image.shape, image.dtype)

    for idx, x in np.ndenumerate(image):
        new_image[idx] = np.clip(alpha * image[idx] + beta, 0, 255)

    return new_image




def alien_filter(frame, color=(220, 100, 100)):
    '''
        Apply a filter of color to the skin and the background of an image
        
        Parameters:
            frame: source image
            color: RGB color for the skin 
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
    
    # Create an image with the skin
    skin_color = np.full(frame.shape, color[::-1], dtype=np.uint8)
    
    # Combine the colored skin and the background
    result = cv2.bitwise_and(skin_color, skin_color, mask=skin_mask)
    
    # Invert the skin mask to get the background
    background_mask = cv2.bitwise_not(skin_mask)
    
    # Apply the background mask to get the background
    background = cv2.bitwise_and(frame, frame, mask=background_mask)
    
    # Combine the colored skin and the background
    result = cv2.add(result, background)
    
    return result



def posterization_filter(frame, div=np.uint8(64)):
    '''
        Apply a posterization filter to an image
        
        Parameters:
            image: source image
            div: divisor for color reduction
        
        Output:
            img_result: result image
    '''
    img_result = (frame // div) * div

    return img_result

#   Optional image processing functions

def gaussian_blurring(frame,delta):

    img_result = cv2.GaussianBlur(frame,delta)
    return img_result

def median_blurring(frame,delta):

    img_result = cv2.medianBlur(frame,delta)
    return img_result

def kaleidoscope_filter(frame,invert,rotation_angle=np.uint8(90)):
    ht, wd = frame.shape[:2]

    # transpose the image
    framet = cv2.transpose(frame)

    # create diagonal bi-tonal mask
    mask = np.zeros((ht,wd),dtype=np.uint8)
    points = np.array([[[0, 0],[wd, 0],[wd, ht]]])
    cv2.fillPoly(mask,points,255)
    if invert == "yes":
        mask = cv2.bitwise_not(mask)

    # composite frame and framet using mask
    compA = cv2.bitwise_and(framet,framet,mask=mask)
    compB = cv2.bitwise_and(frame,frame,mask=255-mask)
    comp = cv2.add(compA,compB)

    # rotate composite
    if rotation_angle == 90:
        comp = cv2.rotate(comp,cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        comp = cv2.rotate(comp,cv2.ROTATE_180)
    elif rotation_angle == 270:
        comp = cv2.rotate(comp, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # mirror (flip) horizontally
    mirror = cv2.flip(comp,1)

    # concatenate horizontally
    top = np.hstack((comp,mirror))

    # mirror (flip) vertically
    bottom = cv2.flip(top,0)

    # concatenate vertically
    kaleidoscope = np.vstack((top,bottom))

    # resize
    kaleidoscope_result = cv2.resize(kaleidoscope, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    return kaleidoscope_result