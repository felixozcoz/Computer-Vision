# ---------------------------------------------
# Fichero: features.py
# ---------------------------------------------
# Escuela de Ingeniería y Arquitectura de Zaragoza
# Visión por Computador	
# 2023- 2024
#
# Félix Ozcoz Eraso             801108
# Victor Marcuello Baquero      741278
#
# Descripción: Fichero que contiene el programa
# que nos permite realizar el estudio comparativo
# entre los diferentes detectores de atributos 
# disponibles en OpenCV
# ---------------------------------------------

import cv2 as cv
import numpy as np
import time

# Función que nos permite calcular las features mediante emparejamiento de 
# fuerza bruta (todos con todos)
def brute_force_matcher1(des1,des2):
    start_time = time.time()
    
    brute_force_matcher = cv.BFMatcher(cv.NORM_L2)
    matches = brute_force_matcher.match(des1,des2)
    matches = sorted(matches,key=lambda x:x.distance)
    
    end_time = time.time()  # Finaliza el contador de tiempo
    execution_time = end_time - start_time  # Calcula el tiempo de ejecución
    print("Tiempo de ejecución de brute_force_matcher1:", execution_time, "segundos")
    
    return matches

# Función que nos permite calcular las features mediante emparejamiento de
# fuerza bruta buscando el vecino más próximo y comprobando el ratio del 
# segundo vecino
def brute_force_matcher2(des1,des2,ratio_threshold):
    start_time = time.time()
    brute_force_matcher = cv.BFMatcher(cv.NORM_L2)
    matches = brute_force_matcher.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    end_time = time.time()  # Finaliza el contador de tiempo
    execution_time = end_time - start_time  # Calcula el tiempo de ejecución
    print("Tiempo de ejecución de brute_force_matcher2:", execution_time, "segundos")  

    return good_matches

# Función que nos permite calcular las features mediante el emparejamiento
# realizado a través de la función FLANN buscando el vecino más próximo y 
# comprobando el ratio del segundo vecino
def flann_matcher(des1,des2,ratio_threshold):
    start_time = time.time()
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann_matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann_matcher.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    end_time = time.time()  # Finaliza el contador de tiempo
    execution_time = end_time - start_time  # Calcula el tiempo de ejecución
    print("Tiempo de ejecución de brute_force_matcher1:", execution_time, "segundos")

    return good_matches

# Función que devuelve una imagen que contiene los features en común que tienen ambas imágenes
def matches_images_display(img1,img2,k1,k2,matches,top_matches):
    matches_img = cv.drawMatches(img1,k1,img2,k2,matches[:top_matches],None,flags=2)
    return matches_img 

# Funciones que permiten la detección de features en una imagen
def harris_features_detection(img):
    start_time = time.time()
    
    dst = cv.cornerHarris(img,blockSize=2,ksize=3,k=0.04)
    dst = cv.dilate(dst,None)
    
    feature_img = np.copy(img)
    mask = dst > 0.01 * dst.max()
    feature_img[mask] = 0
    
    end_time = time.time()  # Finaliza el contador de tiempo
    execution_time = end_time - start_time  # Calcula el tiempo de ejecución
    print("Tiempo de ejecución de harris_matching_function:", execution_time, "segundos")
    
    cv.imshow("Feature Image (Harris)",feature_img)
    
    return feature_img

def orb_features_detection(img,n_features):
    start_time = time.time()
    orb = cv.ORB_create(n_features)
    
    k_orb, des_orb = orb.detectAndCompute(img,None)
    feature_img = cv.drawKeypoints(img,k_orb,None,color=(0,0,255),flags=0)
    
    end_time = time.time()  # Finaliza el contador de tiempo
    execution_time = end_time - start_time  # Calcula el tiempo de ejecución
    print("Tiempo de ejecución de orb_matching_function:", execution_time, "segundos")
    
    cv.imshow("Feature Image (Orb)",feature_img)
    
    return feature_img

def sift_features_detection(img,n_features):
    start_time = time.time()
    sift = cv.SIFT_create(n_features)
    
    k_sift, des_sift = sift.detectAndCompute(img,None)
    feature_img = cv.drawKeypoints(img,k_sift,None,color=(0,0,255),flags=0)
    
    end_time = time.time()  # Finaliza el contador de tiempo
    execution_time = end_time - start_time  # Calcula el tiempo de ejecución
    print("Tiempo de ejecución de harris_matching_function:", execution_time, "segundos")
    
    cv.imshow("Feature Image (Sift)",feature_img)
    
    return feature_img

def akaze_features_detection(img):
    start_time = time.time()
    akaze = cv.AKAZE_create()
    
    k_akaze, des_akaze = akaze.detectAndCompute(img,None)
    feature_img = cv.drawKeypoints(img,k_akaze,None,color=(0,0,255),flags=0)
    
    end_time = time.time()  # Finaliza el contador de tiempo
    execution_time = end_time - start_time  # Calcula el tiempo de ejecución
    print("Tiempo de ejecución de harris_matching_function:", execution_time, "segundos")
    
    cv.imshow("Feature Image (Akaze)",feature_img)
    
    return feature_img

def harris_matching_function(img1,img2,n_features,n_matches,desc_type):
    start_time = time.time()
    img_float1 = np.float32(img1)
    img_float2 = np.float32(img2)
    
    harris1 = cv.cornerHarris(img_float1,blockSize=2,ksize=3,k=0.04)
    harris2 = cv.cornerHarris(img_float2,blockSize=2,ksize=3,k=0.04)
    
    # Convert corner response to keypoints
    threshold = 0.01 * harris1.max()
    k1_harris = np.argwhere(harris1 > threshold)
    k2_harris = np.argwhere(harris2 > threshold)
    
    kps1_harris = [cv.KeyPoint(np.float32(x[1]), np.float32(x[0]), 10) for x in k1_harris]
    kps2_harris = [cv.KeyPoint(np.float32(x[1]), np.float32(x[0]), 10) for x in k2_harris]
    
    if desc_type == "orb":
        orb = cv.ORB_create(n_features)
        _,des1_harris = orb.compute(img1,kps1_harris)
        _,des2_harris = orb.compute(img2,kps2_harris)
        
    elif desc_type == "sift":
        sift = cv.SIFT_create(n_features)
        _,des1_harris = sift.compute(img1,kps1_harris)
        _,des2_harris = sift.compute(img2,kps2_harris)
        
    else: 
        print("Descriptor incorrecto")
        exit(1)
    
    matches_harris1 = brute_force_matcher1(des1_harris,des2_harris)
    matches_harris2 = brute_force_matcher2(des1_harris,des2_harris,threshold)

    matches_harris_img1 = matches_images_display(img1,img2,kps1_harris,kps2_harris,matches_harris1,n_matches) 
    matches_harris_img2 = matches_images_display(img1,img2,kps1_harris,kps2_harris,matches_harris2,n_matches)
    
    cv.imshow("Matches Harris Image",matches_harris_img1)
    cv.imshow("Matches Harris Image",matches_harris_img2)
    
    end_time = time.time()  # Finaliza el contador de tiempo
    execution_time = end_time - start_time  # Calcula el tiempo de ejecución
    print("Tiempo de ejecución de harris_matching_function:", execution_time, "segundos")
    print("Número de atributos detectados (imagen 1): ", len(k1_harris))
    print("Número de atributos detectados (imagen 2): ", len(k2_harris))
    print("Número de emparejamientos detectados (emparejador 1): ", len(matches_harris1))
    print("Número de emparejamientos detectados (emparejador 2): ", len(matches_harris2))


def orb_matching_function(img1,img2,n_features,n_matches,threshold):
    start_time = time.time()
    orb = cv.ORB_create(n_features)
    
    k1_orb, des1_orb = orb.detectAndCompute(img1,None)
    k2_orb, des2_orb = orb.detectAndCompute(img2,None)
 
    matches_orb1 = brute_force_matcher1(des1_orb,des2_orb)
    matches_orb2 = brute_force_matcher2(des1_orb,des2_orb,threshold)
    
    matches_orb_img1 = matches_images_display(img1,img2,k1_orb,k2_orb,matches_orb1,n_matches)
    matches_orb_img2 = matches_images_display(img1,img2,k1_orb,k2_orb,matches_orb2,n_matches)
    
    cv.imshow("Matches Orb Image",matches_orb_img1)
    cv.imshow("Matches Orb Image",matches_orb_img2)
    
    end_time = time.time()  # Finaliza el contador de tiempo
    execution_time = end_time - start_time  # Calcula el tiempo de ejecución
    print("Tiempo de ejecución de orb_matching_function:", execution_time, "segundos")
    print("Número de atributos detectados (imagen 1): ", len(k1_orb))
    print("Número de atributos detectados (imagen 2): ", len(k2_orb))
    print("Número de emparejamientos detectados (emparejador 1): ", len(matches_orb1))
    print("Número de emparejamientos detectados (emparejador 2): ", len(matches_orb2))

def sift_matching_function(img1,img2,n_features,n_matches,threshold):
    start_time = time.time() 
    sift = cv.SIFT_create(n_features)
    
    k1_sift, des1_sift = sift.detectAndCompute(img1,None)
    k2_sift, des2_sift = sift.detectAndCompute(img2,None)
    
    matches_sift1 = brute_force_matcher1(des1_sift,des2_sift)
    matches_sift2 = brute_force_matcher2(des1_sift,des2_sift,threshold)
    
    matches_sift_img1 = matches_images_display(img1,img2,k1_sift,k2_sift,matches_sift1,n_matches)
    matches_sift_img2 = matches_images_display(img1,img2,k1_sift,k2_sift,matches_sift2,n_matches)
    
    cv.imshow("Matches Sift Image",matches_sift_img1)
    cv.imshow("Matches Sift Image",matches_sift_img2)

    end_time = time.time()  # Finaliza el contador de tiempo
    execution_time = end_time - start_time  # Calcula el tiempo de ejecución
    print("Tiempo de ejecución de sift_matching_function:", execution_time, "segundos")
    print("Número de atributos detectados (imagen 1): ", len(k1_sift))
    print("Número de atributos detectados (imagen 2): ", len(k2_sift))
    print("Número de emparejamientos detectados (emparejador 1): ", len(matches_sift1))
    print("Número de emparejamientos detectados (emparejador 2): ", len(matches_sift2))

def akaze_matching_function(img1,img2,n_matches,threshold):
    start_time = time.time() 
    akaze = cv.AKAZE_create()
    
    k1_akaze, des1_akaze = akaze.detectAndCompute(img1,None)
    k2_akaze, des2_akaze = akaze.detectAndCompute(img2,None)
    
    matches_akaze1 = brute_force_matcher1(des1_akaze,des2_akaze)
    matches_akaze2 = brute_force_matcher2(des1_akaze,des2_akaze,threshold)
    
    matches_akaze_img1 = matches_images_display(img1,img2,k1_akaze,k2_akaze,matches_akaze1,n_matches)
    matches_akaze_img2 = matches_images_display(img1,img2,k1_akaze,k2_akaze,matches_akaze2,n_matches)
    
    cv.imshow("Matches Akaze Image",matches_akaze_img1)
    cv.imshow("Matches Akaze Image",matches_akaze_img2)
    
    end_time = time.time()  # Finaliza el contador de tiempo
    execution_time = end_time - start_time  # Calcula el tiempo de ejecución
    
    print("Tiempo de ejecución de akaze_matching_function:", execution_time, "segundos")
    print("Número de atributos detectados (imagen 1): ", len(k1_akaze))
    print("Número de atributos detectados (imagen 2): ", len(k2_akaze))
    print("Número de emparejamientos detectados (emparejador 1): ", len(matches_akaze1))
    print("Número de emparejamientos detectados (emparejador 2): ", len(matches_akaze2))

def image_features_matching_function(image_paths):
    
    src1 = cv.imread(image_paths[0])
    src2 = cv.imread(image_paths[1]) 
    src3 = cv.imread(image_paths[2])
    src4 = cv.imread(image_paths[3])
    src5 = cv.imread(image_paths[4])
    
    img1 = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)
    img3 = cv.cvtColor(src3, cv.COLOR_BGR2GRAY)
    img4 = cv.cvtColor(src4, cv.COLOR_BGR2GRAY)
    img5 = cv.cvtColor(src5, cv.COLOR_BGR2GRAY)
    
    #harris_features_detection(img1)
    #orb_features_detection(img1,5000)
    sift_features_detection(img1,5000)
    #akaze_features_detection(img1)
    
    #harris_matching_function(img1,img2,100,100,"orb")
    #harris_matching_function(img1,img2,100,100,"sift")
    #orb_matching_function(src1,src2,5000,100,0.90)
    sift_matching_function(src1,src2,5000,100,0.90)
    #akaze_matching_function(src1,src2,100,0.90)
    
    cv.waitKey()
    cv.destroyAllWindows()


path_img1 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building1.JPG'
path_img2 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building2.JPG'
path_img3 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building3.JPG'
path_img4 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building4.JPG'
path_img5 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building5.JPG'

image_paths = [path_img1,path_img2,path_img3,path_img4,path_img5]

image_features_matching_function(image_paths)