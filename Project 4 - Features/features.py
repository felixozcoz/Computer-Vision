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
# entre los diferentes detectores de features 
# disponibles en OpenCV
# ---------------------------------------------

import cv2 as cv
import numpy as np

# Función que nos permite calcular las features mediante emparejamiento de 
# fuerza bruta (todos con todos)
def brute_force_matcher1(des1,des2):
    brute_force_matcher = cv.BFMatcher(cv.NORM_L2,crossCheck=True)
    n_matches = brute_force_matcher.match(des1,des2)
    n_matches = sorted(n_matches,key=lambda x:x.distance)
    return n_matches

# Función que nos permite calcular las features mediante emparejamiento de
# fuerza bruta buscando el vecino más próximo y comprobando el ratio del 
# segundo vecino
def brute_force_matcher2(des1,des2):
    return 0

def flann_matcher(des1,des2):
    return 0

def matches_images_display(img1,img2,k1,k2,n_matches,top_matches):
    matches_img = cv.drawMatches(img1,k1,img2,k2,n_matches[:top_matches],None,flags=2)
    return matches_img

def harris_matching_function(img1,img2):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    harris = cv.cornerHarris(img1)
    
    k1_harris, des1_harris = harris.detectAndCompute(img1,None)
    k2_harris, des2_harris = harris.detectAndCompute(img2,None)
    
    n_matches_harris = brute_force_matcher1(des1_harris,des2_harris)
    features_matches_harris = len(n_matches_harris)

    matches_harris_img = matches_images_display(img1,img2,k1_harris,k2_harris,n_matches_harris,features_matches_harris) 
    
    cv.imshow("Matches Harris Image",matches_harris_img)


def orb_matching_function(img1,img2):
    orb = cv.ORB_create()
    
    k1_orb, des1_orb = orb.detectAndCompute(img1,None)
    k2_orb, des2_orb = orb.detectAndCompute(img2,None)
 
    n_matches_orb = brute_force_matcher2(des1_orb,des2_orb)
    
    matches_orb_img = matches_images_display(img1,img2,k1_orb,k2_orb,n_matches_orb,2)
    cv.imshow("Matches Orb Image",matches_orb_img)

def sift_matching_function(img1,img2):
    sift = cv.SIFT_create()
    
    k1_sift, des1_sift = sift.detectAndCompute(img1,None)
    k2_sift, des2_sift = sift.detectAndCompute(img2,None)
    
    n_matches_sift = brute_force_matcher1(des1_sift,des2_sift)
    
    matches_sift_img = matches_images_display(img1,img2,k1_sift,k2_sift,n_matches_sift,10) 
    cv.imshow("Matches Sift Image",matches_sift_img)

def akaze_matching_function(img1,img2):
    akaze = cv.AKAZE_create()
    
    k1_akaze, des1_akaze = akaze.detectAndCompute(img1,None)
    k2_akaze, des2_akaze = akaze.detectAndCompute(img2,None)
    
    n_matches_akaze = brute_force_matcher1(des1_akaze,des2_akaze)
    matches_akaze_img = matches_images_display(img1,img2,k1_akaze,k2_akaze,n_matches_akaze,2) 
    
    cv.imshow("Matches Akaze Image",matches_akaze_img)

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
    
    sift_matching_function(img1,img3)
    
    cv.waitKey()
    cv.destroyAllWindows()


path_img1 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building3.JPG'
path_img2 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building2.JPG'
path_img3 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building4.JPG'
path_img4 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building1.JPG'
path_img5 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building5.JPG'

image_paths = [path_img1,path_img2,path_img3,path_img4,path_img5]

image_features_matching_function(image_paths)