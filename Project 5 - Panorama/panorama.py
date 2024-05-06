# ---------------------------------------------
# Fichero: panorama.py
# ---------------------------------------------
# Escuela de Ingeniería y Arquitectura de Zaragoza
# Visión por Computador	
# 2023- 2024
#
# Félix Ozcoz Eraso             801108
# Victor Marcuello Baquero      741278
#
# Descripción: Fichero que contiene el programa
# que nos permite implementar un panorama
# en función de múltiples imágenes de entrada
# y mediante el uso de un detector de features.  
# ---------------------------------------------

import cv2 as cv
import numpy as np

def simple_panorama_stitching(img1,img2):
    
    return 0


# Función que elimina cualquier espacio en negro resultante de unir
# 2 o más imágenes para crear un nuevo panorama
def image_trimming(img):
    #crop top
    if not np.sum(img[0]):
        return image_trimming(img[1:])
    
    #crop bottom
    elif not np.sum(img[-1]):
        return image_trimming(img[:-1])
    
    #crop left
    elif not np.sum(img[:,0]):
        return image_trimming(img[:,1:]) 
    
    #crop right
    elif not np.sum(img[:,-1]):
        return image_trimming(img[:,:-1])    
    
    return image_trimming

def ransac_algorithm(matches,k1,k2,k_matches,n_iters,threshold,):
    
    return 0


def ransac_homography(matches, keypoints1, keypoints2, threshold=4):
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Use RANSAC to estimate homography
    homography, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, threshold)
    
    print(homography)

    return homography, mask.astype(bool)

def create_panorama1(img1,img2):
    # Detect keypoints and compute descriptors
    sift = cv.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Match descriptors
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.90 * n.distance:
            good_matches.append(m)

    # Run RANSAC to estimate homography
    homography, mask = ransac_homography(good_matches, keypoints1, keypoints2)

    # Warp images to create panorama
    height, width = img1.shape[:2]
    panorama = cv.warpPerspective(img1, homography, (width * 2, height))
    panorama[0:height, 0:width] = img2

    trim_panorama = image_trimming(panorama)    

    return trim_panorama
    
def create_panorama2(img1, img2):
    # Detect keypoints and compute descriptors using SIFT
    sift = cv.SIFT_create()
    keypoints1_sift, descriptors1_sift = sift.detectAndCompute(img1, None)
    keypoints2_sift, descriptors2_sift = sift.detectAndCompute(img2, None)

    # Detect keypoints and compute descriptors using ORB
    orb = cv.ORB_create()
    keypoints1_orb, descriptors1_orb = orb.detectAndCompute(img1, None)
    keypoints2_orb, descriptors2_orb = orb.detectAndCompute(img2, None)

    # Match descriptors for SIFT
    bf_sift = cv.BFMatcher()
    matches_sift = bf_sift.knnMatch(descriptors1_sift, descriptors2_sift, k=2)

    # Match descriptors for ORB
    bf_orb = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches_orb = bf_orb.match(descriptors1_orb, descriptors2_orb)

    # Apply ratio test for SIFT
    good_matches_sift = []
    for m, n in matches_sift:
        if m.distance < 0.9 * n.distance:
            good_matches_sift.append(m)

    # Apply ratio test for ORB
    good_matches_orb = []
    for m in matches_orb:
        if m.distance < 64:  # Adjust this threshold as needed
            good_matches_orb.append(m)

    # Combine good matches from both SIFT and ORB
    good_matches = good_matches_sift + good_matches_orb

    # Run RANSAC to estimate homography
    homography, mask = ransac_homography(good_matches, keypoints1_sift, keypoints2_sift)

    # Warp images to create panorama
    height, width = img1.shape[:2]
    panorama = cv.warpPerspective(img1, homography, (width * 2, height))
    panorama[0:height, 0:width] = img2

    print(panorama)

    # Trim the panorama
    trimmed_panorama = image_trimming(panorama)
    
    print(trimmed_panorama)

    return trimmed_panorama    

def general_panorama_function(image_paths):
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
    
    panorama = create_panorama2(img3,img1)
    
    #cv.imshow('Building Center',img3)
    #cv.imshow('Building Right Side',img1)
    
    print(panorama)
    #cv.imshow('Panorama',panorama)
    cv.waitKey()
    cv.destroyAllWindows()
    

path_img1 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building3.JPG'
path_img2 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building2.JPG'
path_img3 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building4.JPG'
path_img4 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building1.JPG'
path_img5 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building5.JPG'

image_paths = [path_img1,path_img2,path_img3,path_img4,path_img5]    

general_panorama_function(image_paths)    

