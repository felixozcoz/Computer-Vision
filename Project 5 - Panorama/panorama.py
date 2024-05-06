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

def find_homography_ransac(src_points, dst_points, threshold=4, max_iters=1000):
    best_homography = None
    best_inliers = None
    max_inliers = 0
    
    for _ in range(max_iters):
        # Seleccionar puntos aleatorios para estimar la homografía inicial
        indices = np.random.choice(len(src_points), 4, replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]
        
        # Calcular la homografía inicial
        homography = calculate_homography(src_sample, dst_sample)
        
        # Calcular la distancia de reproyección y encontrar inliers
        distances = calculate_distances(src_points, dst_points, homography)
        print("Distances: ",distances)
        inliers = distances < threshold
        
        # Contar el número de inliers
        num_inliers = np.sum(inliers)
        
        print("Inliers (Num):",num_inliers)
        
        # Actualizar la mejor homografía si encontramos más inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_homography = homography
            best_inliers = inliers
    
    print("Homography:",best_homography)
    return best_homography, best_inliers

def calculate_homography(src_points, dst_points):
    # Verificar si src_points y dst_points tienen la forma adecuada
    if len(src_points.shape) != 3 or len(dst_points.shape) != 3 or src_points.shape[1:] != (1, 2) or dst_points.shape[1:] != (1, 2):
        raise ValueError("src_points y dst_points deben tener la forma (N, 1, 2)")
    
    # Construir la matriz de coeficientes A
    A = []
    for src, dst in zip(src_points, dst_points):
        x = src[0][0]
        y = src[0][1]
        xp = dst[0][0]
        yp = dst[0][1]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A)
    
    # Calcular la solución de mínimos cuadrados
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape(3, 3)
    
    return H / H[2, 2]  # Normalizar la homografía dividiendo por H[2, 2]

def calculate_distances(src_points, dst_points, homography):
    # Proyectar puntos fuente sobre el plano de destino utilizando la homografía
    src_points = src_points.reshape(-1, 2)  # Aplanar src_points si es tridimensional
    
    projected_points = np.dot(homography, np.vstack((src_points.T, np.ones(len(src_points)))))
    projected_points = projected_points[:2, :] / projected_points[2, :]
    projected_points = projected_points.T
    
    # Calcular la distancia euclidiana entre los puntos proyectados y los puntos de destino reales
    distances = np.linalg.norm(projected_points - dst_points, axis=1)
    
    return distances

def ransac_homography(matches, keypoints1, keypoints2, threshold=4, max_iters=1000):
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    homography, inliers = find_homography_ransac(src_points, dst_points, threshold=threshold, max_iters=max_iters)
    
    print(homography)
    
    if homography is None:
        raise ValueError("No se encontró ninguna homografía válida.")
    
    # Convertir inliers a formato booleano
    mask = np.zeros(len(matches), dtype=bool)
    mask[np.where(inliers)] = True
    
    homography = homography.astype(np.float32)
    
    return homography, mask
    
def create_panorama2(img1, img2):
    # Detect keypoints and compute descriptors using SIFT
    sift = cv.SIFT_create()
    keypoints1_sift, descriptors1_sift = sift.detectAndCompute(img1, None)
    keypoints2_sift, descriptors2_sift = sift.detectAndCompute(img2, None)

    if len(keypoints1_sift) < 4 or len(keypoints2_sift) < 4:
        print("No hay suficientes puntos clave detectados en una o ambas imágenes.")
        return None

    # Match descriptors for SIFT
    bf_sift = cv.BFMatcher()
    matches_sift = bf_sift.knnMatch(descriptors1_sift, descriptors2_sift, k=2)

    # Apply ratio test for SIFT
    good_matches_sift = []
    for m, n in matches_sift:
        if m.distance < 0.50 * n.distance:
            good_matches_sift.append(m)

    # Run RANSAC to estimate homography
    homography, mask = ransac_homography(good_matches_sift, keypoints1_sift, keypoints2_sift)

    # Warp images to create panorama
    height, width = img1.shape[:2]
    panorama = cv.warpPerspective(img1, homography, (width * 2, height))
    panorama[0:height, 0:width] = img2
      
    return panorama

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
    
    panorama = create_panorama2(img1,img2)
    panorama2 = create_panorama2(panorama,img3)
    panorama3 = create_panorama2(panorama2,img4)
    panorama4 = create_panorama2(panorama3,img5)
    
    #print(panorama)
    cv.imshow('Panorama',panorama4)
    cv.waitKey()
    cv.destroyAllWindows()
    

path_img1 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building3.JPG' # Parte central del edificio
path_img2 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building2.JPG' # Parte izquierda del edificio
path_img3 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building4.JPG' # Parte derecha del edificio
path_img4 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building1.JPG' # Parte extrema izquierda del edificio
path_img5 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building5.JPG' # Parte extrema derecha del edificio
image_paths = [path_img1,path_img2,path_img3,path_img4,path_img5]    

general_panorama_function(image_paths)    

