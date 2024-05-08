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

def brute_force_matcher2(des1,des2,ratio_threshold):
    brute_force_matcher = cv.BFMatcher(cv.NORM_L2)
    matches = brute_force_matcher.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return good_matches

def matches_images_display(img1,img2,k1,k2,matches,top_matches):
    matches_img = cv.drawMatches(img1,k1,img2,k2,matches[:top_matches],None,flags=2)
    return matches_img 

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
        #print("Distances: ",distances)
        inliers = distances < threshold
        
        # Contar el número de inliers
        num_inliers = np.sum(inliers)
        
        #print("Inliers (Num):",num_inliers)
        
        # Actualizar la mejor homografía si encontramos más inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_homography = homography
            best_inliers = inliers
    
    #print("Homography:",best_homography)
    return best_homography, best_inliers


def calculate_homography(src_points, dst_points):
    # Verificar si src_points y dst_points tienen la forma adecuada
    if len(src_points.shape) != 3 or len(dst_points.shape) != 3 or src_points.shape[1:] != (1, 2) or dst_points.shape[1:] != (1, 2):
        raise ValueError("src_points y dst_points deben tener la forma (N, 1, 2)")
    
    # Construir la matriz de coeficientes A
    A = []
    for src, dst in zip(src_points, dst_points):
        #x, y = src
        #xp, yp = dst
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
    
    #homography, inliers = find_homography_ransac(src_points, dst_points, threshold=threshold, max_iters=max_iters)
    homography, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 10.0)
    
    if homography is None:
        raise ValueError("No se encontró ninguna homografía válida.")
    
    # Convertir inliers a formato booleano
    #mask = np.zeros(len(matches), dtype=bool)
    #mask[np.where(inliers)] = True
    
    homography = homography.astype(np.float32)
    
    return homography, mask
    
# Función que permite crear un simple panorama a partir de 2 imágenes    
def create_single_panorama(img1, img2, threshold):
    # Detect keypoints and compute descriptors using SIFT
    sift = cv.SIFT_create()
    keypoints1_sift, descriptors1_sift = sift.detectAndCompute(img1, None)
    keypoints2_sift, descriptors2_sift = sift.detectAndCompute(img2, None)

    if len(keypoints1_sift) < 4 or len(keypoints2_sift) < 4:
        print("No hay suficientes puntos clave detectados en una o ambas imágenes.")
        return None

    # Match descriptors for SIFT 
    good_matches_sift = brute_force_matcher2(descriptors1_sift,descriptors2_sift,threshold)

    # Run RANSAC to estimate homography
    homography, _ = ransac_homography(good_matches_sift, keypoints1_sift, keypoints2_sift)
    
    panorama = cv.warpPerspective(img1, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    panorama[0:img2.shape[0], 0:img2.shape[1]] = img2
    
    return panorama
    

def create_multiple_panorama(images, threshold):
    if len(images) < 2:
        print("Se requieren al menos dos imágenes para crear un panorama.")
        return None
    
    keypoints_list = []
    descriptors_list = []

    # Detect keypoints and compute descriptors for all images
    sift = cv.SIFT_create()
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    
    panorama = images[0]
    for i in range(1, len(images)):
        img1 = panorama
        img2 = images[i]

        keypoints1_sift = keypoints_list[i-1]
        descriptors1_sift = descriptors_list[i-1]
        keypoints2_sift = keypoints_list[i]
        descriptors2_sift = descriptors_list[i]

        if len(keypoints1_sift) < 4 or len(keypoints2_sift) < 4:
            print("No hay suficientes puntos clave detectados en una o ambas imágenes.")
            return None

        # Match descriptors for SIFT 
        good_matches_sift = brute_force_matcher2(descriptors1_sift, descriptors2_sift, threshold)

        # Run RANSAC to estimate homography
        homography, mask = ransac_homography(good_matches_sift, keypoints1_sift, keypoints2_sift)
        
        # Warp the second image onto the panorama
        panorama_shape = (img1.shape[1] + img2.shape[1], img1.shape[0])
        warped_img2 = cv.warpPerspective(img2, homography, panorama_shape)

    return panorama

def general_panorama_function(image_paths):
    src1 = cv.imread(image_paths[0])
    src2 = cv.imread(image_paths[1])
    src3 = cv.imread(image_paths[2])
    src4 = cv.imread(image_paths[3])
    src5 = cv.imread(image_paths[4])
    #src_images = [src1,src2,src3,src4,src5]
    src_images = [src2,src1]
    
    img1 = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)
    img3 = cv.cvtColor(src3, cv.COLOR_BGR2GRAY)
    img4 = cv.cvtColor(src4, cv.COLOR_BGR2GRAY)
    img5 = cv.cvtColor(src5, cv.COLOR_BGR2GRAY)
    
    panorama = create_single_panorama(src5,src4,0.95)
    #panorama = create_multiple_panorama(src_images,0.95)
    
    #print(panorama)
    cv.imshow('Panorama',panorama)
    cv.waitKey()
    cv.destroyAllWindows()
    

path_img1 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building1.JPG' # Parte extrema izquierda del edificio
path_img2 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building2.JPG' # Parte izquierda del edificio
path_img3 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building3.JPG' # Parte central del edificio
path_img4 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building4.JPG' # Parte derecha del edificio
path_img5 = 'C:\\Users\\usuario\\Desktop\\BuildingScene\\building5.JPG' # Parte extrema derecha del edificio
image_paths = [path_img1,path_img2,path_img3,path_img4,path_img5]    

general_panorama_function(image_paths)    

