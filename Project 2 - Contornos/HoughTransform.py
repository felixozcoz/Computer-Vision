import cv2 as cv
import numpy as np
import operators as op


def vanishing_point_detector(path, threshold=100):
    '''
        Detect the vanishing point of an image using the Hough Transform

        Parameters:
            src (numpy array): Source image
            threshold (int): Threshold for the gradient

        Returns:
            numpy array: Image with the vanishing point
    '''
    src = cv.imread(path)

    # Convert image to grayscale
    img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    h, w = img.shape

    # Apply the Sobel filter
    _, _, gradient, orientation = op.Sobel_filter(img)

    # Threshold the gradient
    _, thresh = cv.threshold(gradient, threshold, 255, cv.THRESH_BINARY)

    # Indices of white pixels
    h, w = img.shape

    # Indices of white pixels
    y_idx, x_idx = np.nonzero(thresh) # rows, columns
    diff = np.radians(5)              # 5 degrees margin

    # Threshold the points that are not vertical or horizontal to zero
    for ind in range(len(y_idx)):
        # Get keypoint indices in the image
        i = y_idx[ind]
        j = x_idx[ind]

            # vertical margin   
        a = (-diff < orientation[i, j] < diff)                 
        b = (-np.pi-diff < orientation[i, j] < -np.pi+diff)     
        c = (np.pi-diff < orientation[i, j] < np.pi+diff)
            # horizontal margin   
        d = (np.pi/2-diff < orientation[i, j] < np.pi/2+diff)
        e = (-np.pi/2-diff < orientation[i, j] < -np.pi/2+diff)

        if a | b | c | d | e:
            thresh[i,j] = 0

    # Indices of white pixels
    y_idx, x_idx = np.nonzero(thresh) # filas, columnas

    accumulator, pts = _vote_line(h, w, y_idx, x_idx, orientation)

    final_img = _plot_vanishing_point(src, accumulator, pts)

    # _, final_img = cv.imencode('.jpg', img_plot)

    return final_img



def _vote_line(h, w, y_idx, x_idx, orientation):
    '''
        Vote for the lines in the Hough space

        Parameters:
            h (int): Height of the image
            w (int): Width of the image
            y_idx (numpy array): Rows of the image whos values are different from zero
            x_idx (numpy array): Columns of the image whos values are different from zero
            orientation (numpy array): Orientation of the gradient

        Returns:
            numpy array: Accumulator of the votes
            list: List of points that voted for each line
    
    '''
    # Definitions
    accumulator = np.zeros(w, dtype=int)    # counting votes
    pts = [set() for _ in range(w)]         # store points that voted for each line
    
    # Lines voting process in the Hough space
    for ind in range(len(y_idx)):
        # Get keypoint indices in the image
        i = y_idx[ind]
        j = x_idx[ind]

        # Extract the orientation of the gradient
        theta = orientation[i, j]   
        
        # Transform the image coordinates to the Hough space (centering the origin)
        x = j - w/2
        y = h/2 - i

        # Line equation in the Hough space
        th_sin, th_cos = np.sin(theta), np.cos(theta)
        rho = x * th_cos + y * th_sin

        # Find the x coordinate in the intersection with the y axis in the Hough space ( y = 0 )    
        x_to_find = rho/th_cos 
        voted_x = int(x_to_find + w/2)

        # Check if the voted x is inside the image
        if 0 <= voted_x < w:
            accumulator[voted_x] += 1
            pts[voted_x].add((j,i))

    return accumulator, pts


def _plot_vanishing_point(src, accumulator, pts):
    '''
        Plot the vanishing point in the image

        Parameters:
            src (numpy array): Source image
            accumulator (numpy array): Accumulator of the votes
            pts (list): List of points that voted for each line
            h (int): Height of the image
            w (int): Width of the image

        Returns:
            numpy array: Image with the vanishing point
    '''
    h, _ = src.shape[:2]

    # Find the 5 most voted lines
    ind, _ = _k_maximos_con_indices(accumulator, 5)

    # Plot the most voted points in the horizontal axis
    # for i in ind:
    #     cv.circle(src, (i, h//2), 2, (0, 255, 0), 2)    # Puntos más votados

    # Plot the most voted line
    max_ind = ind[0]
    for p in pts[max_ind]:
        cv.line(src, (p[0], p[1]), (max_ind, h//2), (255, 0, 0), 1)
        cv.circle(src, p, 2, (0, 0, 255), 2)    # points that voted for the most voted point

    cv.circle(src, (max_ind, h//2), 2, (0, 0, 0), 2)  # most voted point

    return src


def _k_maximos_con_indices(arreglo, k):
    '''
        Obtain the K maximum values and their indices

        Parameters:
            arreglo (numpy array): Array of values
            k (int): Number of maximum values to obtain
        
        Returns:
            tuple: Indices of the K maximum values, K maximum values
    '''
    # order the values in descending order
    indices_descendentes = np.argsort(arreglo)[::-1]
    
    # Obtain the K maximum values
    k_indices_maximos = indices_descendentes[:k]
    
    # Obtain the K maximum values and their indices
    k_valores_maximos = arreglo[k_indices_maximos]
    
    return k_indices_maximos, k_valores_maximos


def convertir_a_numpy(imagen):
    if isinstance(imagen, np.ndarray):
        # Si la imagen ya es un numpy array, simplemente devolverla
        return imagen
    else:
        # Si la imagen es de otro tipo (por ejemplo, PIL Image), convertirla a un numpy array
        imagen_np = np.array(imagen)
        return imagen_np
    

# ---------------------------------------------
    
# src = cv.imread(r"C:\Users\felix\OneDrive\Escritorio\otra.jpg")
# cv.imshow("Vanishing Point", vanishing_point_detector(r"C:\Users\felix\OneDrive\Escritorio\otra.jpg", 150))
# cv.waitKey(0)