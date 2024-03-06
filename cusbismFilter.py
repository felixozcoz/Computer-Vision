import cv2
import numpy as np
import matplotlib.pyplot as plt

def cubism_filter(img, grid_size):
    
    # Definir el tamaño de los bloques
    block_size = img.shape[0] // grid_size

    # Crear una máscara de ceros del mismo tamaño que la imagen
    mask = np.zeros_like(img)
    
    # Aplicar el efecto de cubismo en cada región
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            block = img[i:i+block_size, j:j+block_size]
            mean_color = np.mean(block, axis=(0, 1))
            mask[i:i+block_size, j:j+block_size] = mean_color
    
    # Devuelve imagen resultante
    return mask

def add_contours(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detectar los bordes en la imagen
    edges = cv2.Canny(gray, 100, 200)
    
    # Dilatar los bordes para hacerlos más visibles
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Superponer los bordes sobre la imagen original
    result = cv2.bitwise_or(image, image, mask=dilated_edges)
    
    return result



def cubism_filter2(image_path, grid_size=20):
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro de suavizado para simplificar las formas
    blurred = cv2.medianBlur(gray, 7)
    
    # Detectar los bordes en la imagen suavizada
    edges = cv2.Canny(blurred, 100, 200)
    
    # Definir el tamaño de los bloques
    block_size = img.shape[0] // grid_size
    
    # Crear una máscara de ceros del mismo tamaño que la imagen
    mask = np.zeros_like(img)
    
    # Aplicar el efecto de cubismo en cada región
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            block = img[i:i+block_size, j:j+block_size]
            mean_color = np.mean(block, axis=(0, 1))
            mask[i:i+block_size, j:j+block_size] = mean_color
    
    # Fusionar la máscara con los bordes detectados
    result = cv2.bitwise_and(mask, mask, mask=edges)
    
    # Agregar líneas y contornos
    result_with_contours = add_contours(result)

    return result_with_contours


def cubism_filter3(image, block_size=10, n_iterations=5):

    # Get the height and width of the image
    height, width = image.shape[:2]

    # Create an empty image with the same dimensions as the input image
    cubism_image = np.zeros((height, width, 3), np.uint8)

    # Iterate over the input image in blocks of the specified size
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Extract the current block
            block = image[i:i+block_size, j:j+block_size]

            # Apply the cubism effect by iterating over the block several times
            for _ in range(n_iterations):
                # Compute the new coordinates of the block
                x, y = cubism_func(j, i, block_size)

                # Copy the block from the input image to the cubism image
                cubism_image[i:i+block_size, j:j+block_size] = image[y:y+block_size, x:x+block_size]

    # Save the cubism image
    return cubism_image

# Define the cubism function
def cubism_func(x, y, block_size=10):
    # Round the x and y coordinates down to the nearest multiple of the block size
    x = (x // block_size) * block_size
    y = (y // block_size) * block_size

    # Return the new coordinates
    return (x, y)

#cubism_filter(r'C:\Users\Lenovo\OneDrive\Escritorio\lena_color.jpg', r'C:\Users\Lenovo\OneDrive\Escritorio\lena_cubism.jpg')
#img  = cv2.imread(r'C:\Users\Lenovo\OneDrive\Escritorio\lena_color.jpg', cv2.IMREAD_COLOR)
img  = cv2.imread(r'C:\Users\usuario\Desktop\lena_color.png', cv2.IMREAD_COLOR)

# Crear una figura con dos subplots-*
fig, axs = plt.subplots(1, 3, figsize=(10, 5))

# Mostrar la primera imagen en el primer subplot
axs[0].imshow(cv2.cvtColor(cubism_filter(img, 20), cv2.COLOR_BGR2RGB))
axs[0].set_title('Cubism effect 1')

# Mostrar la segunda imagen en el segundo subplot
axs[1].imshow(cv2.cvtColor(cubism_filter2(img, 20), cv2.COLOR_BGR2RGB))
axs[1].set_title('Cubism effect 2')

# Mostrar la segunda imagen en el segundo subplot
axs[2].imshow(cv2.cvtColor(cubism_filter3(img, 20), cv2.COLOR_BGR2RGB))
axs[2].set_title('Cubism effect 3')

# Ajustar espacio entre subplots
plt.tight_layout()

# Mostrar la figura
plt.show()