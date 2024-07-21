# Paquetería
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import entropy
from skimage.feature import graycomatrix
from scipy.stats import entropy
import pandas as pd

# IA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# Procesamiento de imagen
def process_hmiigr512(img, u_down, u_up):
    """
    Lee una imagen, la convierte a escala de grises y filtra los valores dentro de un umbral.

    Parámetros:
    - image_path: str, ruta de la imagen.
    - u_down: int, límite inferior del umbral.
    - u_up: int, límite superior del umbral.

    Retorna:
    - img: Imagen original.
    - gray_img: Imagen en escala de grises.
    - imbn: Imagen binaria filtrada dentro del umbral.
    """

    if img is None:
        raise FileNotFoundError(f"No found: {img}")

    # Limpieza de la etiqueta
    x, y, w, h = 7, 495, 223, 10
    vpixel = 0

    # Crear una copia de la imagen original
    img_modify = img.copy()

    # Cambiar los valores de los píxeles en el área definida
    img_modify[y:y+h, x:x+w] = vpixel

    # Convertir a escala de grises
    gray_img = cv2.cvtColor(img_modify, cv2.COLOR_BGR2GRAY)

    # Filtrar los valores dentro del umbral
    imbn = cv2.inRange(gray_img, u_down, u_up)

    return img_modify, gray_img, imbn


# Procesamiento de imagen
def process_mdiigr512(img, u_down, u_up):
    """
    Lee una imagen, la convierte a escala de grises y filtra los valores dentro de un umbral.

    Parámetros:
    - image_path: str, ruta de la imagen.
    - u_down: int, límite inferior del umbral.
    - u_up: int, límite superior del umbral.

    Retorna:
    - img: Imagen original.
    - gray_img: Imagen en escala de grises.
    - imbn: Imagen binaria filtrada dentro del umbral.
    """
    
    if img is None:
        raise FileNotFoundError(f"No found: {img}")

    # Limpieza de la etiqueta
    x, y, w, h = 1, 495, 183, 15
    vpixel = 0

    # Crear una copia de la imagen original
    img_modify = img.copy()

    # Cambiar los valores de los píxeles en el área definida
    img_modify[y:y+h, x:x+w] = vpixel

    # Convertir a escala de grises
    gray_img = cv2.cvtColor(img_modify, cv2.COLOR_BGR2GRAY)

    clean_gray = gray_img.copy()

    # Limpiar el contorno
    centro = (256, 256)  # Coordenadas (x, y)
    radio = 248  # Radio de la circunferencia
    valor_pixel = 255
    grosor = 3

    cv2.circle(clean_gray, centro, radio, valor_pixel, grosor)

    # Filtrar los valores dentro del umbral
    imbn = cv2.inRange(clean_gray, u_down, u_up)

    return img_modify, clean_gray, imbn


# Contador de pixel's
def count_value(arr, value):
    return np.sum(arr == value)


# Función de optimización de entropía
def calculate_glcm_entropy(image, distances, angles):
    best_entropy = -np.inf
    best_params = (None, None)
    
    for d in distances:
        for a in angles:
            glcm = graycomatrix(image, distances=[d], angles=[a], symmetric=True, normed=True)
            glcm_entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
            
            if glcm_entropy > best_entropy:
                best_entropy = glcm_entropy
                best_params = (d, a)
                
    return best_params, best_entropy


# Función para calcular la dimensión fractal usando el método de conteo de cajas
def box_count(img, k):
    S = np.add.reduceat(
        np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
        np.arange(0, img.shape[1], k), axis=1)
    return len(np.where((S > 0) & (S < k*k))[0])

def fractal_dimension(img):
    # Tamaños de las cajas en función del tamaño de la imagen
    min_size = 2
    max_size = min(img.shape) // 2
    sizes = np.logspace(np.log2(min_size), np.log2(max_size), num=10, base=2, dtype=int)
    sizes = sizes[sizes > 1]

    counts = []
    for size in sizes:
        count = box_count(img, size)
        if count > 0:  # Asegurarse de que count no sea cero
            counts.append(count)
        else:
            counts.append(1)  # Evitar división por cero

    # Verificar los valores calculados
    print(f"Tamaños de caja (S): {sizes}")
    print(f"Conteos (N): {counts}")

    if len(counts) < 2:
        return np.nan  # No se puede calcular la dimensión fractal con menos de dos tamaños de cajas válidos

    log_sizes = np.log(sizes)
    log_counts = np.log(counts)

    # Verificar los valores de logaritmos
    #print(f"Logaritmo de tamaños de caja (log S): {log_sizes}")
    #print(f"Logaritmo de conteos (log N): {log_counts}")

    # Calcular la dimensión fractal usando la relación directa
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return coeffs[0]

# Procesar una sola imagen y calcular la dimensión fractal utilizando la imagen binaria
def process_single_image(imbn):
    fractal_dim = fractal_dimension(imbn)
    img_resized = cv2.resize(imbn, (64, 64)).flatten()
    return fractal_dim, img_resized

def plot_image_processing(image, u_down, u_up):
  
    # Procesamiento
    image, gray_image, filter_image = process_mdiigr512(image, u_down, u_up)
    
    # Histograma
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    # Graficar
    plt.figure(figsize=(12.8, 9.6))

    plt.subplot(221)
    plt.imshow(gray_image, cmap="gray")
    plt.title("Escala de grises")

    plt.subplot(222)
    plt.plot(hist)
    plt.plot([u_down, u_down], [0, max(hist.flatten())], "r", label=f'Línea en {u_down}')
    plt.plot([u_up, u_up], [0, max(hist.flatten())], "r", label=f'Línea en {u_up}')
    plt.ylim([0, 100])
    plt.grid("on")
    plt.title("Ocurrencia de bines")

    plt.subplot(223)
    plt.imshow(filter_image, cmap="gray")
    plt.title("Sunspot")

    plt.show()