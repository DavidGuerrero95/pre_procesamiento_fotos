# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:10:31 2022

@author: Daniel
"""
import numpy as np
import cv2
import copy
from scipy import ndimage as ndi
from skimage.feature import match_template

""" 
*******************************************************************************
--------- 1.Funciones ---------------------------------------------------------
*******************************************************************************
"""

"""Funcion para clasificar por area"""


def areaFun(a):
    li50 = 9
    ls50 = 10
    # Moneda de 100
    li100 = 13
    ls100 = 14
    # Moneda de 200
    li200 = 16
    ls200 = 17.4
    # Moneda de 500
    li500 = 18
    ls500 = 19.5
    # Moneda de 1000
    li1000 = 23
    ls1000 = 25.2
    if (li50 < a < ls50):
        value = 50
    elif (li100 < a < ls100):
        value = 100
    elif (li200 < a < ls200):
        value = 200
    elif (li500 < a < ls500):
        value = 500
    elif (li1000 < a < ls1000):
        value = 1000
    else:
        value = 0
    return value


"""Funcion para individualizar objetos"""


def individualiza_obj(img_bw, img_bgr):
    marco = np.nonzero(img_bw > 0)
    filas = marco[0]
    colum = marco[1]
    fm = min(filas)
    fx = max(filas)
    cm = min(colum)
    cx = max(colum)
    img_obj = img_bgr[fm:fx, cm:cx, :]
    return img_obj


"""Fun algoritmo bwareaopen tomado de 
#       https://stackoverflow.com/questions/2348365/matlab-bwareaopen-equivalent-function-in-opencv
"""


def bwareaopen(img, min_size, connectivity=8):
    """Remove small objects from binary image (approximation of
    bwareaopen in Matlab for 2D images).

    Args:
        img: a binary image (dtype=uint8) to remove small objects from
        min_size: minimum size (in pixels) for an object to remain in the image
        connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).

    Returns:
        the binary image with small objects removed
    """

    # Find all connected components (called here "labels")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=connectivity)

    # check size of all connected components (area in pixels)
    for i in range(num_labels):
        label_size = stats[i, cv2.CC_STAT_AREA]

        # remove connected components smaller than min_size
        if label_size < min_size:
            img[labels == i] = 0

    return img


"""Función para pasar a escala de Grises"""


def grayFun(a):
    b = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return b


"""Función para Binarizar con Othsu"""


def binFun(a):
    ret, binary = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


"""Funcion para Aplicar dilatación"""


def dilateFun(kernel, img, i):
    dilate = cv2.dilate(img, kernel, iterations=i)
    return dilate


"""Funcion para Aplicar Erosion"""


def erodeFun(kernel, img, i):
    erode = cv2.erode(img, kernel, iterations=i)
    return erode


"""Funcion para clasificar por color"""


def classColorFun(a):
    a = cv2.GaussianBlur(a, (5, 5), 0)  # Filtro Gausiano, suavisar imágen, eliminar ruido
    g = a.copy()  # Copia de respsldo
    g = grayFun(g)  # Pasar moneda a escala de grises
    g = binFun(g)  # Binarizar por Othsu
    g = g / 255  # Normalizar
    Area = np.sum(g)  # Cálculo de area de moneda

    LAB = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)  # Pasar a espacio de color LAB
    L, A, B = cv2.split(LAB)  # Seccioanr las capas de color L, A,B
    A[g == 0] = 0  # demarcar la zona sin información en la cada capa.
    B[g == 0] = 0

    # Limites Para la capa A, filtrar color Plateado
    li2 = 120
    ls2 = 147

    # Limites para la capa B, Filtrar color Dorado.
    li3 = 144
    ls3 = 167

    # Aplicar Lilites a la capa A y binarizar
    A[li2 > A] = 0
    A[ls2 < A] = 0
    A[A > 0] = 1

    # Aplicar límites a la capa B y Binarizar
    B[li3 > B] = 0
    B[ls3 < B] = 0
    B[B > 0] = 1

    # Demarcar el color plateado. para las monedas que tienen dos colores
    A = A - B

    # Areas de color Dorado
    dorado = np.sum(B)
    # Areas de color Plateado                                                       #
    plata = np.sum(A)

    # Ponderación de color sobre el área total
    dorado = dorado / Area
    plata = plata / Area

    # Variable para clasificar por color
    color = 0

    "Clasificación por color"
    # Monedas Doradas
    if (dorado > 0.8):
        color = 0
    # Monedas Plateadas
    elif (plata > 0.8):
        color = 1

    # Monedas plateadas y doradas.
    elif (dorado > 0.35) and (plata > 0.35):
        color = 2

    return color


"Función para correlación"


def corrFun(Temp, Coin):
    pila = Temp.copy() * 0  # Copia Vacia del templete.

    # Grados de paso de rotación
    g = 45
    step = int(360 / g)  # Cantidad de rotaciones

    for i, r in zip(range(0, step), range(0, 360, g)):
        Mrotate = ndi.rotate(Coin, r)  # Rotación de la moneda, max corrlación
        result = match_template(Temp, Mrotate)  # Aplicar Correlación en cada rotación
        result = cv2.resize(result, (1280, 720), interpolation=cv2.INTER_AREA)  # Redimensionar resultados
        pila = np.vstack((pila, result))  # Apilado verticarl de resultados
        print(r, result.max())
        # FIN DEL FOR DE CORRELACIOn

    return pila


"""Funcion para clasificar Por Correlación"""


# En relación a la posición donde ocurre el maximo de correlación, se determina el valor de la moneda
def corrValue(x):
    if (170 < x < 220):
        value = 1000  # Moneda de Mil
    elif (440 < x < 500):
        value = 500  # Moneda de Quinientos
    elif (690 < x < 740):
        value = 200  # Moneda de Doscientos
    elif (910 < x < 970):
        value = 100  # Monedas de cien
    elif (1100 < x < 1150):
        value = 50  # Moneda de cincuenta
    else:
        value = 0  # Para valor no correlacionado.
    return value


"Función para clasificar globalmente segun caracteristicas"


def coinFun(a, b, c):
    coin = 0
    if (a == 1000) and (b == 1000) and (c == 2 or c == 1 or c == 0):
        coin = 1000  # Moneda de Mil
    elif (a == 500) and (b == 500) and (c == 2 or c == 1 or c == 0):
        coin = 500  # Moneda de Quinientos
    elif (a == 200) and (b == 200) and (c == 1):
        coin = 200  # Moneda de Doscientos
    elif (a == 100) and (b == 100) and (c == 0):
        coin = 100  # Monedas de cien
    elif (a == 50) and (b == 50) and (c == 1):
        coin = 50  # Moneda de cincuenta
    else:
        coin = 0  # Para valor no correlacionado.

    return coin


"Funcion clase en dataframe para frutas"


def class_fruit(index):
    if 0 <= index < 250:
        class_df = 0 #Aguacate
        print("Aguacate")
    elif 250 <= index < 500:
        class_df = 1 #Banano
        print("Banano")
    elif 500 <= index < 750:
        class_df = 2 # Fresa
        print("Fresa")
    elif 750 <= index < 1000:
        class_df = 3 # Limon
        print("Limon")
    elif 1000 <= index < 1250:
        class_df = 4 # Mango
        print("Mango")
    elif 1250 <= index < 1500:
        class_df = 5 # Manzana
        print("Manzana")
    elif 1500 <= index < 1750:
        class_df = 6 # Pera
        print("Pera")
    elif 1750 <= index < 2000:
        class_df = 7  # Tomate
        print("Tomate")
    elif 2000 <= index < 2250:
        class_df = 8  # Uchuva
        print("Uchuva")
    elif 2250 <= index < 2500:
        class_df = 9  # Uva
        print("Uva")
    return class_df


def translate(lista):
    fruta = ''
    for i in lista:
        if i == 0:
            fruta = 'aguacate'
        if i == 1:
            fruta = 'banano'
        if i == 2:
            fruta = 'fresa'
        if i == 3:
            fruta = 'limon'
        if i == 4:
            fruta = 'mango'
        if i == 5:
            fruta = 'manzana'
        if i == 6:
            fruta = 'pera'
        if i == 7:
            fruta = 'tomate'
        if i == 8:
            fruta = 'uchuva'
        if i == 9:
            fruta = 'uva'
    return fruta