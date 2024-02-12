# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Memoria: codigo de la pregunta 2.1

# AUTOR1: Fernandez Moreno, Jose Luis
# AUTOR2: Ramasco Gorria, Pedro
# PAREJA/TURNO: Grupo 14


# librerias y paquetes por defecto
import numpy as np
# Incluya aqui las librerias que necesite en su codigo
from scipy import ndimage
import skimage.feature
from p2_tarea2 import descripcion_puntos_interes
from p2_tarea1 import detectar_puntos_interes_harris
import matplotlib.pyplot as plt

'''
P2.1 Aplicando los descriptores de tipo ‘hist’ de la tarea 2 sobre la imagen camera() de
Skimage (paquete skimage.data, https://bit.ly/2ZsXfN7), analice como cambian los valores
del descriptor calculado si utiliza:
• Tamaño de vecindario con valores 8 y 16
• Número de bins con valores 16 y 32
'''


imagen=skimage.data.camera()#leemos la imagen del paquete skimage
coords_esquinas=detectar_puntos_interes_harris(imagen, sigma = 1.0, k = 0.05, threshold_rel = 0.2)#aplicamos la funcion de deteccion de puntos de interes si alterar los parametros por defecto

#A continuacion vamos a calcular los descriptores variando el tamaño de las ventanas y los bins
descriptores_vtam8_bins16, new_coords_esquinas_vtam8_bins16=descripcion_puntos_interes(imagen, coords_esquinas, vtam = 8, nbins = 16, tipoDesc='hist')
descriptores_vtam8_bins32, new_coords_esquinas_vtam8_bins32=descripcion_puntos_interes(imagen, coords_esquinas, vtam = 8, nbins = 32, tipoDesc='hist')
descriptores_vtam16_bins16, new_coords_esquinas_vtam16_bins16=descripcion_puntos_interes(imagen, coords_esquinas, vtam = 16, nbins = 16, tipoDesc='hist')
descriptores_vtam16_bins32, new_coords_esquinas_vtam16_bins32=descripcion_puntos_interes(imagen, coords_esquinas, vtam = 16, nbins = 32, tipoDesc='hist')

#Hacemos plots de los descriptores calculados

print('\n------------------- VTAM = 8, NBINS = 16 -------------------\n')
print('\nDESCRIPTORES\n')
print(descriptores_vtam8_bins16)
plt.hist(descriptores_vtam8_bins16)
plt.title('Descriptores si VTAM = 8, NBINS = 16 ')
plt.show()

print('\n------------------- VTAM = 8, NBINS = 32 -------------------\n')
print('\nDESCRIPTORES\n')
print(descriptores_vtam8_bins32)
plt.hist(descriptores_vtam8_bins32)
plt.title('Descriptores si VTAM = 8, NBINS = 32 ')
plt.show()

print('\n------------------- VTAM = 16, NBINS = 16 -------------------\n')
print('\nDESCRIPTORES\n')
print(descriptores_vtam16_bins16)
plt.hist(descriptores_vtam16_bins16)
plt.title('Descriptores si VTAM = 16, NBINS = 16 ')
plt.show()

print('\n------------------- VTAM = 16, NBINS = 32 -------------------\n')
print('\nDESCRIPTORES\n')
print(descriptores_vtam16_bins32)
plt.hist(descriptores_vtam16_bins32)
plt.title('Descriptores si VTAM = 16, NBINS = 32 ')
plt.show()