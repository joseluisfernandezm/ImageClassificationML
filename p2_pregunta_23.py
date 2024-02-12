# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Memoria: codigo de la pregunta 2.3

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
from p2_tarea3 import correspondencias_puntos_interes
from skimage import io , color
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from skimage.feature import plot_matches 
import random #lo usamos para pruebas

'''

P2.3 Aplique las funciones de las tareas 1, 2 y 3 sobre los pares de imágenes
proporcionados en la carpeta ‘img’. Para cada par de imágenes, discuta las diferencias en
las esquinas devueltas (si existieran) y qué fenómenos pueden explicar las diferencias
observadas. Incluya ejemplos visuales de los experimentos que realice. 

'''


#esta funcion hace el plot de las correspondencias entre el par de imagenes
def pinta(imagen1, imagen2, correspondencias, coordenadas_ima1, coordenadas_ima2,seleccion):
    '''
    # Codigo implementado por nosotros en el caso en el que la funcion plot_matches este prohibida, aunque no funciona del todo bien
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Mostrar la imagen 1 en el subplot 1
    ax1.imshow(imagen1,cmap='gray')
    ax1.set_title('Imagen 1')

    # Mostrar la imagen 2 en el subplot 2
    ax2.imshow(imagen2,cmap='gray')
    ax2.set_title('Imagen 2')

    for correspondencia in correspondencias:
    # Convertir puntos a coordenadas de la figura
        x1, y1 = coordenadas_ima1[correspondencia[0]]
        x2, y2 = coordenadas_ima2[correspondencia[1]]

        colores = plt.cm.viridis(random.random())
        
        # Dibujar la flecha
        ax1.scatter(x1, y1, marker='x',color=colores,s=100)
        ax2.scatter(x2, y2, marker='x',color=colores,s=100)

    plt.tight_layout()
    plt.show()
    '''
    #if para poner el titulo a las graficas
    if seleccion:
        titulo='hist'
    else:
        titulo='mag_ori'

    fig, ax = plt.subplots(nrows=1, ncols=1)#creamos subplots
    plot_matches(ax, imagen1, imagen2, coordenadas_ima1, coordenadas_ima2, correspondencias,only_matches=True)#con only_matches enseñamos solo los puntos de interes que presentan correspondencia
    ax.axis("on")
    ax.set_title("Correspondecias de puntos de interés usando: "+titulo)
    plt.show()
    



dir="../practica2_tsv/img/"#directorio en el que se encuentran las imagenes

print('------------------Par de imagenes Egaudi------------------')
#Primero: leemos las dos primeras imagenes de la carpeta de test proporcionada en moodle
Gaudi1=color.rgb2gray(imread(dir+"EGaudi_1.jpg"))
Gaudi2=color.rgb2gray(imread(dir+"EGaudi_2.jpg"))

#Segundo: detectamos los puntos de interes con Harris
coords_esquinas_Gaudi1=detectar_puntos_interes_harris(Gaudi1, sigma = 1.0, k = 0.05, threshold_rel = 0.2) 
coords_esquinas_Gaudi2=detectar_puntos_interes_harris(Gaudi2, sigma = 1.0, k = 0.05, threshold_rel = 0.2) 

#Tercero: sacamos los descriptores hist y mag_ori a pardir de las coordenadas de las esquinas
descriptores_Gauid1_hist,newcoords_esquinas_Gauid1_hist=descripcion_puntos_interes(Gaudi1, coords_esquinas_Gaudi1, vtam = 8, nbins = 16, tipoDesc='hist')
descriptores_Gauid2_hist,newcoords_esquinas_Gauid2_hist=descripcion_puntos_interes(Gaudi2, coords_esquinas_Gaudi2, vtam = 8, nbins = 16, tipoDesc='hist')

descriptores_Gauid1_mag_ori,newcoords_esquinas_Gauid1_mag_ori=descripcion_puntos_interes(Gaudi1, coords_esquinas_Gaudi1, vtam = 8, nbins = 16, tipoDesc='mag-ori')
descriptores_Gauid2_mag_ori,newcoords_esquinas_Gauid2_mag_ori=descripcion_puntos_interes(Gaudi2, coords_esquinas_Gaudi2, vtam = 8, nbins = 16, tipoDesc='mag-ori')

#Cuarto: buscamos las correspondencias entre descriptores hist y mag_ori

correspondencias_Gaudi_hist=correspondencias_puntos_interes(descriptores_Gauid1_hist, descriptores_Gauid2_hist, tipoCorr='mindist',max_distancia=25)
correspondencias_Gaudi_mag_ori=correspondencias_puntos_interes(descriptores_Gauid1_mag_ori, descriptores_Gauid2_mag_ori, tipoCorr='mindist',max_distancia=25)

pinta(Gaudi1, Gaudi2, correspondencias_Gaudi_hist, newcoords_esquinas_Gauid1_hist,newcoords_esquinas_Gauid2_hist,True)
pinta(Gaudi1, Gaudi2, correspondencias_Gaudi_mag_ori, newcoords_esquinas_Gauid1_mag_ori,newcoords_esquinas_Gauid2_mag_ori,False)

print('------------------Par de imagenes Mount_Rushmore------------------')
#Primero: leemos las dos primeras imagenes de la carpeta de test proporcionada en moodle
Mount_Rushmore1=color.rgb2gray(imread(dir+"Mount_Rushmore1.jpg"))
Mount_Rushmore2=color.rgb2gray(imread(dir+"Mount_Rushmore2.jpg"))

#Segundo: detectamos los puntos de interes con Harris
coords_esquinas_Mount_Rushmore1=detectar_puntos_interes_harris(Mount_Rushmore1, sigma = 1.0, k = 0.05, threshold_rel = 0.2) 
coords_esquinas_Mount_Rushmore2=detectar_puntos_interes_harris(Mount_Rushmore2, sigma = 1.0, k = 0.05, threshold_rel = 0.2) 

#Tercero: sacamos los descriptores hist y mag_ori a pardir de las coordenadas de las esquinas
descriptores_Mount_Rushmore1_hist,newcoords_esquinas_Mount_Rushmore1_hist=descripcion_puntos_interes(Mount_Rushmore1, coords_esquinas_Mount_Rushmore1, vtam = 8, nbins = 16, tipoDesc='hist')
descriptores_Mount_Rushmore2_hist,newcoords_esquinas_Mount_Rushmore2_hist=descripcion_puntos_interes(Mount_Rushmore2, coords_esquinas_Mount_Rushmore2, vtam = 8, nbins = 16, tipoDesc='hist')

descriptores_Mount_Rushmore1_mag_ori,newcoords_esquinas_Mount_Rushmore1_mag_ori=descripcion_puntos_interes(Mount_Rushmore1, coords_esquinas_Mount_Rushmore1, vtam = 8, nbins = 16, tipoDesc='mag-ori')
descriptores_Mount_Rushmore2_mag_ori,newcoords_esquinas_Mount_Rushmore2_mag_ori=descripcion_puntos_interes(Mount_Rushmore2, coords_esquinas_Mount_Rushmore2, vtam = 8, nbins = 16, tipoDesc='mag-ori')

#Cuarto: buscamos las correspondencias entre descriptores hist y mag_ori

correspondencias_Mount_Rushmore_hist=correspondencias_puntos_interes(descriptores_Mount_Rushmore1_hist, descriptores_Mount_Rushmore2_hist, tipoCorr='mindist',max_distancia=25)
correspondencias_Mount_Rushmore_mag_ori=correspondencias_puntos_interes(descriptores_Mount_Rushmore1_mag_ori, descriptores_Mount_Rushmore2_mag_ori, tipoCorr='mindist',max_distancia=25)

pinta(Mount_Rushmore1, Mount_Rushmore2, correspondencias_Mount_Rushmore_hist, newcoords_esquinas_Mount_Rushmore1_hist,newcoords_esquinas_Mount_Rushmore2_hist,True)
pinta(Mount_Rushmore1, Mount_Rushmore2, correspondencias_Mount_Rushmore_mag_ori, newcoords_esquinas_Mount_Rushmore1_mag_ori,newcoords_esquinas_Mount_Rushmore2_mag_ori,False)

print('------------------Par de imagenes NotreDame------------------')
#Primero: leemos las dos primeras imagenes de la carpeta de test proporcionada en moodle
NotreDame1=color.rgb2gray(imread(dir+"NotreDame1.jpg"))
NotreDame2=color.rgb2gray(imread(dir+"NotreDame2.jpg"))

#Segundo: detectamos los puntos de interes con Harris Nota: hemos modificado los parametros de la funcion para obtener resultados más robustos aunque sacrifiquemos algunos puntos de interes que se quedaran sin detectar
coords_esquinas_NotreDame1=detectar_puntos_interes_harris(NotreDame1, sigma = 1.5, k = 0.04, threshold_rel = 0.2) 
coords_esquinas_NotreDame2=detectar_puntos_interes_harris(NotreDame2, sigma = 1.5, k = 0.04, threshold_rel = 0.2) 

#Tercero: sacamos los descriptores hist y mag_ori a pardir de las coordenadas de las esquinas Nota:hemos optado por emplear un tamaño de ventana mayor al predeterminado y con el doble de vtam para los nbins
descriptores_NotreDame1_hist,newcoords_esquinas_NotreDame1_hist=descripcion_puntos_interes(NotreDame1, coords_esquinas_NotreDame1, vtam = 16, nbins = 32, tipoDesc='hist')
descriptores_NotreDame2_hist,newcoords_esquinas_NotreDame2_hist=descripcion_puntos_interes(NotreDame2, coords_esquinas_NotreDame2, vtam = 16, nbins = 32, tipoDesc='hist')

descriptores_NotreDame1_mag_ori,newcoords_esquinas_NotreDame_mag_ori=descripcion_puntos_interes(NotreDame1, coords_esquinas_NotreDame1, vtam = 16, nbins = 32, tipoDesc='mag-ori')
descriptores_NotreDame2_mag_ori,newcoords_esquinas_NotreDame_mag_ori=descripcion_puntos_interes(NotreDame2, coords_esquinas_NotreDame2, vtam = 16, nbins = 32, tipoDesc='mag-ori')

#Cuarto: buscamos las correspondencias entre descriptores hist y mag_ori

correspondencias_NotreDame_hist=correspondencias_puntos_interes(descriptores_NotreDame1_hist, descriptores_NotreDame2_hist, tipoCorr='mindist',max_distancia=25)
correspondencias_NotreDame_mag_ori=correspondencias_puntos_interes(descriptores_NotreDame1_mag_ori, descriptores_NotreDame2_mag_ori, tipoCorr='mindist',max_distancia=25)

pinta(NotreDame1, NotreDame2, correspondencias_NotreDame_hist, newcoords_esquinas_NotreDame1_hist,newcoords_esquinas_NotreDame2_hist,True)
pinta(NotreDame1, NotreDame2, correspondencias_NotreDame_mag_ori, newcoords_esquinas_NotreDame_mag_ori,newcoords_esquinas_NotreDame_mag_ori,False)