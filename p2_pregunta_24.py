# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Memoria: codigo de la pregunta 2.4

# AUTOR1: Fernandez Moreno, Jose Luis
# AUTOR2: Ramasco Gorria, Pedro
# PAREJA/TURNO: Grupo 14

'''
P2.4 Extienda la función desarrollada en la tarea 3 para considerar la correspondencia de
puntos de interés con distancia mínima umbralizada alrededor del punto de interés (i.e.
tarea 3) junto con el criterio Nearest Neighborg Distance Ratio. Considere lo siguiente:
• La distancia utilizada siempre será euclídea
• Identificar el método con tipoDesc=‘nndr'
• Utilice un umbral con valor 0.75
Incluya ejemplos visuales de los resultados de la nueva funcionalidad desarrollada

'''
from p2_tarea2 import descripcion_puntos_interes
from p2_tarea1 import detectar_puntos_interes_harris
from p2_tarea3 import correspondencias_puntos_interes
from skimage import io , color
from matplotlib.pyplot import imread
from skimage.feature import plot_matches 
import matplotlib.pyplot as plt


def pinta(imagen1, imagen2, correspondencias, coordenadas_ima1, coordenadas_ima2,seleccion):
  
    if seleccion:
        titulo='hist con nndr'
    else:
        titulo='mag_ori con nndr'

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_matches(ax, imagen1, imagen2, coordenadas_ima1, coordenadas_ima2, correspondencias,only_matches=True)
    ax.axis("on")
    ax.set_title("Correspondecias de puntos de interés usando: "+titulo)
    plt.show()





#hemos implementado el criterio nndr dentro de la tarea3 aqui llamamos un par de imagenes para testear

dir="../practica2_tsv/img/"#directorio en el que se encuentran las imagenes

print('------------------Par de imagenes NotreDame------------------')
#Primero: leemos las dos primeras imagenes de la carpeta de test proporcionada en moodle
NotreDame1=color.rgb2gray(imread(dir+"NotreDame1.jpg"))
NotreDame2=color.rgb2gray(imread(dir+"NotreDame2.jpg"))

#Segundo: detectamos los puntos de interes con Harris
coords_esquinas_NotreDame1=detectar_puntos_interes_harris(NotreDame1, sigma = 1.5, k = 0.04, threshold_rel = 0.2) 
coords_esquinas_NotreDame2=detectar_puntos_interes_harris(NotreDame2, sigma = 1.5, k = 0.04, threshold_rel = 0.2) 

#Tercero: sacamos los descriptores hist y mag_ori a pardir de las coordenadas de las esquinas
descriptores_NotreDame1_hist,newcoords_esquinas_NotreDame1_hist=descripcion_puntos_interes(NotreDame1, coords_esquinas_NotreDame1, vtam = 16, nbins = 32, tipoDesc='hist')
descriptores_NotreDame2_hist,newcoords_esquinas_NotreDame2_hist=descripcion_puntos_interes(NotreDame2, coords_esquinas_NotreDame2, vtam = 16, nbins = 32, tipoDesc='hist')

descriptores_NotreDame1_mag_ori,newcoords_esquinas_NotreDame_mag_ori=descripcion_puntos_interes(NotreDame1, coords_esquinas_NotreDame1, vtam = 16, nbins = 32, tipoDesc='mag-ori')
descriptores_NotreDame2_mag_ori,newcoords_esquinas_NotreDame_mag_ori=descripcion_puntos_interes(NotreDame2, coords_esquinas_NotreDame2, vtam = 16, nbins = 32, tipoDesc='mag-ori')

#Cuarto: buscamos las correspondencias entre descriptores hist y mag_ori

correspondencias_NotreDame_hist=correspondencias_puntos_interes(descriptores_NotreDame1_hist, descriptores_NotreDame2_hist, tipoCorr='nndr',max_distancia=25)
correspondencias_NotreDame_mag_ori=correspondencias_puntos_interes(descriptores_NotreDame1_mag_ori, descriptores_NotreDame2_mag_ori, tipoCorr='nndr',max_distancia=25)

pinta(NotreDame1, NotreDame2, correspondencias_NotreDame_hist, newcoords_esquinas_NotreDame1_hist,newcoords_esquinas_NotreDame2_hist,True)
pinta(NotreDame1, NotreDame2, correspondencias_NotreDame_mag_ori, newcoords_esquinas_NotreDame_mag_ori,newcoords_esquinas_NotreDame_mag_ori,False)