# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 3:  Similitud y correspondencia de puntos de interes

# AUTOR1: Fernandez Moreno, Jose Luis
# AUTOR2: Ramasco Gorria, Pedro
# PAREJA/TURNO: Grupo 14


# librerias y paquetes por defecto
import numpy as np
from p2_tests import test_p2_tarea3

# Incluya aqui las librerias que necesite en su codigo
# ...

def correspondencias_puntos_interes(descriptores_imagen1, descriptores_imagen2, tipoCorr='mindist',max_distancia=25):
    """
    # Esta funcion determina la correspondencias entre dos conjuntos de descriptores mediante
    # el calculo de la similitud entre los descriptores.
    #
    # El parametro 'tipoCorr' determina el criterio de similitud aplicado 
    # para establecer correspondencias entre pares de descriptores:
    #   - Criterio 'mindist': minima distancia euclidea entre descriptores 
    #                         menor que el umbral 'max_distancia'
    #  
    # Argumentos de entrada:
    #   - descriptores1: numpy array con dimensiones [numero_descriptores, longitud_descriptor] 
    #                    con los descriptores de los puntos de interes de la imagen 1.        
    #   - descriptores2: numpy array con dimensiones [numero_descriptores, longitud_descriptor] 
    #                    con los descriptores de los puntos de interes de la imagen 2.        
    #   - tipoCorr: cadena de caracteres que indica el tipo de criterio para establecer correspondencias
    #   - max_distancia: valor de tipo double o float utilizado por el criterio 'mindist' y 'nndr', 
    #                    que determina si se aceptan correspondencias entre descriptores 
    #                    con distancia minima menor que 'max_distancia' 
    #
    # Argumentos de salida
    #   - correspondencias: numpy array con dimensiones [numero_correspondencias, 2] de tipo int64 
    #                       que determina correspondencias entre descriptores de imagen 1 e imagen 2.
    #                       Por ejemplo: 
    #                       correspondencias[0,:]=[5,22] significa que el descriptor 5 de la imagen 1 
    #                                                  corresponde con el descriptor 22 de la imagen 2. 
    #                       correspondencias[1,:]=[6,23] significa que el descriptor 6 de la imagen 1 
    #                                                  corresponde con el descriptor 23 de la imagen 2.
    #
    # NOTA: no modificar los valores por defecto de las variables de entrada tipoCorr y max_distancia, 
    #       pues se utilizan para verificar el correcto funciomaniento de esta funcion
    #
    # CONSIDERACIONES: 
    # 1) La funcion no debe permitir correspondencias de uno a varios descriptores. Es decir, 
    #   un descriptor de la imagen 1 no puede asignarse a multiples descriptores de la imagen 2 
    # 2) En el caso de que existan varios descriptores de la imagen 2 con la misma distancia minima 
    #    con algún descriptor de la imagen 1, seleccione el descriptor de la imagen 2 con 
    #    indice/posicion menor. Por ejemplo, si las correspondencias [5,22] y [5,23] tienen la misma
    #    distancia minima, seleccione [5,22] al ser el indice 22 menor que 23
    """    
    correspondencias = np.empty(shape=[0,2]) # iniciamos la variable de salida (numpy array)        
   
    #incluya su codigo aqui
    #En esta parte vamos a seguir la estrategia 3 de las diapositivas
    if tipoCorr=='mindist':
        filas_descriptor1min = min(descriptores_imagen1.shape[0],descriptores_imagen2.shape[0])#vemos cual de las filas de los descriptores es menor, esto nos vendra bien para el calculo de distancias 2 a 2

        correspondencias = np.zeros((filas_descriptor1min, 2), dtype = 'int64')#inicializamos la matriz de correspondencias
        distancias_euclideas = np.zeros((descriptores_imagen1.shape[0], descriptores_imagen2.shape[0]))#inicializamos la matriz de distancias euclideas
        distancias_euclideas=np.linalg.norm(descriptores_imagen1[:, np.newaxis, :]-descriptores_imagen2,axis=2)#np.newaxis es como usar un reshape y np.linalg.norm te hace la distancia euclidea
            
        for i in range (0, filas_descriptor1min):
            minimo = np.min(distancias_euclideas[i,:])#sacamos la menor de las distancias euclideas
            pos = np.where(distancias_euclideas[i,:]== minimo)#buscamos el minimo en distancias euclideas
            indice = pos[0][0]#gruardamos el indice
            if (minimo <= max_distancia):#comprobamos el umbral, si esta por debajo lo metemos en correspondencias
                correspondencias[i, :] = [i, indice]
                distancias_euclideas[:,indice]= max_distancia + 1

    if tipoCorr=='nndr':
        # El paso 1 seria es de calcular la distania euclidea entre los descriptores asociados al punto de inteeres como hicimos en mindist
        filas_descriptor1min = min(descriptores_imagen1.shape[0],descriptores_imagen2.shape[0])#vemos cual de las filas de los descriptores es menor, esto nos vendra bien para el calculo de distancias 2 a 2
        correspondencias = np.zeros((filas_descriptor1min, 2), dtype = 'int64')#inicializamos la matriz de correspondencias
        distancias_euclideas = np.zeros((descriptores_imagen1.shape[0], descriptores_imagen2.shape[0]))#inicializamos la matriz de distancias euclideas
        distancias_euclideas=np.linalg.norm(descriptores_imagen1[:, np.newaxis, :]-descriptores_imagen2,axis=2)#np.newaxis es como usar un reshape y te hace la distancia euclidea
        
        # El paso 2 es el de seleccionar la correspondencia entre descriptores/puntos de interés como la distancia más pequeña, como con vecinos más próximo, asi que repetimos lo mismo que en mindist
        for i in range (0, filas_descriptor1min):
            minimo = np.min(distancias_euclideas[i,:])#sacamos la menor de las distancias euclideas
            pos = np.where(distancias_euclideas[i,:]== minimo)#buscamos el minimo en distancias euclideas
            indice = pos[0][0]#gruardamos el indice
            if (minimo <= max_distancia):#comprobamos el umbral, si esta por debajo lo metemos en correspondencias
                correspondencias[i, :] = [i, indice]
                distancias_euclideas[:,indice]= max_distancia + 1

        for i in range(0,correspondencias.shape[0]):
            indices_ordenados = np.argsort(distancias_euclideas[i, :]) # esta funcion lo que nos hace es ordenar los indices en donde se encuentrar las diatncias euclideas menores (orden ascendente)
            indices_dminima1 = indices_ordenados[0]#escogemos el indice de los ordenados que nos va a decir cual es la distancia euclidea menor (y por tanto la mas cercana)
            indices_dminima2 = indices_ordenados[1]#escogemos el iniice de los ordenados quen nos va a decir cual es la segunda distancia euclidea menor (y por tanto la segunda mas cercana)

            # El paso 3 es aplicar la formula del nndr dminima1/dminima2
            nndr = distancias_euclideas[i, indices_dminima1] / distancias_euclideas[i, indices_dminima2]

            # El paso 4 es comprobar el umbral
            # En nndr si el valor de nnr es cercano a 1 es mala correspondencia y no se acepta y si es cercano a 0 es buena correspondencia y se acepta
            if nndr < 0.75: #0.75 seria el umbral que nos indica la aceptacion de la correscondencia y es el que nos pide el guion de la practica
                correspondencias[i, :] = [i, indices_dminima1]
                

    return correspondencias



if __name__ == "__main__":
    print("Practica 2 - Tarea 3 - Test autoevaluación\n")                

    ## tests correspondencias tipo 'minDist' (tarea 3a)
    print("Tests completados = " + str(test_p2_tarea3(disptime=-1,stop_at_error=True,debug=True,tipoDesc='hist',tipoCorr='mindist'))) #analizar todas las imagenes con descriptor 'hist' y ver errores
    # print("Tests completados = " + str(test_p2_tarea3(disptime=0,stop_at_error=False,debug=False,tipoDesc='hist',tipoCorr='mindist'))) #analizar todas las imagenes con descriptor 'hist'
    # print("Tests completados = " + str(test_p2_tarea3(disptime=1,stop_at_error=False,debug=False,tipoDesc='mag-ori',tipoCorr='mindist'))) #analizar todas las imagenes con descriptor 'mag-ori'