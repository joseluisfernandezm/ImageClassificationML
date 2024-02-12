# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 2: Descripcion de puntos de interes mediante histogramas.

# AUTOR1: Fernandez Moreno, Jose Luis
# AUTOR2: Ramasco Gorria, Pedro
# PAREJA/TURNO: Grupo 14


# librerias y paquetes por defecto
import numpy as np
from p2_tests import test_p2_tarea2

# Incluya aqui las librerias que necesite en su codigo
from scipy import ndimage

def descripcion_puntos_interes(imagen, coords_esquinas, vtam = 8, nbins = 16, tipoDesc='hist'):
    """
    # Esta funcion describe puntos de interes de una imagen mediante histogramas, analizando 
    # vecindarios con dimensiones "vtam+1"x"vtam+1" centrados en las coordenadas de cada punto de interes
    #   
    # La descripcion obtenida depende del parametro 'tipoDesc'
    #   - Caso 'hist': histograma normalizado de valores de gris 
    #   - Caso 'mag-ori': histograma de orientaciones de gradiente
    #
    # En el caso de que existan puntos de interes en los bordes de la imagen, el descriptor no
    # se calcula y el punto de interes se elimina de la lista <new_coords_esquinas> que devuelve
    # esta funcion. Esta lista indica los puntos de interes para los cuales existe descriptor.
    #
    # Argumentos de entrada:
    #   - imagen: numpy array con dimensiones [imagen_height, imagen_width].        
    #   - coords_esquinas: numpy array con dimensiones [num_puntos_interes, 2] con las coordenadas 
    #                      de los puntos de interes detectados en la imagen. Tipo int64
    #                      Cada punto de interes se encuentra en el formato [fila, columna]
    #   - vtam: valor de tipo entero que indica el tamaño del vecindario a considerar para
    #           calcular el descriptor correspondiente.
    #   - nbins: valor de tipo entero que indica el numero de niveles que tiene el histograma 
    #           para calcular el descriptor correspondiente.
    #   - tipoDesc: cadena de caracteres que indica el tipo de descriptor calculado
    #
    # Argumentos de salida
    #   - descriptores: numpy array con dimensiones [num_puntos_interes, nbins] con los descriptores 
    #                   de cada punto de interes (i.e. histograma de niveles de gris)
    #   - new_coords_esquinas: numpy array con dimensiones [num_puntos_interes, 2], solamente con las coordenadas 
    #                      de los puntos de interes descritos. Tipo int64  <class 'numpy.ndarray'>
    #
    # NOTA: no modificar los valores por defecto de las variables de entrada vtam y nbins, 
    #       pues se utilizan para verificar el correcto funciomaniento de esta funcion
    """
    # Iniciamos variables de salida
    # descriptores = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)
    # new_coords_esquinas = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)

    #incluya su codigo aqui

    imagen=imagen.astype(float)#casteamos a float

    #Normalizar la imagen si procede
    if(np.amax(imagen)>1):
        imagen=imagen/255

    if tipoDesc=='hist':
        descriptores=[]#inicializamos una lista en la que vamos a ir metiendo los descriptores
        new_coords_esquinas=[]#inicializamos una lista en la que vamos a ir metiendo las nuevas coordenadas (omitiendo las que quedan fuera de la imagen)
        
        for coordenada in coords_esquinas:
            y,x=coordenada#coordenada de la esquina detectada

            #If que comprueba que no hay puntos de interés fuera de los bordes de la imagen 
            if (y>=int(vtam/2) and x>=int(vtam/2) and y < (imagen.shape[0] - int(vtam/2)) and x < (imagen.shape[1] - int(vtam/2))):
                #Definicion del vecindario de la esquina detectada
                '''
                # start_y=y-vtam//2
                # end_y=y+vtam//2+1
                # start_x=x-vtam//2
                # end_x=x+vtam//2+1
                '''

                #Comprobamos si el tamaño de la ventana es par o por el contrario impar
                if vtam % 2 == 0:#compruebo si el vtam de entrada es par, si lo es significa que la ventana será impar vtam+1 de lado
                    start_y=y-int(vtam/2)
                    end_y=y+int(vtam/2)+1
                    start_x=x-int(vtam/2)
                    end_x=x+int(vtam/2)+1
                else: 
                    start_y=y-int((vtam+1)/2)
                    end_y=y+int((vtam+1)/2)
                    start_x=x-int((vtam+1)/2)
                    end_x=x+int((vtam+1)/2)



                vecindario=imagen[start_y:end_y,start_x:end_x]#seleccionamos en la imagen el vecindario que hemos definido antes para cada esquina
                histograma,_=np.histogram(vecindario,bins=nbins,range=(0,1))#sacamos el histograma con el metodo de numpy para cada vecindario
                histograma_norm=histograma/np.sum(histograma)# normalizamos el histograma dividiendo cada valor por la suma total del histograma
                descriptores.append(histograma_norm)#añadimos el histograma normalizado a la lista de descriptores
                new_coords_esquinas.append(coordenada)#en este caso la coordenada si que esta dentro de la imagen y entonces si que la añadimos a la lista con la nuevas coordenadas

            else:
                print('La coordenada: ['+str(x)+','+str(y)+'] esta fuera de los limites de la imagen y no se incluye en new_coods_esquinas')

        #usamos np.asarray para que concuerden los tipos
        descriptores=np.asarray(descriptores)
        new_coords_esquinas=np.asarray(new_coords_esquinas)

            
    

    if tipoDesc=='mag-ori':
        #En el apartado 2B vamos a implementar las etapas del descriptor SIFT
        descriptores=[]#inicializamos una lista en la que vamos a ir metiendo los descriptores
        new_coords_esquinas=[]#inicializamos una lista en la que vamos a ir metiendo las nuevas coordenadas (omitiendo las que quedan fuera de la imagen)

        #Etapa 1 Asignacion de orientacion principal a cada punto de interes
        
        #Se obtiene el gradiente de la imagen 2D aplicando un filtro de sobel 
        imagen_derivada_x=ndimage.sobel(imagen,1,mode='constant')
        imagen_derivada_y=ndimage.sobel(imagen,0,mode='constant')   

        #Inicializamos una matriz que guarde los modulos y otra que guarde las orientaciones como haciamos en teoría 
        magnitud_del_gradiente=np.zeros((imagen.shape[0], imagen.shape[1]))
        orientacion_del_gradiente=np.zeros((imagen.shape[0], imagen.shape[1]))

        # grados por intevalo
        intervalos=np.linspace(0,360,num=nbins+1)
        

        # Usamos las formlas de modulo y orientacion como en los problemas para completar las matrices
        
        magnitud_del_gradiente = np.sqrt(imagen_derivada_x**2+imagen_derivada_y**2)
        orientacion_del_gradiente =np.rad2deg(np.arctan2(imagen_derivada_y, imagen_derivada_x))
        orientacion_del_gradiente=orientacion_del_gradiente%360
        # orientacion_del_gradiente = np.where(orientacion_del_gradiente<0.0, orientacion_del_gradiente+360.0, orientacion_del_gradiente)
        # orientacion_del_gradiente = np.where(orientacion_del_gradiente==360, orientacion_del_gradiente-360, orientacion_del_gradiente)
        
        
        for coordenada in coords_esquinas:
            
            y,x=coordenada#coordenada de la esquina detectada

            #If que comprueba que no hay puntos de interés fuera de los bordes de la imagen 
            if (y>=int(vtam/2) and x>=int(vtam/2) and y < (imagen.shape[0] - int(vtam/2)) and x < (imagen.shape[1] - int(vtam/2))):
                
                #Comprobamos si el tamaño de la ventana es par o por el contrario impar
                if vtam % 2 == 0:#compruebo si el vtam de entrada es par, si lo es significa que la ventana será impar vtam+1 de lado
                    start_y=y-int(vtam/2)
                    end_y=y+int(vtam/2)+1
                    start_x=x-int(vtam/2)
                    end_x=x+int(vtam/2)+1
                else: 
                    start_y=y-int((vtam+1)/2)
                    end_y=y+int((vtam+1)/2)
                    start_x=x-int((vtam+1)/2)
                    end_x=x+int((vtam+1)/2)


                vecindario_magnitud=magnitud_del_gradiente[start_y:end_y,start_x:end_x]#seleccionamos las magnitudes en un  vecindario que hemos definido antes para cada esquina
                vecindario_orientacion=orientacion_del_gradiente[start_y:end_y,start_x:end_x]#seleccionamos las orientaciones en un  vecindario que hemos definido antes para cada esquina

                hist,_ = np.histogram(vecindario_orientacion, bins=intervalos, weights=vecindario_magnitud)#generamos los descriptores haciendo un histograma con weights correspondientes al vecindario_magnitud
                   
                descriptores.append(hist)#añadimos el histograma normalizado a la lista de descriptores
                new_coords_esquinas.append(coordenada)#en este caso la coordenada si que esta dentro de la imagen y entonces si que la añadimos a la lista con la nuevas coordenadas

            else:
                print('La coordenada: ['+str(x)+','+str(y)+'] esta fuera de los limites de la imagen y no se incluye en new_coods_esquinas')

        #usamos np.asarray para que concuerden los tipos      
        descriptores=np.asarray(descriptores)
        new_coords_esquinas=np.asarray(new_coords_esquinas)

    return descriptores, new_coords_esquinas
    
if __name__ == "__main__":    
    print("Practica 2 - Tarea 2 - Test autoevaluación\n")                

    ## tests descriptor tipo 'hist' (tarea 2a)
    print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=False,debug=False,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test
    # print("Tests completados = " + str(test_p2_tarea2(disptime=0,stop_at_error=False,debug=False,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test, mostrar imagenes con resultados (1 segundo)
    # print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='hist'))) #analizar todas las imagenes y esquinas del test, pararse en errores y mostrar datos
    #print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='hist',imgIdx = 3, poiIdx = 7))) #analizar solamente imagen #2 y esquina #7    

    ## tests descriptor tipo 'mag-ori' (tarea 2b)
    # print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=False,debug=False,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test
    # print("Tests completados = " + str(test_p2_tarea2(disptime=0.1,stop_at_error=False,debug=False,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test, mostrar imagenes con resultados (1 segundo)
    # print("Tests completados = " + str(test_p2_tarea2(disptime=-1,stop_at_error=True,debug=True,tipoDesc='mag-ori'))) #analizar todas las imagenes y esquinas del test, pararse en errores y mostrar datos
    # print("Tests completados = " + str(test_p2_tarea2(disptime=1,stop_at_error=True,debug=True,tipoDesc='mag-ori',imgIdx = 3,poiIdx = 7))) #analizar solamente imagen #1 y esquina #7       