# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Tarea 1: Deteccion de puntos de interes con Harris corner detector.

# AUTOR1: Fernandez Moreno, Jose Luis
# AUTOR2: Ramasco Gorria, Pedro
# PAREJA/TURNO: Grupo 14

# librerias y paquetes por defecto
import numpy as np
from p2_tests import test_p2_tarea1

# Incluya aqui las librerias que necesite en su codigo
from scipy import ndimage
import skimage.feature


def detectar_puntos_interes_harris(imagen, sigma = 1.0, k = 0.05, threshold_rel = 0.2):
    """
    # Esta funcion detecta puntos de interes en una imagen con el algoritmo de Harris.
    #
    # Argumentos de entrada:
    #   - imagen: numpy array con dimensiones [imagen_height, imagen_width].  
    #   - sigma: valor de tipo double o float que determina el factor de suavizado aplicado
    #   - k: valor de tipo double o float que determina la respuesta R de Harris
    #   - threshold_rel: valor de tipo double o float que define el umbral relativo aplicado sobre el valor maximo de R
    # Argumentos de salida
    #   - coords_esquinas: numpy array con dimensiones [num_puntos_interes, 2] con las coordenadas 
    #                      de los puntos de interes detectados en la imagen. Cada punto de interes 
    #                      se encuentra en el formato [fila, columna] de tipo int64
    #
    # NOTA: no modificar los valores por defecto de las variables de entrada sigma y k, 
    #       pues se utilizan para verificar el correcto funciomaniento de esta funcion
    """
    coords_esquinas = np.empty(shape=[0,0]) # iniciamos la variable de salida (numpy array)

    #incluya su codigo aqui
    
    #El primer paso es el de normalizar la imagen, para ello pasamos a tipo float y dividimos por 255 como valor maximo de la imagen
    #Para comprobar si la imagen ya viene normalizada comprobamos con un if si el valor maximo de la imagen es mayor a 1 o no, si no 
    #lo es significa que la imagen ya esta normalizada y entonces no hace falta dividir por 255

    imagen=imagen.astype(float)#casteamos a float

    #solo si es preciso normalizaremos la imagen
    if(np.amax(imagen)>1):
        imagen=imagen/255

    # El segundo paso como indican las diapositivas es el de obtener las dos imagenes corresponcientes a la derivada horizontal y vertical
    # de la imagen dada, y como son imagenes 2D vamos utilizat un filtro de sobel con el metodo de scipy

    imagen_derivada_x=ndimage.sobel(imagen,1)
    imagen_derivada_y=ndimage.sobel(imagen,0)

    # El tercer paso es el de obtener las tres imagenes prodicto de derivadas parciales mediante productos elemento a elemento de las imagenes
    # correspondientes

    imagen_derivada_xx=imagen_derivada_x*imagen_derivada_x
    imagen_derivada_yy=imagen_derivada_y*imagen_derivada_y
    imagen_derivada_xy=imagen_derivada_x*imagen_derivada_y

    # El cuarto paso setá el de aplicar el filtrado gaussiano a las tres imágenes producto anteriores. Esto añade robustez frente al ruido
    # para eso usamos el metodo gaussian_filter de ndimage

    imagen_derivada_xx_gauss=ndimage.filters.gaussian_filter(imagen_derivada_xx,sigma,mode='constant')
    imagen_derivada_yy_gauss=ndimage.filters.gaussian_filter(imagen_derivada_yy,sigma,mode='constant')
    imagen_derivada_xy_gauss=ndimage.filters.gaussian_filter(imagen_derivada_xy,sigma,mode='constant')

    # El quinto paso consiste en determinar la matriz Mxy para cada pixel de la imagen y calcular la funcion R de cada pixel usando Mxy

    '''
    #------------------ Metodo lento pero que funciona ------------------
    Mxy=np.zeros((imagen_derivada_xx_gauss.shape[0],imagen_derivada_xx_gauss.shape[1],2,2))#inicializo la matriz Mxy a ceros
    Rxy=np.zeros((imagen_derivada_xx_gauss.shape[0],imagen_derivada_xx_gauss.shape[1]))

    for i in range(0,imagen_derivada_xx_gauss.shape[0]):
        for j in range(0,imagen_derivada_xx_gauss.shape[1]):
            
            #Vamos dando forma a la matriz Mxy para cada pixel
            Mxy[:,:,0,0]=imagen_derivada_xx_gauss[i,j]
            Mxy[:,:,0,1]=imagen_derivada_xy_gauss[i,j]
            Mxy[:,:,1,0]=imagen_derivada_xy_gauss[i,j]
            Mxy[:,:,1,1]=imagen_derivada_yy_gauss[i,j]

            #Para cada pixel vamos a calcular la funcion R con la formula de las diapositivas
            det_Mxy=np.linalg.det(Mxy[i,j])#calculo por separado del determinante de Mxy
            traza_Mxy=np.trace(Mxy[i,j])#calculo por separado de la traza de Mxy

            Rxy[i,j]=det_Mxy-0.04*(traza_Mxy**2)#formula de las diapositivas, ponemos de constante k 0.04 que es la recomendada por Harris

    # El sexto paso es una deteccion inicial de las esquinas umbralizando la funcion R
    # para ello usamos el metodo de skimage.feature.corner_peaks con min_distance=5 como indica el guion de la practica
    '''
    #------------------ Metodo rapido vectorizando sin bucles ------------------
    Mxy=np.zeros((imagen_derivada_xx_gauss.shape[0],imagen_derivada_xx_gauss.shape[1],2,2))#inicializo la matriz Mxy a ceros
    Rxy=np.zeros((imagen_derivada_xx_gauss.shape[0],imagen_derivada_xx_gauss.shape[1]))#inicializo la matriz Rxy a ceros
        
    #Vamos dando forma a la matriz Mxy para cada pixel
    Mxy[:,:,0,0]=imagen_derivada_xx_gauss
    Mxy[:,:,0,1]=imagen_derivada_xy_gauss
    Mxy[:,:,1,0]=imagen_derivada_xy_gauss
    Mxy[:,:,1,1]=imagen_derivada_yy_gauss

    det_Mxy=np.linalg.det(Mxy)#calculo por separado del determinante de Mxy
    traza_Mxy=np.trace(Mxy, axis1=2, axis2=3)#calculo por separado de la traza de Mxy

    #Calculamos la funcion de respuesta R para cada pixel utilizando Mxy
    Rxy = det_Mxy - k * traza_Mxy**2 #la K me la pasan como argumento de la funcion

    #Deteccion 
    coords_esquinas = skimage.feature.corner_peaks(Rxy, min_distance = 5, threshold_rel = threshold_rel)#nos devuelve las coordenadas de las esquinas
    
    return coords_esquinas

if __name__ == "__main__":    
    print("Practica 2 - Tarea 1 - Test autoevaluación\n")                
    
    print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=False,debug=False))) #analizar todos los casos sin pararse en errores
    #print("Tests completados = " + str(test_p2_tarea1(disptime=1,stop_at_error=False,debug=False))) #analizar y visualizar todos los casos sin pararse en errores
    #print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=True,debug=False))) #analizar todos los casos y pararse en errores 
    #print("Tests completados = " + str(test_p2_tarea1(disptime=-1,stop_at_error=True,debug=True))) #analizar todos los casos, pararse en errores y mostrar informacion