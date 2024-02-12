# Tratamiento de Señales Visuales/Tratamiento de Señales Multimedia I @ EPS-UAM
# Practica 2: Extraccion, descripcion y correspondencia de caracteristicas locales
# Memoria: codigo de la pregunta 2.2

# AUTOR1: Fernandez Moreno, Jose Luis
# AUTOR2: Ramasco Gorria, Pedro
# PAREJA/TURNO: Grupo 14


'''
P2.2 Analice las correspondencias obtenidas en la tarea 3 con los descriptores ‘hist’ y ‘magori’ sobre la primera imagen de test . Visualice y razone porque los cambios que observe en
los experimentos que realice. 

'''
# En esta pregunta hacemos usos de los test, estas lineas las hemos extraido de las tareas y las hemos modificado
# con disptime=0 para que seamos nosotros los que pasemos de una imagen a otra y con imhIdx para elegir la primera imagen de test

from p2_tests import test_p2_tarea3

print('------------------Hist------------------')
print("Tests completados = " + str(test_p2_tarea3(disptime=0,stop_at_error=False,debug=False,tipoDesc='hist',tipoCorr='mindist',imgIdx=0))) #analizar todas las imagenes con descriptor 'hist'

print('------------------Mag_Ori------------------')
print("Tests completados = " + str(test_p2_tarea3(disptime=0,stop_at_error=False,debug=False,tipoDesc='mag-ori',tipoCorr='mindist',imgIdx=0))) #analizar todas las imagenes con descriptor 'mag-ori'