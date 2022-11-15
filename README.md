# PROYECTO2
Proyecto 2: Reconocimiento facial

Primero se pensaba utilizar la base de datos Celab, pero era demasiado grande entonces se optó por un subconjunto de aproximadamente 15,500 imagenes y, además un conjunto de 170 imagenes de mi rostro.

Las imagenes de mi rostro fueron recortadas a mano, mientras que las imagenes del subconjutno de CelabA se recortaron con un código de python, esto con las paqueterias cv2 y os. El código hace que, una vez dentro de la carpeta en donde se tienen las imagenes del subconjutno de CelabA, se cree una carpeta en la que estarán las imagenes recortadas (esto hace con cada imagen del subconjunto de celabA)

Dado que el conjutno de imagenes de mi rostro es demasiado pequeño, se hizo un codigo con la función ImageDataGenerator para generar más imagenes de mi rostro, este codigo hace pequeños cambios en cada una de mis imagenes para así crecer este conjunto y con esto se llegó a tener 1,700 imagenes de mi rostro, con un total de 17,200 imagenes.
Esto conformo la limpieza de datos. Ya con los datos en orden se realizó y entrenó una Red neuronal convolucional.


## Problemas que se encontraron.
Uno de los primeros problemas fue que, en el codigo que recorta las imagenes algunas veces recortaba mal estas imagenes, entonces estos errores debian ser corregidos manualmente.


## Intentos que no dieron buenos resultados.
Se hicieron varios intentos en los que la Red Neuronal no respondía adecuadamente, esto fue porque existia un desbalance en los datos, hasta que, se llego que, con al menos se debia tener una decima parte de imagenes de mi rostro respecto a las del subconjunto de CelabA.


## Descripción de la Red Neuronal.
Se tiene una Red Neuronal convolucional con el ayuda de la paqueteria de Tnesorflow.
Esta red tiene una estructura secuencial, la cual cuenta con dos capas densas y dos convolucionales.
La red cuenta con una función de costo crossentropy y con el optimizador Adam.


## Conclusión.
Resultó muy interesante trabajar este proyecto, observar lo importante que son las redes neuronales para el reconocimiento facial.
