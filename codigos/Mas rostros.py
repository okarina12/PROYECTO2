from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io
datagen = ImageDataGenerator(
        rotation_range=40,     #Se utilizó rotación de 40
        width_shift_range=0.1,   
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect', cval=125)
x = io.imread('C:/Users/karin/Documents/Redes Neuronales/Proyecto/Datos/Yo/mirostro_0.jpg')
a = x.reshape((1, ) + x.shape)  #Array with shape (1, 256, 256, 3)

i = 0
for batch in datagen.flow(a, batch_size=16,  
                          save_to_dir='K', 
                          save_prefix='mirostro1', 
                          save_format='jpg'):
    i += 1
    if i > 12:
        break  