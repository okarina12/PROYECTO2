from skimage.io import imread_collection , concatenate_images
import numpy as np
import matplotlib.pyplot as plt
import copy  as cp
import copy

folderK = 'Datos/Yo/*.jpg'
folderD = 'Datos/Desconocido/*.jpg'

#Cargamos los datos a python.
imagesK = imread_collection(folderK)
imagesD = imread_collection(folderD) 

#Cantidad de imagenes
nK  = len(imagesK)
nD = len(imagesD)

#Unimos los dos dataset
images = np.append(imagesK, imagesD, axis=1)
images = imagesD.extend(imagesK)

print("Cantidad total de imagenes: ",len(images)) #Verificamos que se hayan unido correctamente con la cantidada de imagenes.

#Se imprime la primer imagen del dataset
plt.imshow(images[0])
print(images[0].shape)

def Create_X():
     return [0]*nK + [1]*nD
X = Create_X()

X = np.array(X)
Y = np.array(images)

from skimage.transform import resize
Y=resize(Y,(len(images),64,64,3))

#plot the first image in the dataset
plt.imshow(Y[0])
print(Y[0].shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

modelo=Sequential() #Varias capas

modelo.add(Conv2D(200,(3,3),input_shape=X.shape[1:]))
modelo.add(Activation('relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

modelo.add(Conv2D(100,(3,3)))
modelo.add(Activation('relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

modelo.add(Conv2D(50,(3,3)))
modelo.add(Activation('relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

modelo.add(Flatten()) #Redimension de imagen

modelo.add(Dropout(0.5))

modelo.add(Dense(50,activation='relu'))

modelo.add(Dense(2,activation='softmax')) 

modelo.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.utils import to_categorical

X=to_categorical(X)

X[0]

X[len(X)-1]

from sklearn.model_selection import train_test_split

Y_train, Y_test, X_train, X_test = train_test_split(Y, X, test_size = 0.30, random_state = 0)

checkpoint = ModelCheckpoint('CNN/model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

history=modelo.fit(Y_train,X_train,epochs=20,callbacks=[checkpoint],validation_split=0.2)

