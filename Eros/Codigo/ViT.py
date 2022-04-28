import tensorflow as tf
import numpy as np
import keras
from keras.utils import np_utils
import cv2
import os
import tensorflow_addons as tfa
import pandas as pd
from numpy.random import seed           #Para controlar generación de pesos

from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())


#Lectura de Imagenes
learning_rate=1e-4
print("--- LECTURA DE IMÁGENES ---")

print("- Cargando imágenes de entrenamiento...")

x_train = []
y_train = []

for root, _, files in os.walk(train_dir):
    for file in files:
        if file.endswith(".jpg"):               
            x_train.append(cv2.imread(os.path.join(root, file), 1))
            y_train.append(int(file.split("_")[0]))

x_test = []
y_test = []
names_test = []

print("- Cargando imágenes de test...")

for root, _, files in os.walk(test_dir):
    for file in files:
        if file.endswith(".jpg"):               
            x_test.append(cv2.imread(os.path.join(root, file), 1))
            y_test.append(int(file.split("_")[0]))
            names_test.append(file)


# Convertir a numpy array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
labels=y_test
names_test = np.array(names_test)


print('- Normalizando -')
# Normalizar las imágenes
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0


print('- Categorias -')
# Convertir a categóricas las etiquetas
y_train = y_train-1
y_test = y_test-1
y_train = np_utils.to_categorical(y_train, 29)
y_test = np_utils.to_categorical(y_test, 29)

#Importamos el modelo de ViT 
from vit_keras import vit

#Creamos el modelo base con uno de los ViT disponibles
vit_model = vit.vit_b32(
        image_size = 224,
        activation = 'softmax',
        pretrained = False,
        include_top = False,
        pretrained_top = False,
        classes = 29)

#Seleccionamos si queremos que el modelo base sea entrenable o no
vit_model.trainable=False

#Creamos el modelo añadiendo el top para la clasificación: Hay distintas opciones para el top model.
model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(160, activation = 'gelu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(29, 'softmax')
    ])


optimizer = tfa.optimizers.RectifiedAdam(learning_rate = learning_rate)
model.compile(optimizer = optimizer, 
              loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2), 
              metrics = ['accuracy'])

print("--- ENTRENAMIENTO DEL MODELO ---")
p_train = np.random.permutation(len(x_train))
p_test = np.random.permutation(len(x_test))

x_train = x_train[p_train]
y_train = y_train[p_train]
x_test = x_test[p_test]
y_test = y_test[p_test]
names_test = names_test[p_test]
labels = labels[p_test]

print("- Entrenando el modelo...")
from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping =EarlyStopping(monitor='val_loss', patience=12)
model_checkpoint = ModelCheckpoint('prueba.h5', save_best_only=True, save_weights_only=True)
history = model.fit(x_train, y_train, batch_size= 64, epochs=250, validation_split=0.1, callbacks=[early_stopping, model_checkpoint])

model.load_weights('prueba.h5')
model.save_weights('my_model_weightsViT2.h5')

#Predicción:
pred = model.predict(x_test,verbose=1)
pred = np.argmax(pred, axis = 1)
pred = pred+1

#Almacenamos los resultados en un csv
results = pd.DataFrame({'id.jpg': names_test, 'label': pred})
results.to_csv('predEEN.csv', index=False) 


