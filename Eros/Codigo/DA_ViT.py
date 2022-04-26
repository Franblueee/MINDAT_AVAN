import tensorflow as tf
import numpy as np
import keras
from keras.utils import np_utils
import cv2
import os
import tensorflow_addons as tfa
import pandas as pd
from numpy.random import seed           #Para controlar generación de pesos
from sklearn.model_selection import train_test_split

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
names_test = []

print("- Cargando imágenes de test...")

for root, _, files in os.walk(test_dir):
    for file in files:
        if file.endswith(".jpg"):               
            x_test.append(cv2.imread(os.path.join(root, file), 1))
            names_test.append(file)


# Convertir a numpy array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
names_test = np.array(names_test)


print('- Normalizando -')
# Normalizar las imágenes
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0

#Data Augmentation:
train, val = train_test_split(np.arange(len(y_train)), test_size=0.1, random_state=21, stratify=y_train)
train_data = x_train[train]
train_labels = y_train[train]
val_data = x_train[val]
val_labels = y_train[val]




data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
      tf.keras.layers.experimental.preprocessing.RandomContrast(0.5)
  ])

batch_size = 64
shuffle_buffer_size = 1000

nclasses = len(np.unique(y_train))
img_shape = x_train.shape[1:4]

train_labels_coded = tf.one_hot(train_labels-1, depth=nclasses, on_value=1, off_value=0)
val_labels_coded = tf.one_hot(val_labels-1, depth=nclasses, on_value=1, off_value=0)

train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(train_data), tf.convert_to_tensor(train_labels_coded)))
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
val_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(val_data), tf.convert_to_tensor(val_labels_coded)))

train_batches = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
val_batches = val_dataset.batch(batch_size)

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

print("- Entrenando el modelo...")
from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping =EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('best.h5', save_best_only=True, save_weights_only=True)
history = model.fit(train_batches, epochs=90, validation_data=val_batches, callbacks=[early_stopping, model_checkpoint])

model.load_weights('best.h5')
model.save_weights('my_model_weights.h5')

#Predicción:
pred = model.predict(x_test,verbose=1)
pred = np.argmax(pred, axis = 1)
pred = pred+1

#Almacenamos los resultados en un csv
results = pd.DataFrame({'id.jpg': names_test, 'label': pred})
results.to_csv('predEEN.csv', index=False) 


