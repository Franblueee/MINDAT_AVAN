# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import utils
import models

# %%
tf.config.list_physical_devices()

# %%
#data_dir = "./reduced_data/"
data_dir = "./data/"
batch_size = 32
img_height = 224
img_width = 224
EPOCHS = 50
patience = 10
weights_file = "./weights/resnet152_baseline_DA1.h5"

# %%
train_data, train_labels, test_data, test_labels, test_names = utils.load_data(data_dir, norm=False)
train_data = tf.keras.applications.resnet.preprocess_input(train_data)
test_data = tf.keras.applications.resnet.preprocess_input(test_data)
train_labels = tf.one_hot(train_labels, 29).numpy()
test_labels = tf.one_hot(test_labels, 29).numpy()

idx = np.arange(len(train_data))
train_idx, val_idx = train_test_split(idx,test_size=0.1, random_state=0)
val_data = train_data[val_idx]
val_labels = train_labels[val_idx]
train_data = train_data[train_idx]
train_labels = train_labels[train_idx]

# %%
data_augmentation = models.DA1()
def prep_fn(img):
    img = tf.expand_dims(img, axis=0)
    new_img = data_augmentation(img)[0]
    return new_img

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=prep_fn)
train_generator.fit(train_data)
val_generator = tf.keras.preprocessing.image.ImageDataGenerator()

# %%
#model = models.resnet50_baseline()
#model = models.resnet101_baseline()
model = models.resnet152_baseline()


loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
metrics = [tf.keras.metrics.categorical_accuracy]
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto', restore_best_weights=True)]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# %%
train_it = train_generator.flow(train_data, train_labels, batch_size)
val_it = val_generator.flow(val_data, val_labels, batch_size)

#hist = model.fit(train_data, train_labels, epochs=30, callbacks=callbacks, validation_data=(test_data, test_labels), batch_size=128)
hist = model.fit(train_it, validation_data=val_it, epochs=EPOCHS, callbacks=callbacks)

# %%
model.save_weights(weights_file)

# %%
preds = model(test_data)
acc = accuracy_score(np.argmax(test_labels, axis=-1), np.argmax(preds, axis=-1))
f1 = f1_score(np.argmax(test_labels, axis=-1), np.argmax(preds, axis=-1), average="micro")
print(acc)
print(f1)
