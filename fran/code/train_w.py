from tracemalloc import stop
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils
import models

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

print(F"GPUS: {tf.config.list_physical_devices()}")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="resnet152_baseline")
parser.add_argument("--DA_name", type=str, default="DA1")
parser.add_argument("--load_model", type=bool, default=False)
parser.add_argument("--ft_mode", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32)

ARGS = parser.parse_args()

model_name = ARGS.model_name
base_model_name = model_name.split("_")[0]
top_model_name = model_name.split("_")[1]
DA_name = ARGS.DA_name
load = ARGS.load_model

MAIN_PATH = "/mnt/homeGPU/fcastro/lulc/"

#data_dir = "../reduced_data/"
data_dir = MAIN_PATH + "data/"
batch_size = ARGS.batch_size
img_height = 224
img_width = 224

early_stop = True
patience = 5
epochs = 200
mini_epochs=2
learning_rate = ARGS.lr
ft_mode = ARGS.ft_mode

if load:
    results_name = f"{model_name}_loadft{ft_mode}_{DA_name}_w"
else:
    results_name = f"{model_name}_ft{ft_mode}_{DA_name}_w"

weights_dir = MAIN_PATH + "weights/"
load_weights_file = f"{base_model_name}_{top_model_name}_ft0_{DA_name}.h5"
save_weights_file = f"{results_name}"
load_weights_path = weights_dir + load_weights_file
save_weights_path = weights_dir + save_weights_file

submission_dir = MAIN_PATH + "submissions/"
submission_file = f"{results_name}.csv"
submission_path = submission_dir + submission_file

submission_history_path = MAIN_PATH + f"submission_history.csv"

train_data, train_labels, test_data, test_labels, test_names = utils.load_data(data_dir, norm=False)

prep_fn = models.get_prep_fn(base_model_name)
if prep_fn is not None:
    train_data = prep_fn(train_data)

train_labels = tf.one_hot(train_labels, 29).numpy()
test_labels = tf.one_hot(test_labels, 29).numpy()

idx = np.arange(len(train_data))
train_idx, val_idx = train_test_split(idx,test_size=0.1, random_state=0)
val_data = train_data[val_idx]
val_labels = train_labels[val_idx]
train_data = train_data[train_idx]
train_labels = train_labels[train_idx]

DA_fn, DA_test_fn = models.get_DA_fn(DA_name)

if DA_fn is not None:
    print("using DA in training")
if DA_test_fn is not None:
    print("using DA in val and test")

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=DA_fn)
train_generator.fit(train_data)
val_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=DA_test_fn)
val_generator.fit(val_data)

model = models.build_model(base_model_name, top_model_name, ft_mode)

if (load):
    print("Loading model")
    model.load_weights(load_weights_path)

"""
if ft_mode==1:
    base_model = model.layers[1]
    base_model.trainable = True
"""

"""
num_layers = len(base_model.layers)
for layer in base_model.layers[num_layers-ft_layers:]:
        layer.trainable = True
print(num_layers)
print(len(base_model.trainable_variables))
print(len(model.trainable_variables))
"""

loss = tf.keras.losses.CategoricalCrossentropy()
#optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
metrics = [tf.keras.metrics.categorical_accuracy]
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=patience, verbose=0, mode='auto', restore_best_weights=True)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto', restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(save_weights_path, save_best_only=True, save_weights_only=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.000001)

callbacks = [reduce_lr]
#callbacks = [model_checkpoint, reduce_lr]
if early_stop:
    callbacks = callbacks + [early_stopping]

train_it = train_generator.flow(train_data, train_labels, batch_size)
val_it = val_generator.flow(val_data, val_labels, batch_size)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
hist = model.fit(train_it, validation_data=val_it, epochs=mini_epochs, callbacks=callbacks)
stops = 0
for e in range(int(epochs/mini_epochs)-1):
    #class_w = utils.compute_error_class_weights(model, train_data, train_labels)
    sample_w = utils.compute_error_sample_weights(model, train_data, train_labels)
    train_it = train_generator.flow(train_data, train_labels, batch_size, sample_weight=sample_w)
    val_it = val_generator.flow(val_data, val_labels, batch_size)
    hist = model.fit(train_it, validation_data=val_it, epochs=mini_epochs, callbacks=callbacks)
    if model.stop_training:
        stops = stops + 1
        if stops == 3:
            break

model.save_weights(f"{save_weights_path}.h5", save_format='h5')
#tf.keras.models.save_model(model, f"{save_weights_path}.h5", save_format='h5')

preds = model.predict(train_data, batch_size)
train_acc = accuracy_score(np.argmax(train_labels, axis=-1), np.argmax(preds, axis=-1))
print(f"Train Acc.: {train_acc}")

if prep_fn is not None:
    test_data = prep_fn(test_data)

if DA_test_fn is not None:
    test_data = DA_test_fn(test_data).numpy()

preds = model.predict(test_data, batch_size)
test_acc = accuracy_score(np.argmax(test_labels, axis=-1), np.argmax(preds, axis=-1))
print(f"Test Acc.: {test_acc}")
true_preds = np.argmax(preds, axis=-1)+1

d = {'id.jpg': test_names, 'label': true_preds}
df = pd.DataFrame(data=d)
print(f"Writing {submission_path}")
df.to_csv(submission_path, index=False)
print("Done")

d = {'submission_name': [results_name], 'train_acc': [train_acc], 'test_acc': [test_acc]}
df = pd.DataFrame(data=d)
print(f"Updating {submission_history_path}")
df.to_csv(submission_history_path, index=False, mode='a', header=False)
print("Done")