import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import argparse

import utils
import models

tf.config.list_physical_devices()

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="resnet152_baseline")
parser.add_argument("--DA_name", type=str, default="DA1")
parser.add_argument("--load_model", type=bool, default=False)
parser.add_argument("--ft_layers", type=int, default=0)

ARGS = parser.parse_args()

model_name = ARGS.model_name
DA_name = ARGS.DA_name
load = ARGS.load_model
ft_layers = ARGS.ft_layers

MAIN_PATH = "/mnt/homeGPU/fcastro/MINDAT_AVAN/fran/"

model_fn = getattr(models, model_name)

#data_dir = "../reduced_data/"
data_dir = MAIN_PATH + "data/"
batch_size = 32
img_height = 224
img_width = 224

if load:
    results_name = f"{model_name}_loadft{ft_layers}_{DA_name}"
else:
    results_name = f"{model_name}_ft{ft_layers}_{DA_name}"

weights_dir = MAIN_PATH + "weights/"
load_weights_file = f"{results_name}.h5"
load_weights_path = weights_dir + load_weights_file

submission_dir = MAIN_PATH + "submissions/"
submission_file = f"{results_name}.csv"
submission_path = submission_dir + submission_file

test_data, test_labels, test_names = utils.load_test_data(data_dir, norm=False)
test_data = tf.keras.applications.resnet.preprocess_input(test_data)
test_labels = tf.one_hot(test_labels, 29).numpy()

model = model_fn()
model.load_weights(load_weights_path)

preds = model.predict(test_data, 8)
preds_argmax = np.argmax(preds, axis=-1)
acc = accuracy_score(np.argmax(test_labels, axis=-1), preds_argmax)
print(f"Test Acc.: {acc}")

print(f"Writing {submission_path}")
d = {'id.jpg': test_names, 'label': preds_argmax}
df = pd.DataFrame(data=d)
df.to_csv(submission_path, index=False)
print("Done")



