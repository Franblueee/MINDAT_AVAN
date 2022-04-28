import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

import utils
import models

print(tf.config.list_physical_devices())


MAIN_PATH = "/mnt/homeGPU/fcastro/lulc/"

#data_dir = "../reduced_data/"
data_dir = MAIN_PATH + "data/"
batch_size = 32
img_height = 224
img_width = 224

water_correction=True
patches_predictions = True

submission_dir = MAIN_PATH + "submissions/"
submission_file = "ensemble.csv"
submission_path = submission_dir + submission_file
new_submission_file = "ensemble_waterpcasvm.csv"
new_submission_path = submission_dir + new_submission_file

weights_dir = MAIN_PATH + "weights/"

test_data, test_labels, test_names = utils.load_test_data(data_dir, norm=False)
test_labels = tf.one_hot(test_labels, 29).numpy()
int_test_labels = np.argmax(test_labels, axis=-1)

#model_names = [
#    "mobilenetv3large_v3-1_ft1_DA4_w.h5", "mobilenetv3large_v3-1_ft1_DA4.h5", "mobilenetv3large_v2-1_ft1_DA4.h5", 
#    "mobilenetv3large_v0_ft1_DA0.h5", "mobilenetv3large_v1_ft1_DA4.h5"                
#                ]

#model_names = [ "mobilenetv3large_v3-1_ft1_DA4_w_1.h5" ]

subm_hist = pd.read_csv("submission_history.csv")
model_names = (subm_hist[subm_hist['test_acc']>0.94])['submission_name']
model_names = [f"{m}.h5" for m in model_names]

print(model_names)

preds_array = []
for name in model_names:
    print(name)
    base_model_name = name.split("_")[0]
    top_model_name = name.split("_")[1]
    ft_name = name.split("_")[2]
    if ft_name=="ft0":
        ft_mode = 0
    else:
        ft_mode = 1
    load_weights_path = weights_dir + name

    model = models.build_model(base_model_name, top_model_name, ft_mode)
    model.load_weights(load_weights_path)

    prep_fn = models.get_prep_fn(base_model_name)
    prep_test_data = prep_fn(test_data)

    if patches_predictions:
        preds = utils.patches_predict(model, prep_test_data)
    else:
        preds = model.predict(prep_test_data, 8)
    preds_array = preds_array + [preds]
    preds_argmax = np.argmax(preds, axis=-1)
    acc = accuracy_score(int_test_labels, preds_argmax)
    print(f"Test Acc.: {acc}")

preds_array = np.array(preds_array)
mean_preds = np.mean(preds_array, axis=0)
mean_preds_argmax = np.argmax(mean_preds, axis=-1)

acc = accuracy_score(int_test_labels, mean_preds_argmax)
print(f"Ensemble Test Acc.: {acc}")

true_preds = mean_preds_argmax+1

if water_correction:

    train_data, train_labels = utils.load_train_data(data_dir, norm=False)
    train_idx = np.logical_or(train_labels==20, train_labels==21)
    water_train_data = train_data[train_idx]/255.0
    water_train_labels = train_labels[train_idx]

    test_idx = np.logical_or(mean_preds_argmax==20, mean_preds_argmax==21)
    water_test_data = test_data[test_idx]/255.0

    water_preds_argmax = utils.train_predict_water(water_train_data, water_train_labels, water_test_data)
    
    new_preds_argmax = mean_preds_argmax.copy()
    j = 0
    for i in range(len(mean_preds_argmax)):
        lab = mean_preds_argmax[i]
        if lab==20 or lab==21:
            new_preds_argmax[i] = water_preds_argmax[j]
            j = j+1

    true_preds = new_preds_argmax+1
    acc = accuracy_score(int_test_labels, new_preds_argmax)
    print(f"Ensemble Test Acc. (after water modification): {acc}")


    """
    pipe = utils.train_water_svm(water_train_data, water_train_labels)
    
    modify_preds_argmax = mean_preds_argmax.copy()
    norm_test_data = test_data/255.0
    for i in range(len(mean_preds_argmax)):
        lab = mean_preds_argmax[i]
        if lab==20 or lab==21:
            img = norm_test_data[i]
            x = img.reshape(-1)
            x = np.expand_dims(x, 0)
            p = pipe.predict(x)[0]
            modify_preds_argmax[i] = p
            print(f"old pred: {mean_preds_argmax[i]}, new pred: {p}")
    true_preds = modify_preds_argmax+1
    acc = accuracy_score(int_test_labels, modify_preds_argmax)
    print(f"Ensemble Test Acc. (after water modification): {acc}")
    """

print(f"Writing {new_submission_path}")
d = {'id.jpg': test_names, 'label': true_preds}
df = pd.DataFrame(data=d)
df.to_csv(new_submission_path, index=False)
print("Done")



