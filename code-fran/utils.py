import os
import tensorflow as tf
import numpy as np

def load_train_data(dir_path, norm=True):
    class_idx_names = [ (n.split("_")[0], n) for n in os.listdir(dir_path)]
    class_idx = [ int(idx) for (idx,_) in class_idx_names]
    args = np.argsort(class_idx)
    class_names_sorted = [ class_idx_names[i][1] for i in args]

    img_height = 224
    img_width = 224

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dir_path,
        seed=0,
        class_names = class_names_sorted,
        image_size=(img_height, img_width),
        batch_size=1)
    train_ds = train_ds.unbatch()
    train_ds_list = list(train_ds.as_numpy_iterator())

    train_data = np.stack([img for (img, _) in train_ds_list])
    train_labels = np.stack([l+1 for (_, l) in train_ds_list])
    
    return train_data, train_labels

def load_test_data(dir_path, norm=True):
    class_idx_names = [ (n.split("_")[0], n) for n in os.listdir(dir_path)]
    class_idx = [ int(idx) for (idx,_) in class_idx_names]
    args = np.argsort(class_idx)
    class_names_sorted = [ class_idx_names[i][1] for i in args]

    img_height = 224
    img_width = 224

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dir_path,
        seed = 0,
        class_names = class_names_sorted,
        image_size=(img_height, img_width),
        batch_size=1)
    test_ds = test_ds.unbatch()
    test_ds_list = list(test_ds.as_numpy_iterator())

    test_data = np.stack([img for (img, _) in test_ds_list])
    #test_labels = np.array(test_labels)
    test_labels = np.stack([l for (_, l) in test_ds_list])
    
    return test_data, test_labels

