import os
import tensorflow as tf
import numpy as np

def load_train_data(dir_path, norm=True):
    class_idx_names = [ (n.split("_")[0], n) for n in os.listdir(f"{dir_path}train/")]
    class_idx = [ int(idx) for (idx,_) in class_idx_names]
    args = np.argsort(class_idx)
    class_names_sorted = [ class_idx_names[i][1] for i in args]

    img_height = 224
    img_width = 224

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{dir_path}train/",
        seed=0,
        class_names = class_names_sorted,
        image_size=(img_height, img_width),
        batch_size=1)
    train_ds = train_ds.unbatch()
    train_ds_list = list(train_ds.as_numpy_iterator())

    train_data = np.stack([img for (img, _) in train_ds_list])
    train_labels = np.stack([l+1 for (_, l) in train_ds_list])

    if norm:
        train_data = train_data/255.0
    
    return train_data, train_labels

def load_test_data(dir_path, norm=True):
    test_labels = [ int(n.split("_")[0]) for n in os.listdir(f"{dir_path}test/") ]
    test_labels = np.array(test_labels)

    img_height = 224
    img_width = 224

    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{dir_path}test/",
        shuffle = False,
        labels = None,
        image_size=(img_height, img_width),
        batch_size=1)
    test_ds = test_ds.unbatch()
    test_ds_list = list(test_ds.as_numpy_iterator())

    test_data = np.stack([img for img in test_ds_list])
    #test_labels = np.stack([l for (_, l) in test_ds_list])

    if norm:
        test_data = test_data/255.0
    
    return test_data, test_labels