import os
import tensorflow as tf
import numpy as np
import cv2


def load_data(path, norm=True):
    
    train_dir_path = f"{path}/train"
    train_dirs = os.listdir(train_dir_path)
    train_data = []
    train_labels = []
    for dir in train_dirs:
        label = int(dir.split("_")[0])
        dir_path = f"{train_dir_path}/{dir}"
        imgs_names = os.listdir(dir_path)
        for name in imgs_names:
            img_path = f"{dir_path}/{name}"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_data.append(img)
            train_labels.append(label)
    
    test_dir_path = f"{path}/test"
    test_names = os.listdir(test_dir_path)
    test_data = []
    test_labels = []
    for name in test_names:
        label = int(name.split("_")[0])
        img_path = f"{test_dir_path}/{name}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_data.append(img)
        test_labels.append(label)
    
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    if norm:
        train_data = train_data/255.0
        test_data = test_data/255.0

    return train_data, train_labels, test_data, test_labels, test_names

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

