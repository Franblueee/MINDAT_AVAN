import os
import tensorflow as tf
import numpy as np
import cv2


def load_data(path, norm=True):
    
    train_data, train_labels = load_train_data(path, norm=norm)
    test_data, test_labels, test_names = load_test_data(path, norm=norm)

    return train_data, train_labels, test_data, test_labels, test_names

def load_train_data(path, norm=True):
    train_dir_path = f"{path}/train"
    train_dirs = os.listdir(train_dir_path)
    train_data = []
    train_labels = []
    for dir in train_dirs:
        label = int(dir.split("_")[0])-1
        dir_path = f"{train_dir_path}/{dir}"
        imgs_names = os.listdir(dir_path)
        for name in imgs_names:
            img_path = f"{dir_path}/{name}"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_data.append(img)
            train_labels.append(label)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    if norm:
        train_data = train_data/255.0
    
    return train_data, train_labels

def load_test_data(path, norm=True):
    test_dir_path = f"{path}/test"
    test_names = os.listdir(test_dir_path)
    test_data = []
    test_labels = []
    for name in test_names:
        label = int(name.split("_")[0])-1
        img_path = f"{test_dir_path}/{name}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_data.append(img)
        test_labels.append(label)
    
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    if norm:
        test_data = test_data/255.0
        
    return test_data, test_labels, test_names

def categorize_labels(labels):

    categories_dic = {
        "lands" : {"labels" : [0, 2, 3, 4], "id" : 0},
        "forest" : {"labels" : [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "id" : 1}, 
        "wetland" : {"labels" : [17, 18, 19], "id" : 2}, 
        "snow" : {"labels" : [1, 22], "id" : 3},
        "waterbody" : {"labels" : [20, 21], "id" : 4}, 
        "crop" : {"labels" : [23, 24, 25, 26, 27], "id" : 5}, 
        "urban" : {"labels" : [28], "id" : 6}        
    }

    categories = []
    for l in labels:
        for k in categories_dic.keys():
            if l in categories_dic[k]["labels"]:
                categories = categories + [ categories_dic[k]["id"] ]
                break
    categories = np.array(categories)
    return categories

def compute_error_class_weights(model, data, labels, batch_size=8):
    prob_preds = model.predict(data, batch_size)
    preds = np.argmax(prob_preds, axis=-1)
    int_labels = np.argmax(labels, axis=-1)
    class_w = {}
    for c in np.unique(int_labels):
        c_idx = int_labels==c
        c_preds = preds[c_idx]
        c_total = np.sum(c_idx)
        c_acc = np.sum(c_preds == c)
        class_w[c] = (c_total+2)/(c_acc+1)
    return class_w

def compute_error_sample_weights(model, data, labels, batch_size=8):
    class_w = compute_error_class_weights(model, data, labels, batch_size)
    print(class_w)
    int_labels = np.argmax(labels, axis=-1)
    sample_w = [class_w[int_labels[i]] for i in range(len(labels))]
    return sample_w

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def train_water_svm(train_data, train_labels):
    fl_train_data = train_data.reshape(train_data.shape[0], -1)
    pca = PCA(n_components=10)
    pca = pca.fit(fl_train_data)
    fl_train_data_trans = pca.transform(fl_train_data)
    svm = SVC(C=10.0, kernel="rbf", gamma=0.1)
    svm = svm.fit(fl_train_data_trans, train_labels)
    train_preds = svm.predict(fl_train_data_trans)
    print(accuracy_score(train_labels, train_preds))

    pipe = Pipeline([ ("pca", pca), ("svm", svm) ])

    return pipe


