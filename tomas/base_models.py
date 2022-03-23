import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB0


def EfficientNetB0_baseline():
    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base.trainable = False
    
    model = base.output
    model = GlobalAveragePooling2D()(model)
    model = Dense(1024, activation='relu')(model)
    model = Dense(29, activation='relu')(model)

    model = Model(base.inputs, model)
    return model

def resnet50_baseline():
    resnet = tf.keras.applications.resnet.ResNet50(include_top = False, weights='imagenet', input_shape=(224, 224, 3))
    resnet.trainable = False

    #for layer in resnet.layers[:fine_tune_at]:
    #    layer.trainable = False

    classifier = resnet.output
    classifier = tf.keras.layers.GlobalAveragePooling2D()(classifier)
    classifier = tf.keras.layers.Dense(1024, activation='relu')(classifier)
    #classifier = tf.keras.layers.Dense(512, activation='relu')(classifier)
    #classifier = tf.keras.layers.Dense(256, activation='relu')(classifier)
    classifier = tf.keras.layers.Dense(29, activation='softmax')(classifier)
    model = tf.keras.Model(inputs=resnet.input, outputs=classifier)

    return model

def resnet101_baseline():
    resnet = tf.keras.applications.resnet.ResNet101(include_top = False, weights='imagenet', input_shape=(224, 224, 3))
    resnet.trainable = False

    #for layer in resnet.layers[:fine_tune_at]:
    #    layer.trainable = False

    classifier = resnet.output
    classifier = tf.keras.layers.GlobalAveragePooling2D()(classifier)
    classifier = tf.keras.layers.Dense(1024, activation='relu')(classifier)
    #classifier = tf.keras.layers.Dense(512, activation='relu')(classifier)
    #classifier = tf.keras.layers.Dense(256, activation='relu')(classifier)
    classifier = tf.keras.layers.Dense(29, activation='softmax')(classifier)
    model = tf.keras.Model(inputs=resnet.input, outputs=classifier)

    return model

def resnet152_baseline():
    resnet = tf.keras.applications.resnet.ResNet152(include_top = False, weights='imagenet', input_shape=(224, 224, 3))
    resnet.trainable = False

    #for layer in resnet.layers[:fine_tune_at]:
    #    layer.trainable = False

    classifier = resnet.output
    classifier = tf.keras.layers.GlobalAveragePooling2D()(classifier)
    classifier = tf.keras.layers.Dense(1024, activation='relu')(classifier)
    #classifier = tf.keras.layers.Dense(512, activation='relu')(classifier)
    #classifier = tf.keras.layers.Dense(256, activation='relu')(classifier)
    classifier = tf.keras.layers.Dense(29, activation='softmax')(classifier)
    model = tf.keras.Model(inputs=resnet.input, outputs=classifier)

    return model

def DA1():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.5)
    ])
    return data_augmentation


def DA2():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.3, 0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.5),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.5)
    ])
    return data_augmentation