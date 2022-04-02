#import tensorflow_text
from pydoc import ModuleScanner
import tensorflow as tf
import tensorflow_hub as hub
import random


model_handle_dic = {
    "bitresnet50x3" : "https://tfhub.dev/google/bit/m-r50x3/1",
    "bitresnet101x3" : "https://tfhub.dev/google/bit/m-r101x3/1", 
    "bitresnet152x4" : "https://tfhub.dev/google/bit/m-r152x4/1", 
    "eurosatresnet50v2" : "https://tfhub.dev/google/remote_sensing/eurosat-resnet50/1", 
    "efficientnetv2im21kb0" : "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2",
    "efficientnetv2im21kxl" : "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2", 
    "VITs16" : "https://tfhub.dev/sayakpaul/vit_s16_fe/1", 
    "VITr26s32" : "https://tfhub.dev/sayakpaul/vit_r26_s32_lightaug_fe/1", 
    "VITb8" : "https://tfhub.dev/sayakpaul/vit_b8_fe/1"
}

def build_base_model(base_model_name):
    if base_model_name=="resnet50":
        base_model = tf.keras.applications.ResNet50(include_top = False, weights='imagenet', input_shape=(224, 224, 3))
    elif base_model_name=="resnet101":
        base_model = tf.keras.applications.ResNet101(include_top = False, weights='imagenet', input_shape=(224, 224, 3))
    elif base_model_name=="resnet152":
        base_model = tf.keras.applications.ResNet152(include_top = False, weights='imagenet', input_shape=(224, 224, 3))
    if base_model_name=="resnet50v2":
        base_model = tf.keras.applications.ResNet50V2(include_top = False, weights='imagenet', input_shape=(224, 224, 3))
    elif base_model_name=="resnet101v2":
        base_model = tf.keras.applications.ResNet101V2(include_top = False, weights='imagenet', input_shape=(224, 224, 3))
    elif base_model_name=="resnet152v2":
        base_model = tf.keras.applications.ResNet152V2(include_top = False, weights='imagenet', input_shape=(224, 224, 3))
    elif base_model_name=="efficientnetv2b0":
        base_model = tf.keras.applications.EfficientNetV2B0(include_top = False, weights='imagenet', input_shape=(224, 224, 3), include_preprocessing=False, pooling=None)
    elif base_model_name=="mobilenetv3large":
        base_model = tf.keras.applications.MobileNetV3Large(include_preprocessing=False, include_top=False, weights='imagenet', pooling=None, input_shape=(224, 224, 3))
    else:       
        base_model = hub.KerasLayer(model_handle_dic[base_model_name])
    return base_model

def build_top_model(base_model, base_model_name, top_model_name, num_classes=29):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    if top_model_name == "baseline":
        if len(x.shape)>2:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
    elif top_model_name == "v0":
        if len(x.shape)>2:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(100, activation='relu')(x)
    elif top_model_name == "v1":
        if len(x.shape)>2:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(526)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.BatchNormalization()(x)    
        x = tf.keras.layers.Activation('relu')(x)
    else:
        version_name = top_model_name.split("-")[0]
        conv_it = int(top_model_name.split("-")[1])
        if version_name == "v2":
            num_filters = x.shape[-1]
            for i in range(conv_it):
                x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3), padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation('relu')(x)
                x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3))(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation('relu')(x)
            if (x.shape[1]>1) and (x.shape[2]>1):
                x = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(256)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
        if version_name == "v3":
            num_filters = x.shape[-1]
            for i in range(conv_it):
                x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3,3))(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation('relu')(x)
            if (x.shape[1]>1) and (x.shape[2]>1):
                x = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(256)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation = 'softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def build_model(base_model_name="resnet50", top_model_name="baseline", ft_mode=0, num_classes=29):

    base_model = build_base_model(base_model_name)
    trainable = ft_mode==1
    base_model.trainable = trainable

    final_model = build_top_model(base_model, base_model_name, top_model_name, num_classes)

    return final_model

def random_rot90(img):
    a = random.randint(0,1)
    x = img
    if a==0:
        x=tf.image.rot90(x)
    return x

def random_resize_and_crop(img, max_size=256):
    x = img
    a = random.randint(0,1)
    if a==0:
        x = tf.image.resize(x, [max_size, max_size])
        x = tf.image.random_crop(x, (224, 224, 3))
    return x

def random_resize_and_crop_2(img, max_size=256):
    x = img
    a = random.randint(0,1)
    if a==0:
        b = random.uniform(0.0, 1.0)
        h = 224 + int((max_size - 224)*b)
        x = tf.image.resize(x, [h, h])
        x = tf.image.random_crop(x, (224, 224, 3))
    return x

def get_prep_fn(base_model_name):
    prep_fn = None
    if base_model_name=="eurosatresnet50v2":
        print("eurosatresnet50v2 preprocessing")
        #prep_fn = tf.keras.layers.Rescaling(scale=1./255)
        prep_fn = lambda x : x/255.0
    elif "resnet" in base_model_name:
        print("resnet preprocessing")
        if "v2" in base_model_name:
            print("v2 preprocessing")
            prep_fn = tf.keras.applications.resnet_v2.preprocess_input
        else:
            print("v1 preprocessing")
            prep_fn = tf.keras.applications.resnet.preprocess_input
    elif ("efficientnet" in base_model_name) or ("mobilenet" in base_model_name) or ("VIT" in base_model_name):
        print("efficientnet preprocessing")
        #prep_fn = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
        prep_fn = lambda x : x/127.5-1.0
    return prep_fn

# las funciones de DA toman como entrada un tensor 3D y devuelven un tensor 3D
def get_DA_fn(name):
    prep_fn = None
    test_prep_fn = None
    if name=="DA1":
        prep_fn = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: random_rot90(x)),
            tf.keras.layers.Lambda(lambda x: tf.image.random_flip_left_right(x)), 
            tf.keras.layers.Lambda(lambda x: tf.image.random_flip_up_down(x)),
            #tf.keras.layers.Lambda(lambda x: tf.keras.preprocessing.image.random_rotation(x, 360*0.2)),
            #tf.keras.layers.Lambda(lambda x: tf.image.random_contrast(x, 0.0, 0.5))
        ])
    elif name=="DA2":
        prep_fn = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, 0.2)),
            tf.keras.layers.Lambda(lambda x: tf.image.random_contrast(x, 0.0, 0.5)),
            tf.keras.layers.Lambda(lambda x: tf.image.random_jpeg_quality(x, 85, 100)),
            #tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, 0.75, 1.25)),
            #tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, 0.1)),
            tf.keras.layers.Lambda(lambda x: tf.image.random_flip_left_right(x)), 
            tf.keras.layers.Lambda(lambda x: tf.image.random_flip_up_down(x)),
            #tf.keras.layers.Lambda(lambda x: tf.keras.preprocessing.image.random_rotation(x, 360*0.2))
        ])
    elif name=="DA3":
        prep_fn = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [256, 256])),
            tf.keras.layers.Lambda(lambda x: tf.image.random_crop(x, (224, 224, 3))),
            tf.keras.layers.Lambda(lambda x: random_rot90(x)),
            tf.keras.layers.Lambda(lambda x: tf.image.random_flip_left_right(x)), 
            tf.keras.layers.Lambda(lambda x: tf.image.random_flip_up_down(x))
        ])

        test_prep_fn = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [256, 256])),
            tf.keras.layers.Lambda(lambda x: tf.image.central_crop(x, 0.765625)),
            tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [224, 224]))
        ])
    elif name=="DA4":
        prep_fn = lambda x: random_resize_and_crop(x, 364)
    elif name=="DA5":
        prep_fn = prep_fn = tf.keras.Sequential([ 
            tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, 0.3)),
            tf.keras.layers.Lambda(lambda x: tf.image.random_contrast(x, 0.0, 0.5)),
            tf.keras.layers.Lambda(lambda x: tf.image.random_jpeg_quality(x, 90, 100)),
            tf.keras.layers.Lambda(lambda x: random_resize_and_crop(x, 256))
        ])


    return prep_fn, test_prep_fn


