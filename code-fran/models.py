import tensorflow as tf


def resnet50_baseline(fine_tune_at=100):
    resnet = tf.keras.applications.resnet.ResNet50(include_top = False, weights='imagenet', input_shape=(224, 224, 3))
    resnet.trainable = True

    for layer in resnet.layers[:fine_tune_at]:
        layer.trainable = False

    classifier = resnet.output
    classifier = tf.keras.layers.GlobalAveragePooling2D()(classifier)
    classifier = tf.keras.layers.Dense(29, activation='softmax')(classifier)
    model = tf.keras.Model(inputs=resnet.input, outputs=classifier)

    loss = tf.keras.losses.CategoricalCrossentropy()
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    metrics = [tf.keras.metrics.categorical_accuracy]
    patience = 10
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto', restore_best_weights=True)]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model