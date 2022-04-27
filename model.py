import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input


def build_model():
    print("Building model ...")
    base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(48, 48, 3))

    new_input = Input(shape=(48, 48, 1), name='image_input')
    x = layers.Conv2D(3, kernel_size=(3, 3), activation='relu', padding='same')(new_input)
    x = layers.GlobalAveragePooling2D()(base_model(x))
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(7, activation="softmax")(x)

    model = keras.Model(inputs=new_input, outputs=x)

    return model
