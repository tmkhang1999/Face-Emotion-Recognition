from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from tensorflow import keras
from keras import Input


def build_model():
    blocks = 4
    convs = [2, 2, 2, 2]
    filters = [32, 64, 128, 256]
    drops = [0.5, 0.5, 0.5, 0.5]
    initializer = keras.initializers.he_normal()
    weight_decay = 1e-5

    model = Sequential()
    model.add(Input(shape=(48, 48, 1)))

    for block in range(blocks):
        for conv in range(convs[block]):
            model.add(keras.layers.Conv2D(filters[block], (3, 3), padding='same',
                                          activation='relu',
                                          kernel_initializer=initializer,
                                          kernel_regularizer=keras.regularizers.l1_l2(weight_decay, weight_decay)))
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Dropout(drops[block]))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(6, activation='softmax'))

    return model
