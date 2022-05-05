import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from model import build_model
from keras.losses import categorical_crossentropy

# Data Preparation
print("Loading data ...")
X = np.load('dataset/train/X.npy', allow_pickle=True)
Y = np.load('dataset/train/Y.npy', allow_pickle=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)

# Build model
model = build_model()
model.compile(loss=categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

# Training
print("Start training ...")
batch_size = 64
epochs = 50

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='./weight/model.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history_data = model.fit(np.array(X_train), np.array(y_train),
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(np.array(X_test), np.array(y_test)),
                         callbacks=[model_checkpoint_callback],
                         shuffle=True,
                         verbose=1)

# graph
plt.plot(history_data.history['accuracy'], label="train_accuracy")
plt.plot(history_data.history['val_accuracy'], label="val_accuracy")
plt.xlabel('iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
