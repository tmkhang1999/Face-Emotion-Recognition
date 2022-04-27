import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")


def create_training_data(directory):
    data = pd.read_csv(directory)
    data_points = data['pixels'].tolist()
    print("Successfully load data from csv file!")

    X = list()
    for line in tqdm(data_points):
        img = [int(point) for point in line.split(' ')]
        img = np.asarray(img).reshape(48, 48)
        X.append(img)
    X = np.asarray(X)
    X = np.expand_dims(X, -1)
    X = X.astype('float32')
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    Y = pd.get_dummies(data['emotion']).to_numpy()
    print('Done')
    return X, Y


train_X, train_Y = create_training_data("dataset/fer2013.csv")
np.save('dataset/X.npy', train_X)
np.save('dataset/Y.npy', train_Y)
