import numpy as np
import tensorflow as tf
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    # def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
    #              n_classes=10, shuffle=True):
    # def __init__(self, list_IDs, labels, batch_size=1, dim=(1, 1), n_classes=1, shuffle=False):
    def __init__(self, list_IDs, series, batch_size=1, dim=(1, 1), n_classes=1, shuffle=False):
        # maybe dim = (1, 1, 1), and not sure about list IDs = list of IDs of all Training and Validation samples
        # series is a read .csv file
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.series = series
        # self.labels = labels
        self.list_IDs = list_IDs
        # self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.list_IDs)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            X[i,] = self.series[0, :]

            # Store class
            y[i] = self.labels[ID]

        # return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


class PreProcessing:
    @staticmethod
    # input raw data from .csv
    def timeseries_to_supervised(data, lag=1):
        # Extract values from raw data
        data_values = data.values
        # frame a sequence as a supervised learning problem
        df = DataFrame(data_values)
        columns = [df.shift(i) for i in range(1, lag + 1)]
        columns.append(df)
        df = concat(columns, axis=1)
        df.fillna(0, inplace=True)
        supervised = df
        supervised_values = supervised.values
        return supervised_values

    @staticmethod
    def scale(data):
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(data)
        # transform data
        data = data.reshape(data.shape[0], data.shape[1])
        data_scaled = scaler.transform(data)
        return scaler, data_scaled

    # def scale(supervised_values):
    #     # fit scaler
    #     scaler = MinMaxScaler(feature_range=(-1, 1))
    #     scaler = scaler.fit(supervised_values)
    #     # transform train
    #     supervised_values = supervised_values.reshape(supervised_values.shape[0], supervised_values.shape[1])
    #     supervised_scaled = scaler.transform(supervised_values)
    #     return supervised_scaled

    @staticmethod
    # inverse scaling for a forecasted value
    def invert_scale(scaler, X, value):
        new_row = [x for x in X] + [value]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]
