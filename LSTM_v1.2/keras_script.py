import numpy as np
# from tensorflow.keras.models import Sequential
import tensorflow as tf
from my_classes import DataGenerator
from pandas import read_csv
import os
from my_classes import PreProcessing

device = 'cpu'

# Parameters
# params = {'dim': (32,32,32),
#           'batch_size': 64,
#           'n_classes': 6,
#           'n_channels': 1,
#           'shuffle': True}
params = {'dim': (1, 1),
          'batch_size': 1,
          'n_classes': 1,
          'shuffle': True}

# load dataset
dataSetRoot = r'../Dataset'
Series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data_1k.csv'), header=0, index_col=0, squeeze=True)
# print(series.head())

# Datasets
Supervised = PreProcessing.timeseries_to_supervised(Series)  # Preprocessed input Training data
Supervised_Scaled = PreProcessing.scale(Supervised)

Partition = {'train': 900, 'validation': 100}  # IDs

# Generators
training_generator = DataGenerator(Partition['train'], Supervised_Scaled, **params)
validation_generator = DataGenerator(Partition['validation'], Supervised_Scaled, **params)

# Design model
model = tf.keras.models.Sequential()
if device == 'cpu':
    model.add(
        tf.keras.layers.LSTM(neurons=10, return_sequences=True, batch_input_shape=(params['batch_size'], params['dim']),
                             stateful=True))
    model.add(
        tf.keras.layers.LSTM(neurons=10, return_sequences=True, batch_input_shape=(params['batch_size'], params['dim']),
                             stateful=True))
    model.add(
        tf.keras.layers.LSTM(neurons=10, return_sequences=True, batch_input_shape=(params['batch_size'], params['dim']),
                             stateful=True))
    model.add(
        tf.keras.layers.LSTM(neurons=10, return_sequences=True, batch_input_shape=(params['batch_size'], params['dim']),
                             stateful=True))
    model.add(tf.keras.layers.LSTM(neurons=10, batch_input_shape=(params['batch_size'], params['dim']), stateful=True))
else:
    model.add(
        tf.keras.layers.CuDNNLSTM(neurons=10, return_sequences=True,
                                  batch_input_shape=(params['batch_size'], params['dim']),
                                  stateful=True))
    model.add(
        tf.keras.layers.CuDNNLSTM(neurons=10, return_sequences=True,
                                  batch_input_shape=(params['batch_size'], params['dim']),
                                  stateful=True))
    model.add(
        tf.keras.layers.CuDNNLSTM(neurons=10, return_sequences=True,
                                  batch_input_shape=(params['batch_size'], params['dim']),
                                  stateful=True))
    model.add(
        tf.keras.layers.CuDNNLSTM(neurons=10, return_sequences=True,
                                  batch_input_shape=(params['batch_size'], params['dim']),
                                  stateful=True))
    model.add(
        tf.keras.layers.CuDNNLSTM(neurons=10, batch_input_shape=(params['batch_size'], params['dim']), stateful=True))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')  # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
