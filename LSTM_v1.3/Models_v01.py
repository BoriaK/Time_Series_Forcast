import numpy as np
import tensorflow as tf
from pandas import read_csv
import os
from pandas import DataFrame
from matplotlib import pyplot as plt
from DataSets_v01 import WindowGenerator


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


# Models:
def linear():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])
    return model


def dense():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    return model


def dense_multi_out(num_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=num_features)
    ])
    return model


def multi_step_dense():
    model = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])
    return model


def conv_model(conv_width):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(conv_width,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])
    return model


# batch_input_shape=
def lstm_model():
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        # tf.keras.layers.LSTM(32, return_sequences=True, batch_input_shape=LSTM_Window.example[0].shape, stateful=True),
        tf.keras.layers.LSTM(units=32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    return model


def lstm_model_multi_out(num_features):
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        # tf.keras.layers.LSTM(32, return_sequences=True, batch_input_shape=LSTM_Window.example[0].shape, stateful=True),
        tf.keras.layers.LSTM(units=32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=num_features)
    ])
    return model


def residual_lstm_multi_out(num_features):
    model = ResidualWrapper(
        tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(
                num_features,
                # The predicted deltas should start small.
                # Therefore, initialize the output layer with zeros.
                kernel_initializer=tf.initializers.zeros())
        ]))
    return model


def deep_lstm_model():
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(units=32, return_sequences=True),
        tf.keras.layers.LSTM(units=32, return_sequences=True),
        tf.keras.layers.LSTM(units=32, return_sequences=True),
        tf.keras.layers.LSTM(units=32, return_sequences=True),
        tf.keras.layers.LSTM(units=32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    return model


class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta


# Operational functions
def compile_and_fit(model, window, epochs, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min',
                                                      verbose=1)

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history
