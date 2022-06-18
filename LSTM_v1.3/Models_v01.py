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


# The resulting output, when using the "valid" padding option, has a shape of: output_shape = (input_shape - pool_size + 1) / strides)
# The resulting output shape when using the "same" padding option is: output_shape = input_shape / strides
def deep_conv_model(conv_width):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=(conv_width,),
                               activation='relu'),
        # tf.keras.layers.BatchNormalization(axis=-1,),
        # tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same'),
        # tf.keras.layers.MaxPooling1D(pool_size=1),
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(conv_width,),
                               activation='relu'),
        # tf.keras.layers.BatchNormalization(axis=-1,),
        # tf.keras.layers.MaxPooling1D(pool_size=1),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])
    return model


def lstm_model(lstm_units):
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(units=lstm_units, return_sequences=False,
                             stateful=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    return model


def cnn_lstm(window_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(window_size,),
                               activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=1),
        # tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.Dense(units=32, activation='relu'),
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(units=window_size, return_sequences=False,
                             stateful=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    return model


def cnn_deep_lstm(window_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(window_size,),
                               activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=1),
        # tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.LSTM(units=window_size, return_sequences=True,
                             stateful=False),
        tf.keras.layers.LSTM(units=window_size, return_sequences=True,
                             stateful=False),
        tf.keras.layers.LSTM(units=window_size, return_sequences=True,
                             stateful=False),
        tf.keras.layers.LSTM(units=window_size, return_sequences=True,
                             stateful=False),
        tf.keras.layers.LSTM(units=window_size, return_sequences=False,
                             stateful=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    return model


# batch_input_shape=
def lstm_model_stateful(window_size, lstm_window_shape):
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(units=window_size, batch_input_shape=lstm_window_shape, return_sequences=False,
                             stateful=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    return model


def lstm_model_multi_out(num_features):
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        # tf.keras.layers.LSTM(32, return_sequences=True,
        # batch_input_shape=LSTM_Window.example[0].shape, stateful=True),
        tf.keras.layers.LSTM(units=32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=num_features)
    ])
    return model


def deep_lstm_model(lstm_units):
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(units=lstm_units, return_sequences=True,
                             stateful=False),
        tf.keras.layers.LSTM(units=lstm_units, return_sequences=True,
                             stateful=False),
        tf.keras.layers.LSTM(units=lstm_units, return_sequences=True,
                             stateful=False),
        tf.keras.layers.LSTM(units=lstm_units, return_sequences=True,
                             stateful=False),
        tf.keras.layers.LSTM(units=lstm_units, return_sequences=False,
                             stateful=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    return model


def deep_lstm_model_stateful(window_size, lstm_window_shape):
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(units=window_size, batch_input_shape=lstm_window_shape, return_sequences=True,
                             stateful=True),
        tf.keras.layers.LSTM(units=window_size, batch_input_shape=lstm_window_shape, return_sequences=True,
                             stateful=True),
        tf.keras.layers.LSTM(units=window_size, batch_input_shape=lstm_window_shape, return_sequences=True,
                             stateful=True),
        tf.keras.layers.LSTM(units=window_size, batch_input_shape=lstm_window_shape, return_sequences=True,
                             stateful=True),
        tf.keras.layers.LSTM(units=window_size, batch_input_shape=lstm_window_shape, return_sequences=False,
                             stateful=True),
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


def residual_lstm(window_size):
    model = ResidualWrapper(
        tf.keras.Sequential([
            tf.keras.layers.LSTM(units=window_size, return_sequences=True),
            tf.keras.layers.Dense(
                units=1,
                # The predicted deltas should start small.
                # Therefore, initialize the output layer with zeros.
                kernel_initializer=tf.initializers.zeros())
        ]))
    return model


def residual_deep_lstm_model(window_size):
    model = ResidualWrapper(
        tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(units=window_size, return_sequences=True,
                                 stateful=False),
            tf.keras.layers.LSTM(units=window_size, return_sequences=True,
                                 stateful=False),
            tf.keras.layers.LSTM(units=window_size, return_sequences=True,
                                 stateful=False),
            tf.keras.layers.LSTM(units=window_size, return_sequences=True,
                                 stateful=False),
            tf.keras.layers.LSTM(units=window_size, return_sequences=True,
                                 stateful=False),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(
                units=1,
                # The predicted deltas should start small.
                # Therefore, initialize the output layer with zeros.
                kernel_initializer=tf.initializers.zeros())
        ]))
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


def multi_linear_model(out_steps, num_features):
    model = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(out_steps * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([out_steps, num_features])
    ])
    return model


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(units=1)


def call(self, inputs, training=None):
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []
    # Initialize the LSTM state.
    prediction, state = self.warmup(inputs)

    # Insert the first prediction.
    predictions.append(prediction)

    # Run the rest of the prediction steps.
    for n in range(1, self.out_steps):
        # Use the last prediction as input.
        x = prediction
        # Execute one lstm step.
        x, state = self.lstm_cell(x, states=state,
                                  training=training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output.
        predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions


FeedBack.call = call


def warmup(self, inputs):
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    x, *state = self.lstm_rnn(inputs)

    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state


FeedBack.warmup = warmup


# Operational functions
def compile_and_fit(model, window, epochs, patience=5):
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


def compileModel(model):
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    return model


def fitModel(model, window, epochs, patience=3):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min',
                                                      verbose=1)
    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        # callbacks=[early_stopping]
                        )
    return history
