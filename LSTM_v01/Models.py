from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# fit an LSTM network to training data
def fit_lstm_w_cpt(train, batch_size, nb_epoch, neurons, callback):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False, callbacks=[callback])  # Pass
        # callback to training
        model.reset_states()
    return model


# try - create LSTM model
def lstm_model(x_shape1, x_shape2, batch_size, neurons):
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, x_shape1, x_shape2), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def lstm_model_deep(x_shape1, x_shape2, batch_size, neurons):  # a model with 5 deep LSTM layers
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, batch_input_shape=(batch_size, x_shape1, x_shape2), stateful=True))
    model.add(LSTM(neurons, return_sequences=True, batch_input_shape=(batch_size, x_shape1, x_shape2), stateful=True))
    model.add(LSTM(neurons, return_sequences=True, batch_input_shape=(batch_size, x_shape1, x_shape2), stateful=True))
    model.add(LSTM(neurons, return_sequences=True, batch_input_shape=(batch_size, x_shape1, x_shape2), stateful=True))
    model.add(LSTM(neurons, batch_input_shape=(batch_size, x_shape1, x_shape2), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]
