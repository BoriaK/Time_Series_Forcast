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
from matplotlib import pyplot as plt
import numpy
from DataSets import parser
from DataSets import timeseries_to_supervised
from DataSets import difference
from DataSets import inverse_difference
from DataSets import scaleUni
from DataSets import invert_scale
from Models import lstm_model_deep
from Models import forecast_lstm
import os.path
import tensorflow as tf

dataSetRoot = r'.\Dataset'

checkpoint_filepath = r'Checkpoints'

# load dataset
series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data_1k.csv'), header=0, index_col=0, squeeze=True)
# print(series.head())

# transform data to be stationary
raw_values = series.values

# transform data to be supervised learning
supervised = timeseries_to_supervised(raw_values, 1)
supervised_values = supervised.values
# print(supervised_values)

# split data into train and test-sets
# train, test = supervised_values[0:-100], supervised_values[-100:]
train = supervised_values

# transform the scale of the data
# scaler, train_scaled, test_scaled = scale(train, test)
scaler, train_scaled = scaleUni(train)

X, y = train_scaled[:, 0:-1], train_scaled[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])

# create the model
batch_size = 1
LSTM_Model = lstm_model_deep(X.shape[1], X.shape[2], batch_size=batch_size, neurons=10)
nb_epoch = 500  # number of training epochs should be ~3K

# Train loop
for i in range(nb_epoch):
    print(i)
    LSTM_Model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    LSTM_Model.reset_states()

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
LSTM_Model.predict(train_reshaped, batch_size=1)

CheckPoint = os.path.join(checkpoint_filepath, 'cp_5x10_' + str(nb_epoch) + '_epochs')
# LSTM_Model.save(CheckPoint)
print('checkpoint ' + 'cp_5x10_' + str(nb_epoch) + '_epochs ' + 'is saved')
