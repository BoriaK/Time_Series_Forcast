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

dataSetRoot = r'C:\Users\bkupcha\OneDrive - Intel Corporation\Documents\PythonProjects\Time_Series_Forcast_LSTM\Dataset'

checkpoint_filepath = r'/Checkpoints'
CheckPoint = os.path.join(checkpoint_filepath, 'cp.ckpt')

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CheckPoint,
                                                 save_weights_only=True,
                                                 verbose=1)
# load dataset
series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data_10k.csv'), header=0, index_col=0, squeeze=True)
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
nb_epoch = 10  # number of training epochs should be ~3K

# Train loop
for i in range(nb_epoch):
    print(i)
    LSTM_Model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False, callbacks=[cp_callback])  # Pass
    # callback to training
    LSTM_Model.reset_states()
