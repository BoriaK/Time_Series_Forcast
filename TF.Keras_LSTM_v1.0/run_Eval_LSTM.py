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
# # load dataset
# series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data_10k.csv'), header=0, index_col=0, squeeze=True)
# print(series.head())

# load training dataset
Train_series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data_10k.csv'), header=0, index_col=0, squeeze=True)

# load testing dataset
Test_series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data_1k.csv'), header=0, index_col=0, squeeze=True)

# transform data to be stationary
# raw_values = series.values
raw_Train_values = Train_series.values
raw_Test_values = Test_series.values

# # transform data to be supervised learning
# supervised = timeseries_to_supervised(raw_values, 1)
# supervised_values = supervised.values
# # print(supervised_values)
supervised_Train = timeseries_to_supervised(raw_Train_values, 1)
supervised_Train_values = supervised_Train.values
supervised_Test = timeseries_to_supervised(raw_Test_values, 1)
supervised_Test_values = supervised_Test.values

# # split data into train and test-sets
# train, test = supervised_values[0:-100], supervised_values[-100:]
train = supervised_Train_values
test = supervised_Test_values

# # transform the scale of the data
# scaler, train_scaled, test_scaled = scale(train, test)
# transform the scale of the data
_, train_scaled = scaleUni(train)
scaler, test_scaled = scaleUni(test)

X, y = train_scaled[:, 0:-1], train_scaled[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])

# Create a basic model instance
batch_size = 1
LSTM_Model = lstm_model_deep(X.shape[1], X.shape[2], batch_size=batch_size, neurons=10)

# Loads the weights
LSTM_Model.load_weights(CheckPoint)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
LSTM_Model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(LSTM_Model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    # yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)
    # expected = raw_values[len(train) + i + 1]
    # expected = raw_values[len(train) + i]
    expected = raw_Test_values[i]
    # print('Sec=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))


# report performance
# rmse = sqrt(mean_squared_error(raw_values[-100:], predictions))
rmse = sqrt(mean_squared_error(raw_Test_values, predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
# plt.plot(raw_values[-100:])
plt.plot(raw_Test_values[-200:])
# plt.plot(predictions)
plt.plot(predictions[-200:])
plt.xlabel('Time [s]')
plt.ylabel('Data Volume [Gb]')
plt.grid()
plt.title('Data Volume over Time')
plt.legend(['Validation dataset', 'Predictions'])
plt.show()
