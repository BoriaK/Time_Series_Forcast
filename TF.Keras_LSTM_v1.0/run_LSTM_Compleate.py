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
from DataSets import scale
from DataSets import invert_scale
from Models import fit_lstm
from Models import forecast_lstm
import os.path
import tensorflow as tf

dataSetRoot = r'C:\Users\bkupcha\OneDrive - Intel Corporation\Documents\PythonProjects\Time_Series_Forcast_LSTM\Dataset'

# checkpoint_filepath = r'C:\Users\bkupcha\OneDrive - Intel ' \
#                       r'Corporation\Documents\PythonProjects\Time_Series_Forcast\Checkpoints'
# CheckPoint = os.path.join(checkpoint_filepath, 'cp1.ckpt')

# Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CheckPoint,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# load dataset
# series = read_csv(os.path.join(dataSetRoot, 'shampoo-sales.csv'), header=0, parse_dates=[0], index_col=0, squeeze=True,
#                   date_parser=parser)
series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data.csv'), header=0, index_col=0, squeeze=True)
print(series.head())

# transform data to be stationary
raw_values = series.values
# diff_values = difference(raw_values, 1)
# print(diff_values)

# transform data to be supervised learning
# supervised = timeseries_to_supervised(diff_values, 1)
supervised = timeseries_to_supervised(raw_values, 1)
supervised_values = supervised.values
print(supervised_values)

# split data into train and test-sets
train, test = supervised_values[0:-100], supervised_values[-100:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 3000, 5)  # train dataset, batch size, num of epochs, num of neurons
# lstm_model = fit_lstm(train_scaled, 1, 3, 5)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)



# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    # yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)
    # expected = raw_values[len(train) + i + 1]
    expected = raw_values[len(train) + i]
    # print('Sec=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))
    # print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-100:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(raw_values[-100:])
plt.plot(predictions)
plt.xlabel('Time [s]')
plt.ylabel('Data Volume [Gb]')
plt.grid()
plt.title('Data Volume over Time')
plt.legend(['Validation dataset', 'Predictions'])
plt.show()
