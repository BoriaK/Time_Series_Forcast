from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
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
import argparse

# this version load entire model, and runs on an independent test set

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of tested model')
args = parser.parse_args()
print(args)

if not tf.config.list_physical_devices('GPU'):
    Device = 'cpu'
else:
    Device = 'cuda'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

nb_epoch = args.n_epochs

dataSetRoot = r'../Dataset'

checkpoint_filepath = r'./Checkpoints'
# CheckPoint = os.path.join(checkpoint_filepath, 'cp_5x10_' + str(nb_epoch) + '_epochs_' + Device)
CheckPoint = os.path.join(checkpoint_filepath, 'cp_5x10_500_epochs')

# load testing dataset
Test_series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data_1k.csv'), header=0, index_col=0, squeeze=True)

# transform data to be stationary
raw_Test_values = Test_series.values

# # transform data to be supervised learning
supervised_Test = timeseries_to_supervised(raw_Test_values, 1)
supervised_Test_values = supervised_Test.values
# print(supervised_Test_values)

test = supervised_Test_values

# transform the scale of the data
scaler, test_scaled = scaleUni(test)

# Loads the pre-trained model
LSTM_Model = tf.keras.models.load_model(CheckPoint)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(LSTM_Model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # store forecast
    predictions.append(yhat)
    expected = raw_Test_values[i]

# report performance
rmse = sqrt(mean_squared_error(raw_Test_values, predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(raw_Test_values[-200:])
plt.plot(predictions[-200:])
############# for debug:
# plt.plot(raw_Test_values[:200])
# plt.plot(predictions[:200])
##########################
plt.xlabel('Time [s]')
plt.ylabel('Data Volume [Gb]')
plt.grid()
plt.title('Data Volume over Time')
plt.legend(['Validation dataset', 'Predictions'])
plt.show()
# plt.savefig('./Result_Plots/' + 'cp_5x10_' + str(nb_epoch) + '_epochs.png', bbox_inches='tight')
