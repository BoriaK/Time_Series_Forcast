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
from Models import lstm_model_deep
from Models import forecast_lstm
import os.path
import tensorflow as tf
import argparse

# this version splits training data in to training and testing, and performs evaluation every 50 epochs. also it save
# the entire model as checkpoint

parser = argparse.ArgumentParser()
# parser.add_argument('--resume_epoch', type=int, default=None, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--data', type=str, default=r'../Dataset', help='root directory of the '
                                                                    'dataset')
# parser.add_argument('--save_results', action='store_false', default=True,
#                     help='save result plots after each eval cycle')
# parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--root_chkps', type=str, default=r'./Checkpoints', help='checkpoint folder')
args = parser.parse_args()
print(args)
'''adding types to arguments'''
# args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(args.device)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if not os.path.isdir(args.root_chkps):
    os.mkdir(args.root_chkps)

# dataSetRoot = r'../Dataset'
dataSetRoot = args.data

# checkpoint_filepath = r'./Checkpoints'
checkpoint_filepath = args.root_chkps

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
train, test = supervised_values[0:-100], supervised_values[-100:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

X, y = train_scaled[:, 0:-1], train_scaled[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])

# create the model
batch_size = 1
LSTM_Model = lstm_model_deep(X.shape[1], X.shape[2], batch_size=batch_size, neurons=10)
# nb_epoch = 1000  # number of training epochs should be ~3K
nb_epoch = args.n_epochs

# Train loop
for i in range(nb_epoch):
    print(i)
    LSTM_Model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    LSTM_Model.reset_states()
    if (i + 1) % 50 == 0:
        # walk-forward validation on the test data
        predictions = list()
        for j in range(len(test_scaled)):
            # make one-step forecast
            X1, y1 = test_scaled[j, 0:-1], test_scaled[j, -1]
            yhat = forecast_lstm(LSTM_Model, 1, X1)
            # invert scaling
            yhat = invert_scale(scaler, X1, yhat)
            # store forecast
            predictions.append(yhat)
            expected = raw_values[len(train) + j]
        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-100:], predictions))
        print('Test RMSE: %.3f' % rmse)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
LSTM_Model.predict(train_reshaped, batch_size=1)

CheckPoint = os.path.join(checkpoint_filepath, 'cp_5x10_' + str(nb_epoch) + '_epochs')
LSTM_Model.save(CheckPoint)
print('checkpoint ' + 'cp_5x10_' + str(nb_epoch) + '_epochs ' + 'is saved')
