import os.path
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
import numpy


# set of preprocessing functions for LSTM

# date-time parsing function for loading the shampoo dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

#
# def scaleTrain(train):
#     # fit scaler
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaler = scaler.fit(train)
#     # transform train
#     train = train.reshape(train.shape[0], train.shape[1])
#     train_scaled = scaler.transform(train)
#     return scaler, train_scaled
#
#
# def scaleTest(test):
#     # fit scaler
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaler = scaler.fit(test)
#     # transform train
#     test = test.reshape(test.shape[0], test.shape[1])
#     test_scaled = scaler.transform(test)
#     return scaler, test_scaled


def scaleUni(data):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data)
    # transform data
    data = data.reshape(data.shape[0], data.shape[1])
    data_scaled = scaler.transform(data)
    return scaler, data_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# dataSetRoot = r'C:\Users\bkupcha\OneDrive - Intel Corporation\Documents\PythonProjects\Time_Series_Forcast\Dataset'
#
# # load dataset
# series = read_csv(os.path.join(dataSetRoot, 'shampoo-sales.csv'), header=0, parse_dates=[0], index_col=0, squeeze=True,
#                   date_parser=parser)
# print(series.head())
#
# # transform to be stationary
# differenced = difference(series, 1)
# print(differenced.head())
# # invert transform
# inverted = list()
# for i in range(len(differenced)):
#     value = inverse_difference(series, differenced[i], len(series) - i)
#     inverted.append(value)
# inverted = Series(inverted)
# print(inverted.head())
#
# # transform to supervised learning
# X = series.values
# supervised = timeseries_to_supervised(X, 1)
# print(supervised.head())
#
# # transform scale
# X = series.values
# X = X.reshape(len(X), 1)
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler = scaler.fit(X)
# scaled_X = scaler.transform(X)
# scaled_series = Series(scaled_X[:, 0])
# print(scaled_series.head())
# # invert transform
# inverted_X = scaler.inverse_transform(scaled_X)
# inverted_series = Series(inverted_X[:, 0])
# print(inverted_series.head())
