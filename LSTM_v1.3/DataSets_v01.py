# from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pandas import read_csv
import os


# set of preprocessing functions for LSTM
# load data from .csv
def loadData(data_file_csv):
    dataSetRoot = r'../Dataset'
    csv_data = read_csv(os.path.join(dataSetRoot, data_file_csv), header=0, index_col=0, squeeze=True)
    df = DataFrame(csv_data)  # get data frame from csv
    return df


def splitData(df, t_len):
    # Split the data to training and validation, testing will be from a separate set
    n = len(df)
    # train_df = df[0:int(n * 0.9) - 1]
    train_df = df[0:int(n * t_len)]
    val_df = df[int(n * (1 - t_len)):]
    return train_df, val_df


# Zero Mean and scale the data
def normAndScale(data_df):
    # Zero-Mean the Data
    data_mean = data_df.mean()
    zero_mean_df = data_df - data_mean

    # Normalize the data between [-1,1]
    normed_df = zero_mean_df / zero_mean_df.abs().max()
    return normed_df


# Zero Mean the data
def zeroMean(data_df):
    # Zero-Mean the Data
    data_mean = data_df.mean()
    zero_mean_df = data_df - data_mean

    return zero_mean_df


# reverse Zero Mean and scale - Need to add

def generateWindow(window_size, train_df, val_df, test_df):
    # window size effectively determines the size of the sliding window, and the number of processing units in lstm
    window = WindowGenerator(train_df, val_df, test_df,
                             input_width=window_size,
                             label_width=1,
                             shift=1,
                             label_columns=['Data [Gb]'])
    return window


def generateMultistepWindow(window_size, predictions_size, train_df, val_df, test_df):
    # window size effectively determines the size of the sliding window, and the number of processing units in lstm
    window = WindowGenerator(train_df, val_df, test_df,
                             input_width=window_size,
                             label_width=predictions_size,
                             shift=predictions_size,
                             label_columns=['Data [Gb]'])
    return window


class WindowGenerator:
    # def __init__(self, input_width, label_width, shift,
    #              train_df=train_df, val_df=val_df, test_df=test_df,
    #              label_columns=None):
    def __init__(self, train_df, val_df, test_df, input_width, label_width, shift, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        # self.column_indices = {name: i for i, name in
        #                        enumerate(train_df.columns)}
        self.column_indices = {label_columns[0]: 0}  # for a 1 dimensional dataset

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


# def plot(self, model=None, plot_col='Data [Gb]', max_subplots=1):
#     inputs, labels = self.example
#     plt.figure(figsize=(12, 8))
#     if plot_col is not None:
#         plot_col_index = self.column_indices[plot_col]
#     else:
#         plot_col_index = self.column_indices[0]
#     max_n = min(max_subplots, len(inputs))
#     for n in range(max_n):
#         plt.subplot(max_n, 1, n + 1)
#         plt.ylabel(f'{plot_col} [normed]')
#         plt.plot(self.input_indices, inputs[n, :, plot_col_index],
#                  label='Inputs', marker='.', zorder=-10)
#
#         if self.label_columns:
#             label_col_index = self.label_columns_indices.get(plot_col, None)
#         else:
#             label_col_index = plot_col_index
#
#         if label_col_index is None:
#             continue
#
#         plt.scatter(self.label_indices, labels[n, :, label_col_index],
#                     edgecolors='k', label='Labels', c='#2ca02c', s=64)
#         if model is not None:
#             predictions = model(inputs)
#             plt.scatter(self.label_indices, predictions[n, :, label_col_index],
#                         marker='X', edgecolors='k', label='Predictions',
#                         c='#ff7f0e', s=64)
#
#         if n == 0:
#             plt.legend()
#
#     plt.xlabel('Time [sec]')
#     plt.grid()
#     plt.show()

# create dataset object
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=512, )
    # batch_size=1, )

    ds = ds.map(self.split_window)

    return ds


@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def val(self):
    return self.make_dataset(self.val_df)


@property
def test(self):
    return self.make_dataset(self.test_df)


@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result


WindowGenerator.split_window = split_window
# WindowGenerator.plot = plot
WindowGenerator.make_dataset = make_dataset
WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
