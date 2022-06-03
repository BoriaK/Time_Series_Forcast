import IPython
import IPython.display
import numpy as np
# from tensorflow.keras.models import Sequential
import tensorflow as tf
# from DataSets import WindowGenerator
from pandas import read_csv
import os
from pandas import DataFrame
from matplotlib import pyplot as plt

# load dataset
dataSetRoot = r'../Dataset'
Series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data_1k.csv'), header=0, index_col=0, squeeze=True)
raw_values = Series.values

# split the data to training and testing
# df = Series  # get data frame from csv
df = DataFrame(Series)
# df = DataFrame(raw_values)  # get data frame from time series
n = len(df)
train_df = df[0:int(n * 0.9) - 1]  # if take from csv
# train_df = df[0:int(n * 0.9)]  # if take from time series
# val_df = df[int(n*0.7):int(n*0.9)]
val_df = df[int(n * 0.9):]


# test_df = df[int(n*0.9):]


class WindowGenerator():
    # def __init__(self, input_width, label_width, shift,
    #              train_df=train_df, val_df=val_df, test_df=test_df,
    #              label_columns=None):
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        # self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
        # self.column_indices = {train_df.name: 0}  # for a 1 dimentional dataset

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


###Example Window generator###############################
# w1 = WindowGenerator(input_width=24, label_width=1, shift=1,
#                      label_columns=['Nex Time Sample'])
w2 = WindowGenerator(input_width=6, label_width=1, shift=1)


# w3 = WindowGenerator(input_width=1, label_width=1, shift=1,
#                      label_columns=None
#                      )
# print(w3)


#################################################

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


WindowGenerator.split_window = split_window


###Example Split window###############################
# Stack three slices, the length of the total window.
# This method works if I use raw data from a time series
# example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
#                            np.array(train_df[100:100 + w2.total_window_size]),
#                            np.array(train_df[200:200 + w2.total_window_size])])

# This method works if I use raw data from .csv
# example_window = tf.stack([np.array(train_df[:w2.total_window_size-1]),
#                            np.array(train_df[100:100 + w2.total_window_size-1]),
#                            np.array(train_df[200:200 + w2.total_window_size-1])])
#
# example_inputs, example_labels = w2.split_window(example_window)
#
# print('All shapes are: (batch, time, features)')
# print(f'Window shape: {example_window.shape}')
# print(f'Inputs shape: {example_inputs.shape}')
# print(f'Labels shape: {example_labels.shape}')


###########################################################

def plot(self, model=None, plot_col='Data [Gb]', max_subplots=3):
    # def plot(self, model=None, plot_col=None, max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    if plot_col is not None:
        plot_col_index = self.column_indices[plot_col]
    else:
        plot_col_index = self.column_indices[0]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [sec]')


WindowGenerator.plot = plot


####### Example plot#################
# w2.example = example_inputs, example_labels
# w2.plot()


##########################################################


def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32, )

    ds = ds.map(self.split_window)

    return ds


WindowGenerator.make_dataset = make_dataset


### Dataset Debug################
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


WindowGenerator.train = train
WindowGenerator.val = val
# WindowGenerator.test = test
WindowGenerator.example = example

###################################################################

# Each element is an (inputs, label) pair.
w2.train.element_spec

for example_inputs, example_labels in w2.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
