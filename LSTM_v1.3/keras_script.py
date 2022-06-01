import IPython
import IPython.display
import numpy as np
# from tensorflow.keras.models import Sequential
import tensorflow as tf
# from DataSets import WindowGenerator
from pandas import read_csv
import os
from pandas import DataFrame

# load dataset
dataSetRoot = r'../Dataset'
Series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data_1k.csv'), header=0, index_col=0, squeeze=True)
raw_values = Series.values

# split the data to training and testing
# df = DataFrame(Series)
df = DataFrame(raw_values)
n = len(df)
# train_df = df[0:int(n * 0.9)-1]
train_df = df[0:int(n * 0.9)]
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


WindowGenerator.split_window = split_window

###Debug####
w1 = WindowGenerator(input_width=24, label_width=1, shift=1,
                     label_columns=['Data [Gb]'])
w2 = WindowGenerator(input_width=6, label_width=1, shift=1)
w3 = WindowGenerator(input_width=1, label_width=1, shift=1,
                     label_columns=None
                     )
print(w3)

# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w3.total_window_size]),
                           np.array(train_df[100:100 + w3.total_window_size]),
                           np.array(train_df[200:200 + w3.total_window_size])])

example_inputs, example_labels = w3.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

############
