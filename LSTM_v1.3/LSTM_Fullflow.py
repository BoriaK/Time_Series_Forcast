# import IPython
# import IPython.display
import numpy as np
# from tensorflow.keras.models import Sequential
import tensorflow as tf
# from DataSets import WindowGenerator
from pandas import read_csv
import os
from pandas import DataFrame
from matplotlib import pyplot as plt

#####This is an example of full flow of:
# load data -> generate window -> create model -> compile and fit


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

# Normalize the Data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std


class WindowGenerator:
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


def plot(self, model=None, plot_col='Data [Gb]', max_subplots=1):
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
    plt.show()


def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=1, )

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
WindowGenerator.plot = plot
WindowGenerator.make_dataset = make_dataset
WindowGenerator.train = train
WindowGenerator.val = val
# WindowGenerator.test = test
WindowGenerator.example = example

MAX_EPOCHS = 10


def compile_and_fit(model, window, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


# wide_window = WindowGenerator(input_width=24, label_width=24, shift=1, label_columns=['Data [Gb]'])

Num_Neurons = 10
LSTM_Window = WindowGenerator(input_width=Num_Neurons, label_width=1, shift=1, label_columns=['Data [Gb]'])
print('Input shape = [batch, time, features]: ', LSTM_Window.example[0].shape)

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    # tf.keras.layers.LSTM(32, return_sequences=True, batch_input_shape=LSTM_Window.example[0].shape, stateful=True),
    tf.keras.layers.LSTM(units=Num_Neurons, batch_input_shape=LSTM_Window.example[0].shape, return_sequences=True, stateful=True),
    # tf.keras.layers.LSTM(units=Num_Neurons, batch_input_shape=LSTM_Window.example[0].shape, return_sequences=True),
    # tf.keras.layers.LSTM(units=32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

print('Output shape:', lstm_model(LSTM_Window.example[0]).shape)

# print('Input shape = [batch, time, features]: ', LSTM_Window.example[0].shape)
# print('Output shape:', lstm_model(LSTM_Window.example[0]).shape)

History = compile_and_fit(lstm_model, LSTM_Window)

PredictionsVal = lstm_model.predict(LSTM_Window.val_df.values[0:10])
# PredictionsEval = lstm_model.evaluate(LSTM_Window.val)
# PredictionsTrain = lstm_model.predict(LSTM_Window.train)

LSTM_Window.plot(lstm_model)

#############For printing Only####################
# Label_Width = len(val_df)
# Input_Width = Label_Width
# Wide_LSTM_Window = WindowGenerator(
#                                    input_width=Input_Width,
#                                    label_width=Label_Width,
#                                    shift=1,
#                                    label_columns=['Data [Gb]'])
#
# # print('Input shape:', Wide_LSTM_Window.example[0].shape)
# # print('Output shape:', lstm_model(Wide_LSTM_Window.example[0]).shape)
# Wide_LSTM_Window.plot(lstm_model)
# print('')
