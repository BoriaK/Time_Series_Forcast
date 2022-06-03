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
from DataSets import WindowGenerator

# load dataset
dataSetRoot = r'../Dataset'
Series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data_1k.csv'), header=0, index_col=0, squeeze=True)
# raw_values = Series.values
# print(raw_values)

df = DataFrame(Series)  # get data frame from csv
# Split the data to training and validation, testing will be from a separate set
n = len(df)
train_df = df[0:int(n * 0.9) - 1]  # if take from csv
val_df = df[int(n * 0.9):]

###Example Window generator###############################
# w1 = WindowGenerator(input_width=24, label_width=1, shift=1,
#                      label_columns=['Nex Time Sample'])
w2 = WindowGenerator(train_df, val_df, input_width=6, label_width=1, shift=1)

# w3 = WindowGenerator(input_width=1, label_width=1, shift=1,
#                      label_columns=None
#                      )
# print(w2)


#########################################################


###Example Split window###############################
# Stack three slices, the length of the total window.
# This method works if I use raw data from a time series
# example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
#                            np.array(train_df[100:100 + w2.total_window_size]),
#                            np.array(train_df[200:200 + w2.total_window_size])])

# This method works if I use raw data from .csv
example_window = tf.stack([np.array(train_df[:w2.total_window_size-1]),
                           np.array(train_df[100:100 + w2.total_window_size-1]),
                           np.array(train_df[200:200 + w2.total_window_size-1])])
#
example_inputs, example_labels = w2.split_window(example_window)
#
# print('All shapes are: (batch, time, features)')
# print(f'Window shape: {example_window.shape}')
# print(f'Inputs shape: {example_inputs.shape}')
# print(f'Labels shape: {example_labels.shape}')

###########################################################

####### Example plot#################
# w2.example = example_inputs, example_labels
# w2.plot()

##########################################################

### Dataset Debug################

# Each element is an (inputs, label) pair.
w2.train.element_spec

for example_inputs, example_labels in w2.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
