import numpy as np
import tensorflow as tf
from pandas import read_csv
import os
from pandas import DataFrame
from matplotlib import pyplot as plt
from DataSets_v01 import WindowGenerator
from Models_v01 import Baseline

# load the dataset
dataSetRoot = r'../Dataset'
Series = read_csv(os.path.join(dataSetRoot, 'Traffic_Data_1k.csv'), header=0, index_col=0, squeeze=True)
df = DataFrame(Series)  # get data frame from csv
# Split the data to training and validation, testing will be from a separate set
n = len(df)
train_df = df[0:int(n * 0.9) - 1]  # if take from csv
val_df = df[int(n * 0.9):]

# Normalize the Data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
# test_df = (test_df - train_mean) / train_std

# Single step model
Single_Step_Window = WindowGenerator(train_df, val_df,
                                     input_width=1, label_width=1, shift=1,
                                     label_columns=None)
# print(Single_Step_Window)

# Wide_Window = WindowGenerator(train_df, val_df,
#                               input_width=24, label_width=24, shift=1,
#                               label_columns=None)

for example_inputs, example_labels in Single_Step_Window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

# baseline = Baseline(label_index=Single_Step_Window.column_indices['Data [Gb]'])
baseline = Baseline()

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {'Baseline': baseline.evaluate(Single_Step_Window.val)}
# performance = {}
# performance['Baseline'] = baseline.evaluate(Single_Step_Window.test, verbose=0)

# Generate and Plot a complete Evaluation window

Label_Width = len(val_df)
Input_Width = Label_Width
Wide_Baseline_Window = WindowGenerator(train_df, val_df,
                                       input_width=Input_Width,
                                       label_width=Label_Width,
                                       shift=1,
                                       label_columns=None)

# print('Input shape:', Wide_Baseline_Window.example[0].shape)
# print('Output shape:', baseline(Wide_Baseline_Window.example[0]).shape)
Wide_Baseline_Window.plot(baseline)
