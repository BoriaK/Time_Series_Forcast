import numpy as np
import tensorflow as tf
from pandas import read_csv
import os
from pandas import DataFrame
from matplotlib import pyplot as plt
from DataSets_v01 import WindowGenerator
from Models_v01 import linear
from Models_v01 import compile_and_fit

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
                                     label_columns=['Data [Gb]'])
# print(Single_Step_Window)

# for example_inputs, example_labels in single_step_window.train.take(1):
#     print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#     print(f'Labels shape (batch, time, features): {example_labels.shape}')


########for Printing Only#########################
Wide_Window = WindowGenerator(train_df, val_df,
                              input_width=24, label_width=24, shift=1,
                              label_columns=['Data [Gb]'])
# print('Input shape:', Wide_Window.example[0].shape)
# print('Output shape:', baseline(Wide_Window.example[0]).shape)

# Wide_Window.plot(baseline)
##########################################################

Linear_Model = linear()
print('Input shape:', Single_Step_Window.example[0].shape)
print('Output shape:', Linear_Model(Single_Step_Window.example[0]).shape)

Max_Epochs = 20

History = compile_and_fit(Linear_Model, Single_Step_Window, epochs=Max_Epochs)

val_performance = {'Linear': Linear_Model.evaluate(Single_Step_Window.val)}
# performance = {}
# performance['Linear'] = Linear_Model.evaluate(Single_Step_Window.test, verbose=0)

# print('Input shape:', Wide_Window.example[0].shape)
# print('Output shape:', baseline(Wide_Window.example[0]).shape)
Wide_Window.plot(Linear_Model)

# Just for debug plot the model weights##############
# plt.bar(x=range(len(train_df.columns)),
#         height=linear.layers[0].kernel[:, 0].numpy())
# axis = plt.gca()
# axis.set_xticks(range(len(train_df.columns)))
# _ = axis.set_xticklabels(train_df.columns, rotation=90)
###########################################################