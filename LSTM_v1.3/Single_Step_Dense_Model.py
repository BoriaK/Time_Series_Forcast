import numpy as np
import tensorflow as tf
from pandas import read_csv
import os
from pandas import DataFrame
from matplotlib import pyplot as plt
from DataSets_v01 import WindowGenerator
from Models_v01 import dense
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

Dense_Model = dense()

Max_Epochs = 20

History = compile_and_fit(Dense_Model, Single_Step_Window, epochs=Max_Epochs)

val_performance = {'Dense': Dense_Model.evaluate(Single_Step_Window.val)}
# performance = {}
# performance['Dense'] = Dense_Model.evaluate(Single_Step_Window.test, verbose=0)

########for Printing Only#########################
Wide_Window = WindowGenerator(train_df, val_df,
                              input_width=len(val_df), label_width=len(val_df), shift=1,
                              label_columns=['Data [Gb]'])

# print('Input shape:', Wide_Window.example[0].shape)
# print('Output shape:', baseline(Wide_Window.example[0]).shape)
Wide_Window.plot(Dense_Model)
####################################################

