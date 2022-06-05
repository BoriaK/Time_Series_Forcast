import IPython
import IPython.display
import numpy as np
import tensorflow as tf
from pandas import read_csv
import os
from pandas import DataFrame
from matplotlib import pyplot as plt
from DataSets_v01 import WindowGenerator
from Models_v01 import deep_lstm_model
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


Num_Neurons = 10  # effectively determines the size of the sliding window
LSTM_Window = WindowGenerator(train_df, val_df,
                              input_width=Num_Neurons,
                              label_width=1,
                              shift=1,
                              label_columns=['Data [Gb]'])

# print(LSTM_Window)

Deep_RNN_LSTM_Model = deep_lstm_model()

print('Input shape:', LSTM_Window.example[0].shape)
print('Output shape:', Deep_RNN_LSTM_Model(LSTM_Window.example[0]).shape)

Max_Epochs = 100

History = compile_and_fit(Deep_RNN_LSTM_Model, LSTM_Window, epochs=Max_Epochs)
IPython.display.clear_output()
val_performance = {'Deep RNN LSTM': Deep_RNN_LSTM_Model.evaluate(LSTM_Window.val)}
# performance = {}
# performance['Dense'] = Dense_Model.evaluate(Single_Step_Window.test, verbose=0)

# Conv_Window.plot(Multi_Step_Conv_Model)

# Generate and Plot a complete Evaluation window
Label_Width = len(val_df)
Input_Width = Label_Width
Wide_LSTM_Window = WindowGenerator(train_df, val_df,
                                   input_width=Input_Width,
                                   label_width=Label_Width,
                                   shift=1,
                                   label_columns=['Data [Gb]'])

Wide_LSTM_Window.plot(Deep_RNN_LSTM_Model)
####################################################
