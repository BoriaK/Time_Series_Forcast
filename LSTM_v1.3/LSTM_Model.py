# import IPython
# import IPython.display
import numpy as np
import tensorflow as tf
from pandas import read_csv
import os
from pandas import DataFrame
from matplotlib import pyplot as plt
from DataSets_v01 import WindowGenerator
from Models_v01 import lstm_model
from Models_v01 import compile_and_fit
from Plot_Function import plotFunction

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


Window_Size = 10  # effectively determines the size of the sliding window
LSTM_Window = WindowGenerator(train_df, val_df,
                              input_width=Window_Size,
                              label_width=1,
                              shift=1,
                              label_columns=['Data [Gb]'])

LSTM_Model = lstm_model(Window_Size, LSTM_Window.example[0].shape)
# LSTM_Model = lstm_model(Window_Size)

print('Input shape = [batch, time, features]: ', LSTM_Window.example[0].shape)
print('Output shape:', LSTM_Model(LSTM_Window.example[0]).shape)

Max_Epochs = 10

History = compile_and_fit(LSTM_Model, LSTM_Window, epochs=Max_Epochs)
# IPython.display.clear_output()
ModelEval = LSTM_Model.evaluate(LSTM_Window.val)

# Generate and Plot a complete Evaluation window

Eval_Predictions = []
for i in range(len(LSTM_Window.val_df) - Window_Size):
    Eval_Input = LSTM_Window.val_df.values[i:Window_Size + i].reshape([1, -1, 1])
    # print(Eval_Input)
    Eval_Prediction = LSTM_Model.predict(Eval_Input)
    Eval_Predictions.append(Eval_Prediction[0][0])
Eval_Labels = LSTM_Window.val_df.values[Window_Size:]
# print(Eval_Labels)

plotFunction(Eval_Labels, Eval_Predictions, Window_Size, ModelEval)

# performance = {}
# performance['Dense'] = LSTM_Model.evaluate(LSTM_Window.test, verbose=0)