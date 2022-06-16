import os
import tensorflow as tf
from DataSets_v01 import loadData
from DataSets_v01 import splitData
from DataSets_v01 import normAndScale
from DataSets_v01 import zeroMean
from DataSets_v01 import generateMultistepWindow
from Additional_Functions import makePredictionsAndLabelsTestMultiStepOut
from Additional_Functions import plotFunction
from matplotlib import pyplot as plt

# Loads the pre-trained model
checkpoint_filepath = r'./Checkpoints'
CheckPoint = os.path.join(checkpoint_filepath, 'Batch_32_Out_8_1k_samples_Random_Data_w_16_5000_epochs_MultiStep_Linear_Model')
MultiStep_Model = tf.keras.models.load_model(CheckPoint)

# load the dataset
DF = loadData('Traffic_Data_1k.csv')

Normed_Test_DF = zeroMean(DF)

Window_Size = 16
OUT_STEPS = 8
Multi_Window = generateMultistepWindow(Window_Size, OUT_STEPS, train_df=None, val_df=None, test_df=Normed_Test_DF)

# Generate Predictions and Labels Array
Test_Labels, Test_Predictions = makePredictionsAndLabelsTestMultiStepOut(MultiStep_Model, Multi_Window, Window_Size)

# plot and compare the models
ModelEval = MultiStep_Model.evaluate(Multi_Window.test)
Max_Epochs = 5000
plotFunction(Test_Labels, Test_Predictions, Window_Size, ModelEval, 'Batch_32_1x16_LSTM_Model', Max_Epochs)

