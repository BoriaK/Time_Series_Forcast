import os
import tensorflow as tf
from DataSets_v01 import loadData
from DataSets_v01 import splitData
from DataSets_v01 import normAndScale
from DataSets_v01 import zeroMean
from DataSets_v01 import generateWindow
from Additional_Functions import makePredictionsAndLabelsTest
from Additional_Functions import plotFunction
from matplotlib import pyplot as plt

# Loads the pre-trained model
checkpoint_filepath = r'./Checkpoints'
checkpoint_name = 'Batch_64_LSTM_Model_1x128_Window_512_1k_samples_Random_Data_d_0.2_5000_epochs'
CheckPoint = os.path.join(checkpoint_filepath, checkpoint_name)
LSTM_Model = tf.keras.models.load_model(CheckPoint)

# load the dataset
DF = loadData('Traffic_Data_d_0.2_1k_Samples.csv')

Normed_Test_DF = zeroMean(DF)

Window_Size = 512
LSTM_Window = generateWindow(Window_Size, train_df=None, val_df=None, test_df=Normed_Test_DF)

# Generate Predictions and Labels Array
Test_Labels, Test_Predictions = makePredictionsAndLabelsTest(LSTM_Model, LSTM_Window, Window_Size)

# plot and compare the models
ModelEval = LSTM_Model.evaluate(LSTM_Window.test)
plotFunction(Test_Labels, Test_Predictions, Window_Size, ModelEval, checkpoint_name)
