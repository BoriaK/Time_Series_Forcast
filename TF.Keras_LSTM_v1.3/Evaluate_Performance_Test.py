import os
import tensorflow as tf
from DataSets_v01 import loadData
from DataSets_v01 import splitData
from DataSets_v01 import normAndScale
from DataSets_v01 import zeroMean
from DataSets_v01 import generateWindow
from DataSets_v01 import inverse_difference_full
from Additional_Functions import makePredictionsAndLabelsTest
from Additional_Functions import plotFunction
from matplotlib import pyplot as plt

# Loads the pre-trained model
checkpoint_filepath = r'./Checkpoints_cpu'
checkpoint_name = 'Batch_1028_Diff_True_LSTM_Model_3x128_Window_64_9k_samples_Random_Data_d_0.2_10_epochs'
# checkpoint_name = 'Batch_64_LSTM_Model_1x128_Window_64_1k_samples_Random_Data_d_0.2_5000_epochs'
CheckPoint = os.path.join(checkpoint_filepath, checkpoint_name)
LSTM_Model = tf.keras.models.load_model(CheckPoint)

if 'Diff_True' in checkpoint_name:
    Diff = True
else:
    Diff = False
# load the dataset
DF, raw_values = loadData('Traffic_Data_d_0.2_1k_Samples.csv', Diff)


Normed_Test_DF = zeroMean(DF)
Normed_raw_values = zeroMean(raw_values)

Window_Size = 64
LSTM_Window = generateWindow(Window_Size, train_df=None, val_df=None, test_df=Normed_Test_DF)

# Generate Predictions and Labels Array
Test_Labels, Test_Predictions = makePredictionsAndLabelsTest(LSTM_Model, LSTM_Window, Window_Size)

# plot and compare the models
ModelEval = LSTM_Model.evaluate(LSTM_Window.test)
if Diff:
    # during the differencing process the first sample is LOST
    Test_Labels = inverse_difference_full(Normed_raw_values, Test_Labels)
    Test_Predictions = inverse_difference_full(Normed_raw_values, Test_Predictions)

plotFunction(Test_Labels, Test_Predictions, Window_Size, ModelEval, checkpoint_name)
