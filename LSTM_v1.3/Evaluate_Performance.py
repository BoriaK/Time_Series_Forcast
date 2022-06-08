import os
import tensorflow as tf
from DataSets_v01 import loadData
from DataSets_v01 import splitData
from DataSets_v01 import normAndScale
from DataSets_v01 import zeroMean
from DataSets_v01 import generateWindow
from Additional_Functions import makePredictionsAndLabels
from Additional_Functions import plotFunction
from matplotlib import pyplot as plt

# Loads the pre-trained model
checkpoint_filepath = r'./Checkpoints'
CheckPoint = os.path.join(checkpoint_filepath, 'cp_5x10_10k samples 50_epochs_Deep_LSTM_Model')
LSTM_Model = tf.keras.models.load_model(CheckPoint)

# load the dataset
DF = loadData('Traffic_Data_1k.csv')

Train_DF, Val_DF = splitData(DF)

# Normed_Train_DF = normAndScale(Train_DF)
# Normed_Val_DF = normAndScale(Val_DF)
Normed_Train_DF = zeroMean(Train_DF)
Normed_Val_DF = zeroMean(Val_DF)

Window_Size = 10
LSTM_Window = generateWindow(Window_Size, Normed_Train_DF, Normed_Val_DF, test_df=None)

# Generate Predictions and Labels Array
Eval_Labels, Eval_Predictions = makePredictionsAndLabels(LSTM_Model, LSTM_Window, Window_Size)

# plot and compare the models
ModelEval = LSTM_Model.evaluate(LSTM_Window.val)
Max_Epochs = 50
plotFunction(Eval_Labels, Eval_Predictions, Window_Size, ModelEval, 'Deep_LSTM_Model', Max_Epochs)
