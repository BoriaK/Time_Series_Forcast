import os
import tensorflow as tf
from DataSets_v01 import loadData
from DataSets_v01 import splitData
from DataSets_v01 import normAndScale
from DataSets_v01 import generateWindow
from Additional_Functions import makePredictionsAndLabelsTest
from Additional_Functions import plotFunction

# Loads the pre-trained model
checkpoint_filepath = r'./Checkpoints'
CheckPoint = os.path.join(checkpoint_filepath, 'cp_5x10_20_epochs_Deep_LSTM_Model_Stateful')
Deep_LSTM_Model = tf.keras.models.load_model(CheckPoint)

# load the dataset
DF = loadData('Traffic_Data_1k.csv')

# Train_DF, Val_DF = splitData(DF)

# Normed_Train_DF = normAndScale(Train_DF)
# Normed_Val_DF = normAndScale(Val_DF)
Normed_Test_DF = normAndScale(DF)

Window_Size = 10
LSTM_Window = generateWindow(Window_Size, train_df=None, val_df=None, test_df=Normed_Test_DF)

# Generate Predictions and Labels Array
# Eval_Labels, Eval_Predictions = makePredictionsAndLabels(Deep_LSTM_Model, LSTM_Window, Window_Size)
Test_Labels, Test_Predictions = makePredictionsAndLabelsTest(Deep_LSTM_Model, LSTM_Window, Window_Size)

# plot and compare the models
ModelEval = Deep_LSTM_Model.evaluate(LSTM_Window.test)
Max_Epochs = 20
plotFunction(Test_Labels, Test_Predictions, Window_Size, ModelEval, 'Deep_LSTM_Model_Stateful', Max_Epochs)
