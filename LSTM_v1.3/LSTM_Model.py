import os
from DataSets_v01 import loadData
from DataSets_v01 import splitData
from DataSets_v01 import normAndScale
from DataSets_v01 import zeroMean
from DataSets_v01 import generateWindow
from Models_v01 import lstm_model
from Models_v01 import compileModel
from Models_v01 import fitModel
from Additional_Functions import makePredictionsAndLabels
from Additional_Functions import plotFunction

# load the dataset

DF = loadData('Traffic_Data_1k.csv')
Train_DF, Val_DF = splitData(DF)

# Normed_Train_DF = normAndScale(Train_DF)
# Normed_Val_DF = normAndScale(Val_DF)
Normed_Train_DF = zeroMean(Train_DF)
Normed_Val_DF = zeroMean(Val_DF)

Window_Size = 10
LSTM_Window = generateWindow(Window_Size, Normed_Train_DF, Normed_Val_DF, test_df=None)

Max_Epochs = 20

LSTM_Model = lstm_model(Window_Size, LSTM_Window.example[0].shape)
# LSTM_Model = lstm_model(Window_Size)

LSTM_Model = compileModel(LSTM_Model)
History = fitModel(LSTM_Model, LSTM_Window, epochs=Max_Epochs)

# save checkpoint
checkpoint_filepath = r'./Checkpoints'
CheckPoint = os.path.join(checkpoint_filepath, 'cp_1x10_' + str(Max_Epochs) + '_epochs_LSTM_Model')
LSTM_Model.save(CheckPoint)
print('checkpoint ' + '5x10_' + str(Max_Epochs) + '_epochs_LSTM_Model' + ' is saved')

# Generate Predictions and Labels Array
Eval_Labels, Eval_Predictions = makePredictionsAndLabels(LSTM_Model, LSTM_Window, Window_Size)

# plot and Evaluate the models
ModelEval = LSTM_Model.evaluate(LSTM_Window.val)
plotFunction(Eval_Labels, Eval_Predictions, Window_Size, ModelEval, 'LSTM_Model_Stateful', Max_Epochs)
