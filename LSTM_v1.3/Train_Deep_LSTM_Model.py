import os
from DataSets_v01 import loadData
from DataSets_v01 import splitData
from DataSets_v01 import normAndScale
from DataSets_v01 import generateWindow
from Models_v01 import deep_lstm_model
from Models_v01 import compileModel
from Models_v01 import fitModel


# load the dataset
DF = loadData('Traffic_Data_1k.csv')

Train_DF, Val_DF = splitData(DF)

Normed_Train_DF = normAndScale(Train_DF)
Normed_Val_DF = normAndScale(Val_DF)

Window_Size = 10
LSTM_Window = generateWindow(Window_Size, Normed_Train_DF, Normed_Val_DF, test_df=None)

Max_Epochs = 20

# Version1: use lstm model with stateful=False (can release constraint about model input shape)
Deep_LSTM_Model = deep_lstm_model(Window_Size)
Deep_LSTM_Model = compileModel(Deep_LSTM_Model)
History = fitModel(Deep_LSTM_Model, LSTM_Window, epochs=Max_Epochs)

# save checkpoint
checkpoint_filepath = r'./Checkpoints'
CheckPoint = os.path.join(checkpoint_filepath, 'cp_5x10_' + str(Max_Epochs) + '_epochs_Deep_LSTM_Model')
Deep_LSTM_Model.save(CheckPoint)
print('checkpoint ' + '5x10_' + str(Max_Epochs) + '_epochs_Deep_LSTM_Model' + ' is saved')

