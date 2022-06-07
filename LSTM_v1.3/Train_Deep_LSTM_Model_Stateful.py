import os
from DataSets_v01 import loadData
from DataSets_v01 import splitData
from DataSets_v01 import normAndScale
from DataSets_v01 import generateWindow
from Models_v01 import deep_lstm_model_stateful
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

# Version2: use lstm model with stateful=True, (need constraint about model input shape). make Epochs = 1,
# and put inside an external for loop over "Max_Epochs" with manually reset_state() between epochs.
# just like in LSTM_v1.0
Deep_LSTM_Model_Stateful = deep_lstm_model_stateful(Window_Size, LSTM_Window.example[0].shape)
Deep_LSTM_Model_Stateful = compileModel(Deep_LSTM_Model_Stateful)

for i in range(Max_Epochs):
    print('Epoch number ' + str(i))
    History = fitModel(Deep_LSTM_Model_Stateful, LSTM_Window, epochs=1)
    Deep_LSTM_Model_Stateful.reset_states()

# save checkpoint
checkpoint_filepath = r'./Checkpoints'
CheckPoint = os.path.join(checkpoint_filepath, 'cp_5x10_' + str(Max_Epochs) + '_epochs_Deep_LSTM_Model_Stateful')
Deep_LSTM_Model_Stateful.save(CheckPoint)
print('checkpoint ' + '5x10_' + str(Max_Epochs) + '_epochs_Deep_LSTM_Model_Stateful' + ' is saved')

