import os
import matplotlib.pyplot as plt
from DataSets_v01 import loadData
from DataSets_v01 import splitData
from DataSets_v01 import normAndScale
from DataSets_v01 import zeroMean
from DataSets_v01 import generateWindow
from Models_v01 import lstm_model
from Models_v01 import compileModel
from Models_v01 import fitModel

# load the dataset
DF = loadData('Traffic_Data_10k.csv')

Train_DF, Val_DF = splitData(DF)

# Normed_Train_DF = normAndScale(Train_DF)
# Normed_Val_DF = normAndScale(Val_DF)
Normed_Train_DF = zeroMean(Train_DF)
Normed_Val_DF = zeroMean(Val_DF)

Window_Size = 10
LSTM_Window = generateWindow(Window_Size, Normed_Train_DF, Normed_Val_DF, test_df=None)

Max_Epochs = 50

# Version1: use lstm model with stateful=False (can release constraint about model input shape)
LSTM_Model = lstm_model(Window_Size)
LSTM_Model = compileModel(LSTM_Model)
History = fitModel(LSTM_Model, LSTM_Window, epochs=Max_Epochs)

Train_Loss = History.history['loss']
Train_MAE = History.history['mean_absolute_error']
Validation_Loss = History.history['val_loss']
Validation_MAE = History.history['val_mean_absolute_error']

plt.subplot(2, 1, 1)
plt.plot(Train_Loss)
plt.plot(Validation_Loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(Train_MAE)
plt.plot(Validation_MAE)
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend(['Training MAE', 'Validation MAE'])
plt.grid()
plt.savefig('./Result_Plots/' + 'LSTM_Model ' + '10k samples ' + str(Max_Epochs) + '_epochs.png', bbox_inches='tight')
# plt.show()

# save checkpoint
checkpoint_filepath = r'./Checkpoints'
CheckPoint = os.path.join(checkpoint_filepath, 'cp_1x10_' + '10k samples ' + str(Max_Epochs) + '_epochs_LSTM_Model')
LSTM_Model.save(CheckPoint)
print('checkpoint ' + '1x10_' + '10k samples ' + str(Max_Epochs) + '_epochs_LSTM_Model' + ' is saved')
