import os
from DataSets_v01 import loadData
from DataSets_v01 import splitData
from DataSets_v01 import normAndScale
from DataSets_v01 import zeroMean
from DataSets_v01 import generateWindow
from Models_v01 import deep_lstm_model_stateful
from Models_v01 import compileModel
from Models_v01 import fitModel
import matplotlib.pyplot as plt
import numpy as np
import shutil

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

# Version2: use lstm model with stateful=True, (need constraint about model input shape). make Epochs = 1,
# and put inside an external for loop over "Max_Epochs" with manually reset_state() between epochs.
# just like in TF.Keras_LSTM_v1.0
Deep_LSTM_Model_Stateful = deep_lstm_model_stateful(Window_Size, LSTM_Window.example[0].shape)
Deep_LSTM_Model_Stateful = compileModel(Deep_LSTM_Model_Stateful)


Train_Loss = list()
Train_MAE = list()
Validation_Loss = list()
Validation_MAE = list()

# save checkpoint
checkpoint_filepath = r'./Checkpoints'
for i in range(Max_Epochs):
    print('Epoch number ' + str(i))
    History = fitModel(Deep_LSTM_Model_Stateful, LSTM_Window, epochs=1)
    train_loss = History.history['loss'][0]
    train_mae = History.history['mean_absolute_error'][0]
    validation_loss = History.history['val_loss'][0]
    validation_mae = History.history['val_mean_absolute_error'][0]
    Train_Loss.append(train_loss)
    Train_MAE.append(train_mae)
    Validation_Loss.append(validation_loss)
    Validation_MAE.append(validation_mae)

    Deep_LSTM_Model_Stateful.reset_states()

    if i == 0:
        # Save the 1st checkpoint:
        CheckPoint = os.path.join(checkpoint_filepath, 'cp_5x10_' + '10k samples ' + str(i + 1) + '_epochs_Deep_LSTM_Model_Stateful')
        Deep_LSTM_Model_Stateful.save(CheckPoint)
        print('checkpoint ' + '5x10_' + '10k samples ' + str(i + 1) + '_epochs_Deep_LSTM_Model_Stateful' + ' is saved')
    elif validation_mae <= np.amin(Validation_MAE):
        # Save Best checkpoint:
        # remove last saved checkpoint:
        shutil.rmtree(CheckPoint, ignore_errors=True)
        print("Deleted '%s' directory successfully" % CheckPoint)
        CheckPoint = os.path.join(checkpoint_filepath, 'cp_5x10_' + '10k samples ' + str(i + 1) + '_epochs_Deep_LSTM_Model_Stateful')
        Deep_LSTM_Model_Stateful.save(CheckPoint)
        print('checkpoint ' + '5x10_' + '10k samples ' + str(i + 1) + '_epochs_Deep_LSTM_Model_Stateful' + ' is saved')

# Save Training Loss and MAE and Validation Loss and MAE
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
plt.savefig('./Result_Plots/' + 'Deep_LSTM_Model_Stateful ' + '10k samples ' + str(i + 1) + '_epochs.png', bbox_inches='tight')
# plt.show()



# for i in range(Max_Epochs):
#     print('Epoch number ' + str(i))
#     History = fitModel(Deep_LSTM_Model_Stateful, LSTM_Window, epochs=1)
#     Deep_LSTM_Model_Stateful.reset_states()
#
# # save checkpoint
# checkpoint_filepath = r'./Checkpoints'
# CheckPoint = os.path.join(checkpoint_filepath, 'cp_5x10_' + str(Max_Epochs) + '_epochs_Deep_LSTM_Model_Stateful')
# Deep_LSTM_Model_Stateful.save(CheckPoint)
# print('checkpoint ' + '5x10_' + str(Max_Epochs) + '_epochs_Deep_LSTM_Model_Stateful' + ' is saved')

