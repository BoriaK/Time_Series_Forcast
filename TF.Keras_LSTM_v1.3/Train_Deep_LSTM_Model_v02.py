import os
import matplotlib.pyplot as plt
from DataSets_v01 import splitData
from DataSets_v01 import zeroMean
from DataSets_v01 import generateWindow
from DataSets_v01 import difference
from DataSets_v01 import inverse_difference_full
from Models_v01 import deep1_lstm_model
from Models_v01 import deep2_lstm_model
from Models_v01 import deep3_lstm_model
from Models_v01 import deep5_lstm_model
from Models_v01 import compileModel
from Models_v01 import fitModel
import numpy as np
import shutil
from DC_Traffic_Generator.Chaotic_Map_Generator import genDataset
from pandas import DataFrame
from Additional_Functions import compareList

# Version1: use lstm model with stateful=False (can release constraint about model input shape), use large batch size
# for training use the same loop as in "stateful" model,
# generate a NEW instance of the data set each iteration (Epoch)


Window_Size = 64
Units = 128
depth = 3
Diff = True  # an indicator weather we're training over original data or a difference series.

if depth == 1:
    Deep_LSTM_Model = deep1_lstm_model(Units)
if depth == 2:
    Deep_LSTM_Model = deep2_lstm_model(Units)
if depth == 3:
    Deep_LSTM_Model = deep3_lstm_model(Units)
elif depth == 5:
    Deep_LSTM_Model = deep5_lstm_model(Units)

Deep_LSTM_Model = compileModel(Deep_LSTM_Model)

Max_Epochs = 10

Train_Loss = list()
Train_MAE = list()
Validation_Loss = list()
Validation_MAE = list()

checkpoint_filepath = r'./Checkpoints_cpu'
for i in range(Max_Epochs):
    print('Epoch number ' + str(i))
    # load the dataset
    # DF = loadData('Traffic_Data_1k.csv')
    d = 0.2
    Data_Length = 10000
    Data = genDataset(d, length=Data_Length)  # generate a new time series from the chaotic map generator
    if Diff:
        Diff_Data = difference(Data)  # make a difference series with interval of 1
        ############# debug:
        # during the differencing process the first sample is LOST
        # Data_hat = inverse_difference_full(Data, Diff_Data)
        # print(compareList(Data[1:], Data_hat))
        ###############################
        DF = DataFrame(Diff_Data)
    else:
        DF = DataFrame(Data)

    Train_Length = 0.9  # split the dataset to Train_Length*dataset for training and (1-Train_Length)*dataset for
    # validation
    Train_DF, Val_DF = splitData(DF, Train_Length)

    # Normed_Train_DF = normAndScale(Train_DF)
    # Normed_Val_DF = normAndScale(Val_DF)
    Normed_Train_DF = zeroMean(Train_DF)
    Normed_Val_DF = zeroMean(Val_DF)

    LSTM_Window = generateWindow(Window_Size, Normed_Train_DF, Normed_Val_DF, test_df=None)

    History = fitModel(Deep_LSTM_Model, LSTM_Window, epochs=1)
    train_loss = History.history['loss'][0]
    train_mae = History.history['mean_absolute_error'][0]
    validation_loss = History.history['val_loss'][0]
    validation_mae = History.history['val_mean_absolute_error'][0]
    Train_Loss.append(train_loss)
    Train_MAE.append(train_mae)
    Validation_Loss.append(validation_loss)
    Validation_MAE.append(validation_mae)

    # LSTM_Model_Stateful.reset_states()

    if i == 0:
        # Save the 1st checkpoint:
        CheckPoint = os.path.join(checkpoint_filepath,
                                  'Batch_1028_Diff_' + str(Diff) + '_LSTM_Model_' + str(depth) + 'x' + str(Units)
                                  + '_Window_' + str(Window_Size) + '_' + str(int((len(Data) * Train_Length) / 1000))
                                  + 'k_samples_Random_Data_d_' + str(d) + '_' + str(i + 1) + '_epochs')
        Deep_LSTM_Model.save(CheckPoint)
        print(
            'checkpoint ' + str(depth) + 'x' + str(Units) + ' Window' + str(Window_Size) + ' ' + str(
                int((len(Data) * Train_Length) / 1000)) + 'k_samples_Random_Data ' + 'd=' + str(d) + ' ' + str(
                i + 1) + ' epochs LSTM_Model' + ' is saved')
    elif validation_mae <= np.amin(Validation_MAE):
        # Save Best checkpoint:
        # remove last saved checkpoint:
        shutil.rmtree(CheckPoint, ignore_errors=True)
        print("Deleted '%s' directory successfully" % CheckPoint)
        CheckPoint = os.path.join(checkpoint_filepath,
                                  'Batch_1028_Diff_' + str(Diff) + '_LSTM_Model_' + str(depth) + 'x' + str(Units)
                                  + '_Window_' + str(Window_Size) + '_' + str(int((len(Data) * Train_Length) / 1000))
                                  + 'k_samples_Random_Data_d_' + str(d) + '_' + str(i + 1) + '_epochs')
        Deep_LSTM_Model.save(CheckPoint)
        print(
            'checkpoint ' + str(depth) + 'x' + str(Units) + ' Window' + str(Window_Size) + ' ' + str(
                int((len(Data) * Train_Length) / 1000)) + 'k_samples_Random_Data ' + 'd=' + str(d) + ' ' + str(
                i + 1) + ' epochs LSTM_Model' + ' is saved')
    elif (i + 1) % 5000 == 0:
        # Save every 5000th checkpoint:
        CheckPoint = os.path.join(checkpoint_filepath,
                                  'Batch_1028_Diff_' + str(Diff) + '_LSTM_Model_' + str(depth) + 'x' + str(Units)
                                  + '_Window_' + str(Window_Size) + '_' + str(int((len(Data) * Train_Length) / 1000))
                                  + 'k_samples_Random_Data_d_' + str(d) + '_' + str(i + 1) + '_epochs')
        Deep_LSTM_Model.save(CheckPoint)
        print(
            'checkpoint ' + str(depth) + 'x' + str(Units) + ' Window' + str(Window_Size) + ' ' + str(
                int((len(Data) * Train_Length) / 1000)) + 'k_samples_Random_Data ' + 'd=' + str(d) + ' ' + str(
                i + 1) + ' epochs LSTM_Model' + ' is saved')
    elif i == Max_Epochs - 1:
        # Save the last checkpoint:
        CheckPoint = os.path.join(checkpoint_filepath,
                                  'Batch_1028_Diff_' + str(Diff) + '_LSTM_Model_' + str(depth) + 'x' + str(Units)
                                  + '_Window_' + str(Window_Size) + '_' + str(int((len(Data) * Train_Length) / 1000))
                                  + 'k_samples_Random_Data_d_' + str(d) + '_' + str(i + 1) + '_epochs')
        Deep_LSTM_Model.save(CheckPoint)
        print(
            'checkpoint ' + str(depth) + 'x' + str(Units) + ' Window' + str(Window_Size) + ' ' + str(
                int((len(Data) * Train_Length) / 1000)) + 'k_samples_Random_Data ' + 'd=' + str(d) + ' ' + str(
                i + 1) + ' epochs LSTM_Model' + ' is saved')

plt.figure()
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
plt.savefig('./Result_Plots/Training_Process/' + 'Batch_1028_Diff_' + str(Diff) + '_LSTM_Model_' + str(depth) + 'x' +
            str(Units) + '_Window_' + str(Window_Size) + '_' + str(
            int((len(Data) * Train_Length) / 1000)) + 'k_samples_Random_Data_d_' + str(d) + '_' + str(i + 1) +
            '_epochs.png', bbox_inches='tight')
plt.show()
