import os
import matplotlib.pyplot as plt
from DataSets_v01 import splitData
from DataSets_v01 import zeroMean
from DataSets_v01 import generateWindow
from Models_v01 import linear_model
from Models_v01 import compileModel
from Models_v01 import fitModel
import numpy as np
import shutil
from DC_Traffic_Generator.Chaotic_Map_Generator import genDataset
from pandas import DataFrame

# Version1: use lstm model with stateful=False (can release constraint about model input shape), use large batch size
# for training use the same loop as in "stateful" model,
# generate a NEW instance of the data set each iteration (Epoch)


Window_Size = 32
Units = 128
Linear_Model = linear_model(Units)
Linear_Model = compileModel(Linear_Model)

Max_Epochs = 10000

Train_Loss = list()
Train_MAE = list()
Validation_Loss = list()
Validation_MAE = list()

checkpoint_filepath = r'./Checkpoints'
for i in range(Max_Epochs):
    print('Epoch number ' + str(i))
    # load the dataset
    # DF = loadData('Traffic_Data_1k.csv')
    d = 0.2
    Data = genDataset(d, length=2000)  # generate a new time series from the chaotic map generator
    DF = DataFrame(Data)

    Train_DF, Val_DF = splitData(DF)  # 0.5 Training and 0.5 Validation

    # Normed_Train_DF = normAndScale(Train_DF)
    # Normed_Val_DF = normAndScale(Val_DF)
    Normed_Train_DF = zeroMean(Train_DF)
    Normed_Val_DF = zeroMean(Val_DF)

    Linear_Window = generateWindow(Window_Size, Normed_Train_DF, Normed_Val_DF, test_df=None)

    History = fitModel(Linear_Model, Linear_Window, epochs=1)
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
                                  'Batch_64_Linear_Model_1x' + str(Units) + '_Window_' + str(Window_Size) + '_' + str(
                                      int((len(Data)/2) / 1000)) + 'k_samples_Random_Data_d_' + str(d) + '_' + str(
                                      i + 1) + '_epochs')
        Linear_Model.save(CheckPoint)
        print(
            'checkpoint ' + '1x' + str(Units) + ' Window' + str(Window_Size) + ' ' + str(
                int((len(Data)/2) / 1000)) + 'k_samples_Random_Data ' + 'd=' + str(d) + ' ' + str(
                i + 1) + ' epochs Linear_Model' + ' is saved')
    elif validation_mae <= np.amin(Validation_MAE):
        # Save Best checkpoint:
        # remove last saved checkpoint:
        shutil.rmtree(CheckPoint, ignore_errors=True)
        print("Deleted '%s' directory successfully" % CheckPoint)
        CheckPoint = os.path.join(checkpoint_filepath,
                                  'Batch_64_Linear_Model_1x' + str(Units) + '_Window_' + str(Window_Size) + '_' + str(
                                      int((len(Data)/2) / 1000)) + 'k_samples_Random_Data_d_' + str(d) + '_' + str(
                                      i + 1) + '_epochs')
        Linear_Model.save(CheckPoint)
        print(
            'checkpoint ' + '1x' + str(Units) + ' Window' + str(Window_Size) + ' ' + str(
                int((len(Data)/2) / 1000)) + 'k_samples_Random_Data ' + 'd=' + str(d) + ' ' + str(
                i + 1) + ' epochs Linear_Model' + ' is saved')
    elif i == Max_Epochs - 1:
        # Save the last checkpoint:
        CheckPoint = os.path.join(checkpoint_filepath,
                                  'Batch_64_Linear_Model_1x' + str(Units) + '_Window_' + str(Window_Size) + '_' + str(
                                      int((len(Data)/2) / 1000)) + 'k_samples_Random_Data_d_' + str(d) + '_' + str(
                                      i + 1) + '_epochs')
        Linear_Model.save(CheckPoint)
        print(
            'checkpoint ' + '1x' + str(Units) + ' Window' + str(Window_Size) + ' ' + str(
                int((len(Data)/2) / 1000)) + 'k_samples_Random_Data ' + 'd=' + str(d) + ' ' + str(
                i + 1) + ' epochs Linear_Model' + ' is saved')

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
plt.savefig('./Result_Plots/Training_Process/' + 'Batch_64_Linear_Model_1x' + str(Units) + '_Window_' + str(
    Window_Size) + '_' + str(
    int((len(Data)/2) / 1000)) + 'k_samples_Random_Data_d_' + str(d) + '_' + str(
    i + 1) + '_epochs.png',
            bbox_inches='tight')
plt.show()
