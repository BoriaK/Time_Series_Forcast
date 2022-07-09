import os
import matplotlib.pyplot as plt
from DataSets_v01 import splitData
from DataSets_v01 import zeroMean
from DataSets_v01 import generateMultistepWindow
from Models_v01 import multi_linear_model
from Models_v01 import compileModel
from Models_v01 import fitModel
import numpy as np
import shutil
from DC_Traffic_Generator.Chaotic_Map_Generator import genDataset
from pandas import DataFrame

# in this method will need to generate a window with 8 time steps forcast (ahead)
OUT_STEPS = 8
Window_Size = 16
MultiStep_Linear_Model = multi_linear_model(out_steps=OUT_STEPS, num_features=1)
MultiStep_Linear_Model = compileModel(MultiStep_Linear_Model)

Max_Epochs = 5000

Train_Loss = list()
Train_MAE = list()
Validation_Loss = list()
Validation_MAE = list()

checkpoint_filepath = r'./Checkpoints'
for i in range(Max_Epochs):
    print('Epoch number ' + str(i))
    # load the dataset
    # DF = loadData('Traffic_Data_1k.csv')
    Data = genDataset(d=0.5, length=1000)  # generate a new time series from the chaotic map generator
    DF = DataFrame(Data)

    Train_DF, Val_DF = splitData(DF)

    # Normed_Train_DF = normAndScale(Train_DF)
    # Normed_Val_DF = normAndScale(Val_DF)
    Normed_Train_DF = zeroMean(Train_DF)
    Normed_Val_DF = zeroMean(Val_DF)

    Multi_Window = generateMultistepWindow(Window_Size, OUT_STEPS, Normed_Train_DF, Normed_Val_DF, test_df=None)
    # print(Multi_Window)

    History = fitModel(MultiStep_Linear_Model, Multi_Window, epochs=1)
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
                                  'Batch_32_Out_' + str(OUT_STEPS) + '_' + str(int(len(Data) / 1000)) + 'k_samples_Random_Data_w_' + str(
                                      Window_Size) + '_' + str(
                                      i + 1) + '_epochs_MultiStep_Linear_Model')
        MultiStep_Linear_Model.save(CheckPoint)
        print(
            'checkpoint ' + 'Out_' + str(OUT_STEPS) + ' ' + str(int(len(Data) / 1000)) + 'k_samples_Random_Data_w_' + str(
                Window_Size) + ' ' + str(
                i + 1) + ' epochs MultiStep_Linear_Model' + ' is saved')
    elif validation_mae <= np.amin(Validation_MAE):
        # Save Best checkpoint:
        # remove last saved checkpoint:
        shutil.rmtree(CheckPoint, ignore_errors=True)
        print("Deleted '%s' directory successfully" % CheckPoint)
        CheckPoint = os.path.join(checkpoint_filepath,
                                  'Batch_32_Out_' + str(OUT_STEPS) + '_' + str(int(len(Data) / 1000)) + 'k_samples_Random_Data_w_' + str(
                                      Window_Size) + '_' + str(
                                      i + 1) + '_epochs_MultiStep_Linear_Model')
        MultiStep_Linear_Model.save(CheckPoint)
        print(
            'checkpoint ' + 'Out_' + str(OUT_STEPS) + ' ' + str(
                int(len(Data) / 1000)) + 'k_samples_Random_Data_w_' + str(
                Window_Size) + ' ' + str(
                i + 1) + ' epochs MultiStep_Linear_Model' + ' is saved')
    elif i == Max_Epochs - 1:
        # Save the last checkpoint:
        CheckPoint = os.path.join(checkpoint_filepath,
                                  'Batch_32_Out_' + str(OUT_STEPS) + '_' + str(int(len(Data) / 1000)) + 'k_samples_Random_Data_w_' + str(
                                      Window_Size) + '_' + str(
                                      i + 1) + '_epochs_MultiStep_Linear_Model')
        MultiStep_Linear_Model.save(CheckPoint)
        print(
            'checkpoint ' + 'Out_' + str(OUT_STEPS) + ' ' + str(
                int(len(Data) / 1000)) + 'k_samples_Random_Data_w_' + str(
                Window_Size) + ' ' + str(
                i + 1) + ' epochs MultiStep_Linear_Model' + ' is saved')

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
plt.savefig(
    './Result_Plots/Training_Process/' + 'Batch_32_Out_' + str(OUT_STEPS) + '_MultiStep_Linear_Model_1k_samples_Random_Data_w_' + str(
        Window_Size) + '_' + str(
        i + 1) + '_epochs.png',
    bbox_inches='tight')
plt.show()
