from matplotlib import pyplot as plt
import numpy as np


def plotFunction(labels_array, predictions_array, window_length, model_eval, model_name, num_epochs):
    plt.figure()
    Time = range(window_length, len(labels_array)+window_length)
    print(Time)
    plt.plot(Time, labels_array)
    plt.plot(Time, predictions_array)
    # ############# for debug:
    # # plt.plot(raw_Test_values[:200])
    # # plt.plot(predictions[:200])
    # ##########################
    plt.xlabel('Time [s]')
    plt.ylabel('Data Volume [Gb]')
    plt.grid()
    plt.title('Data Volume over Time, ' + model_name + ' Evaluation Loss = ' + str(model_eval[0]) +
              'MAE = ' + str(model_eval[1]))
    plt.legend(['Validation dataset', 'Predictions'])
    plt.show()
    plt.savefig('./Result_Plots/' + model_name + ' ' + str(num_epochs) + '_epochs.png', bbox_inches='tight')

