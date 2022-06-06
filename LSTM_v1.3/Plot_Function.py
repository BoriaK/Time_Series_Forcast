from matplotlib import pyplot as plt
import numpy as np

# Eval_Input = np.random.random(20)
# print(Eval_Input)
# warm_up = 10
# Eval_Labels = Eval_Input[warm_up:]
# print(Eval_Labels)
# Eval_Predictions = np.random.random(len(Eval_Labels))
# # Eval_Predictions = np.random.random()
# print(Eval_Predictions)

# for i in range(warm_up):


def plotFunction(labels_array, predictions_array, window_length, model_eval):
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
    plt.title('Data Volume over Time, ' + 'Evaluation Loss = ' + str(model_eval[0]) + 'MAE = ' + str(model_eval[1]))
    plt.legend(['Validation dataset', 'Predictions'])
    plt.show()


# plotFunction(Eval_Labels, Eval_Predictions, warm_up)
