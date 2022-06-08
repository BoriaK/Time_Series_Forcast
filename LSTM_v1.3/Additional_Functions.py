from matplotlib import pyplot as plt


def makePredictionsAndLabels(model, window, w_size):
    eval_predictions = []
    for i in range(len(window.val_df) - w_size):
        eval_input = window.val_df.values[i:w_size + i].reshape([1, -1, 1])
        # print(Eval_Input)
        eval_prediction = model.predict(eval_input)
        eval_predictions.append(eval_prediction[0][0])
    eval_labels = window.val_df.values[w_size:]
    return eval_labels, eval_predictions


def makePredictionsAndLabelsTest(model, window, w_size):
    test_predictions = []
    for i in range(len(window.test_df) - w_size):
        test_input = window.test_df.values[i:w_size + i].reshape([1, -1, 1])
        # print(Test_Input)
        test_prediction = model.predict(test_input)
        test_predictions.append(test_prediction[0][0])
    test_labels = window.test_df.values[w_size:]
    return test_labels, test_predictions


def plotFunction(labels_array, predictions_array, window_length, model_eval, model_name, num_epochs):
    plt.figure()
    Time = range(window_length, len(labels_array) + window_length)
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
              ' MAE = ' + str(model_eval[1]))
    plt.legend(['Validation dataset', 'Predictions'])
    plt.savefig('./Result_Plots/' + model_name + ' ' + str(num_epochs) + '_epochs.png', bbox_inches='tight')
    plt.show()
