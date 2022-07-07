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


def makePredictionsAndLabelsTestMultiStepOut(model, window, w_size):
    test_predictions = []
    for i in range(len(window.test_df) - w_size):
        test_input = window.test_df.values[i:w_size + i].reshape([1, -1, 1])
        # print(Test_Input)
        test_prediction = model.predict(test_input)
        test_predictions.append(test_prediction[0])
    test_labels = window.test_df.values[w_size:]
    return test_labels, test_predictions


def plotFunction(labels_array, predictions_array, window_length, model_eval, model_name):
    plt.figure()
    plt.suptitle(model_name, fontsize=16)
    Time = range(window_length, len(labels_array) + window_length)
    print(Time)
    plt.subplot(2, 1, 1)
    plt.plot(Time, labels_array)
    plt.plot(Time, predictions_array)
    plt.xlabel('Time [s]')
    plt.ylabel('Data Volume [Gb]')
    plt.grid()
    plt.title('Time Series Predictions, ' + ' Evaluation Loss = ' + str(model_eval[0]) +
              ' MAE = ' + str(model_eval[1]))
    plt.legend(['Validation dataset', 'Predictions'])
    plt.subplot(2, 1, 2)
    # ABS_Error = abs(labels_array.squeeze() - predictions_array)
    ABS_Error = abs(labels_array - predictions_array)
    plt.plot(Time, ABS_Error)
    plt.xlabel('Time Samples')
    plt.ylabel('Prediction Error')
    plt.grid()
    plt.title('ABS Prediction Error')
    plt.legend(['ABS Error'])
    # plt.subplot(3, 1, 3)
    # # Relative_Error = abs((labels_array.squeeze() - predictions_array) / labels_array.mean())
    # Relative_Error = abs((labels_array - predictions_array) / labels_array.mean())
    # plt.plot(Time, Relative_Error)
    # plt.xlabel('Time Samples')
    # plt.ylabel('Prediction Error')
    # plt.grid()
    # plt.title('Prediction Error over Time')
    # plt.legend(['Relative Error'])

    # plt.savefig(
    #     './Result_Plots/Evaluation/' + model_name + '.png', bbox_inches='tight')
    plt.show()


def plotPrediction(labels_array, predictions_array, window_length, model_eval, model_name, num_epochs):
    plt.figure()
    Time = range(window_length, len(labels_array) + window_length)
    print(Time)
    plt.plot(Time, labels_array)
    plt.plot(Time, predictions_array)
    plt.xlabel('Time [s]')
    plt.ylabel('Data Volume [Gb]')
    plt.grid()
    plt.title('Data Volume over Time, ' + model_name + ' Evaluation Loss = ' + str(model_eval[0]) +
              ' MAE = ' + str(model_eval[1]))
    plt.legend(['Validation dataset', 'Predictions'])
    plt.savefig(
        './Result_Plots/' + model_name + ' window ' + str(window_length) + ' ' + str(num_epochs) + '_epochs.png',
        bbox_inches='tight')
    plt.show()


def plotError(labels_array, predictions_array, window_length, model_eval, model_name, num_epochs):
    plt.figure()
    Time = range(window_length, len(labels_array) + window_length)
    print(Time)
    plt.plot(Time, labels_array - predictions_array)
    plt.xlabel('Time Samples')
    plt.ylabel('Prediction Error')
    plt.grid()
    plt.title('Absolute Prediction Error over Time, ' + model_name + ' Evaluation Loss = ' + str(model_eval[0]) +
              ' MAE = ' + str(model_eval[1]))
    plt.legend(['Error'])
    plt.savefig(
        './Result_Plots/Prediction_Error' + model_name + ' window ' + str(window_length) + ' ' + str(
            num_epochs) + '_epochs.png',
        bbox_inches='tight')
    plt.show()


def compareList(l1, l2):
    if l1 == l2:
        return "Equal"
    else:
        return "Non equal"
