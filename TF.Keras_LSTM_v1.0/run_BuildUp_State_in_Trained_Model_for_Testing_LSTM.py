from DataSets import timeseries_to_supervised
from DataSets import scaleUni
import tensorflow as tf


# this version loads the already-trained model, and the entire training dataset, and passes an Evaluation iteration over
# the entire set in order to build up the state for future predictions. returns the model with state

def buildUpState(checkpoint_name, data_series):
    # transform data to be stationary
    raw_Train_values = data_series.values

    # transform data to be supervised learning
    supervised_Train = timeseries_to_supervised(raw_Train_values, 1)
    supervised_Train_values = supervised_Train.values
    # print(supervised_Train_values)

    train = supervised_Train_values

    # transform the scale of the data
    _, train_scaled = scaleUni(train)

    # Loads the pre-trained model
    lstmModel = tf.keras.models.load_model(checkpoint_name)

    # # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstmModel.predict(train_reshaped, batch_size=1)
    return lstmModel
