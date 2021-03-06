import os.path
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot


# The persistence forecast is where the observation from the prior time step (t-1) is used to predict the observation
# at the current time step (t) load dataset


def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


dataSetRoot = r'C:\Users\bkupcha\OneDrive - Intel Corporation\Documents\PythonProjects\Time_Series_Forcast_LSTM\Dataset'
series = read_csv(os.path.join(dataSetRoot, 'shampoo-sales.csv'), header=0, parse_dates=[0], index_col=0, squeeze=True,
                  date_parser=parser)
# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # make prediction
    predictions.append(history[-1])
    # observation
    history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()
