from matplotlib import pyplot as plt
import numpy as np
from DataSets_v01 import loadData
from DataSets_v01 import splitData
from DataSets_v01 import normAndScale
from DataSets_v01 import zeroMean

# load the dataset
DF = loadData('Traffic_Data_1k.csv')

Train_DF, Val_DF = splitData(DF)
plt.figure(0)
plt.plot(Val_DF.values.squeeze())
plt.grid()

ZeroMean_Train_DF = zeroMean(Train_DF)
ZeroMean_Val_DF = zeroMean(Val_DF)
plt.figure(1)
plt.plot(ZeroMean_Val_DF.values.squeeze())
plt.grid()

Normed_Train_DF = normAndScale(Train_DF)
Normed_Val_DF = normAndScale(Val_DF)
plt.figure(3)
plt.plot(Normed_Val_DF.values.squeeze())
plt.grid()

plt.show()



