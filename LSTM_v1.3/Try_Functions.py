from matplotlib import pyplot as plt
import numpy as np
from DC_Traffic_Generator.Chaotic_Map_Generator import genDataset
import csv
import os.path

d = 0.2
N_Samples = 10000
data = genDataset(d, N_Samples)
Time = np.arange(len(data))
plt.figure()
plt.plot(Time, data)
plt.xlabel('Time [s]')
plt.ylabel('Data Volume [Gb]')
plt.grid()
plt.title('Data Volume over Time ' + 'd = ' + str(d))
plt.show()

# # save data as .csv
dataSetRoot = r'..\Dataset'

header = ['Time [s]', 'Data [Gb]']
# yy = y.squeeze()
xx = np.array(data).squeeze()
arr = np.stack((Time, xx), axis=1)
with open(os.path.join(dataSetRoot, 'Traffic_Data_d_' + str(d) + '_' + str(int(N_Samples / 1000)) + 'k_Samples.csv'),
          'w',
          encoding='UTF8',
          newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(arr)
