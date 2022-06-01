import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
import csv
import os.path

# 1.5 < m1,m2 <= 2
m1 = 2
m2 = 2
# 0 < d < 1, small d -> mostly mice, large d -> mostly elephants
d = 0.5

# randomly generate initial network state x_0 from uniform distribution: x_0~U(0,1)
rng = default_rng()
vals = rng.uniform(0.01, 1, 1)
x_0 = vals.astype(float)
x = [x_0]

N = 100000  # number of samples in dataset
Time = np.arange(N)

for i in range(0, N):
    if 0 < x[i] <= d:
        x_nxt = x[i] + (1 - d) * np.power((x[i] / d), m1)
    elif d < x[i] < 1:
        x_nxt = x[i] - d * np.power((1 - x[i]) / (1 - d), m2)

    # print(x_nxt)
    x.append(x_nxt)

x = x[1:]
y = np.array(x) * 10  # upscale the data to be in range 0:10
# y = [i * 10 for i in x]

# print out the last 1000 samples
plt.figure()
plt.plot(Time[-1000:], y[-1000:])
plt.xlabel('Time [s]')
plt.ylabel('Data Volume [Gb]')
plt.grid()
plt.title('Data Volume over Time')
plt.show()

# save data as .csv

dataSetRoot = r'.\Dataset'

header = ['Time [s]', 'Data [Gb]']
yy = y.squeeze()
arr = np.stack((Time, yy), axis=1)
with open(os.path.join(dataSetRoot, 'Traffic_Data_'+str(int(N / 1000))+'k.csv'), 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(arr)
