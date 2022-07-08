import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from numpy.random import default_rng
import csv
import os.path


def gen_data2(d, seq_len=1000):
    # 1.5 < m1,m2 <= 2
    m1 = 2
    m2 = 2
    # 0 < d < 1, small d -> mostly mice, large d -> mostly elephants
    # d = 0.5
    # randomly generate initial network state x_0 from uniform distribution: x_0~U(0,1)
    x0 = np.random.uniform(0.01, 1, 1).astype(np.float32)
    x = np.zeros(seq_len + 1, dtype=np.float32)
    x[0] = x0
    for i in range(0, seq_len):
        if 0 < x[i] <= d:
            x[i + 1] = x[i] + (1 - d) * np.power((x[i] / d), m1)
        elif d < x[i] < 1:
            x[i + 1] = x[i] - d * np.power((1 - x[i]) / (1 - d), m2)
    x = x[1:]  # unscaled data [0,1)
    return x


def gen_data(d, length=1000):
    # 1.5 < m1,m2 <= 2
    m1 = 2
    m2 = 2
    # 0 < d < 1, small d -> mostly mice, large d -> mostly elephants
    # d = 0.5
    # randomly generate initial network state x_0 from uniform distribution: x_0~U(0,1)
    x0 = np.random.uniform(0.01, 1, 1).astype(np.float32)
    x = np.zeros(length + 1, dtype=np.float32)
    x[0] = x0
    for i in range(0, length):
        if 0 < x[i] <= d:
            x[i + 1] = x[i] + (1 - d) * np.power((x[i] / d), m1)
        elif d < x[i] < 1:
            x[i + 1] = x[i] - d * np.power((1 - x[i]) / (1 - d), m2)
    x = x[1:]  # unscaled data [0,1)
    return x


def genDataset(d, length):
    # 1.5 < m1,m2 <= 2
    m1 = 2
    m2 = 2
    # 0 < d < 1, small d -> mostly mice, large d -> mostly elephants
    # d = 0.5

    # randomly generate initial network state x_0 from uniform distribution: x_0~U(0,1)
    rng = default_rng()
    vals = rng.uniform(0.01, 1, 1)
    x_0 = vals.astype(float)
    x = [x_0]

    # length = 1000  # number of samples in dataset
    Time = np.arange(length)

    for i in range(0, length):
        if 0 < x[i] <= d:
            x_nxt = x[i] + (1 - d) * np.power((x[i] / d), m1)
        elif d < x[i] < 1:
            x_nxt = x[i] - d * np.power((1 - x[i]) / (1 - d), m2)

        # print(x_nxt)
        x.append(x_nxt)
    x = x[1:]  # unscaled data [0,1)
    return x


if __name__ == "__main__":
    x = genDataset(0.5, 1000)
    x = np.array([xx[0] for xx in x])
    plt.plot(x)
    plt.show()
