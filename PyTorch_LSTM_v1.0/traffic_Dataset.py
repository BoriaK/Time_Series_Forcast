from multiprocessing.spawn import import_main_path
import os
import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from augs import Augs
from PyAstronomy import pyaC



def gen_data(d, seq_len=1000):
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


def gen_data_b(d, seq_len, win_len, step, mode='train'):
    attempts = 0
    zc = 0
    while zc < 3 and attempts < 10:
        # 1.5 < m1,m2 <= 2
        m1 = 2
        m2 = 2
        # 0 < d < 1, small d -> mostly mice, large d -> mostly elephants
        # d = 0.5
        # randomly generate initial network state x_0 from uniform distribution: x_0~U(0,1)
        x0 = torch.rand(1) * 0.99 + 0.01
        x = torch.zeros(seq_len + 1, dtype=torch.float)
        x[0] = x0
        for i in range(0, seq_len):
            if x[i] <= d:
                x[i + 1] = x[i] + (1 - d) * np.power((x[i] / d), m1)
            else:
                x[i + 1] = x[i] - d * np.power((1 - x[i]) / (1 - d), m2)
        x = x[1:]  # unscaled data [0,1)
        y = x[win_len::step]
        if mode == 'train':
            x = x[:-step].unfold(dimension=0, size=win_len, step=step)

        # Check the number of zero crossings, to avoid "flat" data set generation
        x_norm = x - 0.5
        # Get coordinates and indices of zero crossings
        xc = pyaC.zerocross1d(np.arange(len(x_norm)), x_norm.data, getIndices=False)
        zc = len(xc)  # the number of zero crossings for the generated dataset
        attempts += 1
        if zc < 3 and attempts == 10:
            raise ValueError("problem with data generation")
            break
    return x, y


class Trafficdataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, win_len, step, d, augs=None):
        self.seq_len = seq_len
        self.augs = Augs(augs=augs, p=0.5) if augs is not None else None
        self.win_len = win_len
        self.step = step
        self.d = d
        self.n_sigs = (seq_len - win_len) // step
        self.x = []
        self.y = []

    def __getitem__(self, index):
        ii = index % self.n_sigs
        x = self.x[ii, :]
        y = self.y[ii]
        if self.augs is not None:
            x, y = self.augs(x, y)
        return x.unsqueeze(0), y

    def __len__(self):
        return self.n_sigs

    def gen_data(self):
        self.x, self.y = gen_data_b(d=self.d, seq_len=self.seq_len, win_len=self.win_len, step=self.step)


if __name__ == "__main__":
    x, _ = gen_data_b(0.2, 8192, 64, 16, mode='test')
    import matplotlib.pyplot as plt

    plt.plot(x.view(-1))
    plt.show()
