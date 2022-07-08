import torch
from traffic_Dataset import gen_data_b
import matplotlib.pyplot as plt


def yw(X, y, p=1):
    win_len = p
    X = X.flip(dims=[-1, ])
    p = torch.linalg.solve(torch.matmul(X.T, X), torch.matmul(X.T, y))
    y_pred = torch.matmul(X, p)
    return y_pred, p

if __name__ == "__main__":
    win_len = 128
    X, y = gen_data_b(d=0.5, seq_len=8192, win_len=win_len, step=1, mode='train')
    y_pred, p = yw(X, y, p=win_len)
    err_rel = 20*torch.log10((y_pred-y).abs().mean()/y.abs().mean())
    err_abs = (y_pred-y).abs()
    plt.plot(y)
    plt.plot(y_pred, 'r-')
    plt.title("rel err:{:.2f}".format(err_rel))
    plt.show()