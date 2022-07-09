import numpy as np
from traffic_Dataset import gen_data_b, Trafficdataset
import torch
import torch.nn.functional as F
import yaml
import time
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_res", default='outputs/lstm_out/args.yml', type=Path)
    args = parser.parse_args()
    return args


def create_model(args, cp_path):
    if args['net_type'] == 'cnn':
        from modules import Net as Net
    elif args['net_type'] == 'lstm':
        from modules import LSTEMO as Net
    elif args['net_type'] == 'cnn_lstm':
        from modules import LSTEMO2 as Net
    else:
        raise ValueError("wrong net type, received {}".format(args['net_type']))

    net = Net()

    net.load_state_dict(torch.load(cp_path)['model_dict'])

    net.eval()
    net.to(Device)
    return net


def evalAndPlot(ext_args, cpname, checkpoint):
    net = create_model(ext_args, checkpoint)
    net.eval()
    net.to(Device)

    criterion = torch.nn.L1Loss()
    x, y = gen_data_b(d=0.2, seq_len=10000, win_len=ext_args['win_len'], step=1, mode='test')
    ys = torch.zeros_like(y)
    n_frames = x.shape[0] - ext_args['win_len']
    loss = 0
    with torch.no_grad():
        for i in range(n_frames):
            # print(i, i + args['win_len'])
            xi = x[i:i + ext_args['win_len']]
            xi = xi.view(1, 1, -1).to(Device)
            pred = net(xi)
            loss += criterion(pred.view_as(y[i]), y[i]).item()
            ys[i] = pred.cpu()
    loss /= n_frames
    print(loss)
    from yw import yw
    y = x[ext_args['win_len']::ext_args['step']]
    X = x[:-ext_args['step']].unfold(dimension=0, size=ext_args['win_len'], step=ext_args['step'])
    y_yw, _ = yw(X, y, p=ext_args['win_len'])

    # return signals to be between 0 and 1
    # y_unNorm = y - y.min()
    # ys_unNorm = ys - ys.min()
    # from numpy.random import default_rng
    rng = np.random.default_rng()
    FirstSample = rng.integers(0, len(y)-1000)
    # FirstSample = 0
    # print(FirstSample)
    # Plot a random slice of 1000 samples:
    plt.figure()
    plt.suptitle(cpname, fontsize=16)
    plt.subplot(2, 1, 1)
    plt.plot(y.view(-1)[FirstSample:FirstSample+1000], '-', linewidth=0.5)
    plt.plot(ys.view(-1)[FirstSample:FirstSample+1000], 'r-', linewidth=0.5)
    # plt.plot(y_yw.view(-1), 'k-')
    plt.xlabel('Time Samples')
    plt.ylabel('Traffic [Gb]')
    plt.grid()
    plt.title('Predictions, from ' + str(FirstSample) + ', 1k samples' + ' MAE = ' + str(loss))
    plt.legend(['Testing Data', 'Predictions'])
    plt.subplot(2, 1, 2)
    ABS_Error = abs(y.view(-1) - ys.view(-1))
    plt.plot(ABS_Error[FirstSample:FirstSample+1000], 'b-', linewidth=0.5)
    # plt.plot(y.view(-1)-y_yw.view(-1), 'r.-')
    plt.xlabel('Time Samples')
    plt.ylabel('Prediction Error')
    plt.grid()
    plt.title('ABS Prediction Error, from ' + str(FirstSample) + ', 1k samples')
    plt.legend(['ABS Error'])
    plt.savefig(
        './outputs/result_plots/' + cpname + '.png',
        bbox_inches='tight')
    # plt.show()
    return


def run():
    args = parse_args()
    # Model_Name = 'LSTM'
    with args.f_res.open() as f:
        # args = yaml.load(f, Loader=yaml.FullLoader)
        args = yaml.load(f, Loader=yaml.Loader)  # for collab

    if args['save_path']:
        if args['best_epoch'] != args['last_epoch']:
            CheckPointName = args['net_type'] + '_Best_epoch_' + str(args['best_epoch'])
            CheckPoint = os.path.join(args['save_path'], 'chkpnt_' + CheckPointName + '.pt')
            evalAndPlot(args, CheckPointName, CheckPoint)  # evaluate "Best" model
        CheckPointName = args['net_type'] + '_Last_epoch_' + str(args['last_epoch'])
        CheckPoint = os.path.join(args['save_path'], 'chkpnt_' + CheckPointName + '.pt')
        evalAndPlot(args, CheckPointName, CheckPoint)  # evaluate "Last" model
    else:
        raise ValueError("no saved checkpoint")

    return


if __name__ == "__main__":
    run()
