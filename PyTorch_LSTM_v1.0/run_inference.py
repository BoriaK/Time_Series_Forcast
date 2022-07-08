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


def run():
    args = parse_args()
    # Model_Name = 'LSTM'
    with args.f_res.open() as f:
        # args = yaml.load(f, Loader=yaml.FullLoader)
        args = yaml.load(f, Loader=yaml.Loader)  # for collab

    if args['save_path']:
        if args['best_epoch'] != args['last_epoch']:
            CheckPoint = os.path.join(args['save_path'],
                                      'chkpnt_' + args['net_type'] + '_Best_epoch_' + str(args['best_epoch']) + '.pt')
        else:
            CheckPoint = os.path.join(args['save_path'],
                                      'chkpnt_' + args['net_type'] + '_Last_epoch_' + str(args['last_epoch']) + '.pt')
    else:
        raise ValueError("no saved checkpoint")

    net = create_model(args, CheckPoint)
    net.eval()
    net.to(Device)

    criterion = torch.nn.L1Loss()
    x, y = gen_data_b(d=0.2, seq_len=10000, win_len=args['win_len'], step=1, mode='test')
    ys = torch.zeros_like(y)
    n_frames = x.shape[0] - args['win_len']
    loss = 0
    with torch.no_grad():
        for i in range(n_frames):
            # print(i, i + args['win_len'])
            xi = x[i:i + args['win_len']]
            xi = xi.view(1, 1, -1).to(Device)
            pred = net(xi)
            loss += criterion(pred.view_as(y[i]), y[i]).item()
            ys[i] = pred.cpu()
    loss /= n_frames
    print(loss)
    from yw import yw
    y = x[args['win_len']::args['step']]
    X = x[:-args['step']].unfold(dimension=0, size=args['win_len'], step=args['step'])
    y_yw, _ = yw(X, y, p=args['win_len'])
    plt.figure()
    plt.plot(y.view(-1), '.-')
    plt.plot(ys.view(-1), 'r.-')
    plt.plot(y_yw.view(-1), 'k.-')
    plt.figure()
    plt.plot(y.view(-1) - ys.view(-1), 'b.-')
    # plt.plot(y.view(-1)-y_yw.view(-1), 'r.-')
    plt.savefig(
        './outputs/result_plots/' + args['net_type'] + '_best_epoch_' + str(args['best_epoch']) + '.png',
        bbox_inches='tight')
    # plt.show()
    return


if __name__ == "__main__":
    run()
