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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_res", default='outputs/lstm_augs/args.yml', type=Path)
    args = parser.parse_args()
    return args


def create_model(args):
    if args['net_type'] == 'cnn':
        from modules import Net as Net
    elif args['net_type'] == 'lstm':
        from modules import LSTEMO as Net
    elif args['net_type'] == 'cnn_lstm':
        from modules import LSTEMO2 as Net
    else:
        raise ValueError("wrong net type, received {}".format(args['net_type']))

    net = Net()
    if args['save_path']:
        net.load_state_dict(torch.load(Path(args['save_path']) / 'chkpnt.pt')['model_dict'])
    net.eval()
    net.to(device)
    return net


def run():
    args = parse_args()
    with args.f_res.open() as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    net = create_model(args)
    net.eval()
    net.to(device)

    criterion = torch.nn.L1Loss()
    x, y = gen_data_b(d=0.5, seq_len=10000, win_len=args['win_len'], step=1, mode='test')
    ys = torch.zeros_like(y)
    n_frames = x.shape[0] - args['win_len']
    loss = 0
    with torch.no_grad():
        for i in range(n_frames):
            # print(i, i + args['win_len'])
            xi = x[i:i + args['win_len']]
            xi = xi.view(1, 1, -1).to(device)
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
    plt.show()
    return


if __name__ == "__main__":
    run()