import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
import time
import argparse
from pathlib import Path
import torchvision.transforms as T
from helper_funcs import accuracy
import logger
from traffic_Dataset import Trafficdataset
import os
import shutil

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='configs/cfg.yml', type=Path)
    args = parser.parse_args()
    return args


def set_wd(net, wd):
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if name.find('bias') != -1 or (name.find('weight') != -1 and len(param.size()) == 1):
            no_decay.append(param)
        else:
            decay.append(param)

    params = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.}]
    return params


def create_dataset(args, device):
    # from traffic_Dataset import Trafficdataset
    if device == 'cuda':
        train_set = Trafficdataset(seq_len=args['seq_len_cuda'], win_len=args['win_len'], step=args['step'],
                                   d=args['d'], augs=args['augs'])
        # train_set = Trafficdataset(seq_len=args['seq_len_cuda'], win_len=args['win_len'], step=args['step'],
        #                            d=args['d'], augs=None)
        test_set = Trafficdataset(seq_len=args['seq_len_cuda'], win_len=args['win_len'], step=args['step'], d=args['d'],
                                  augs=None)
        # for debug:
        print('seq_len = ' + str(args['seq_len_cuda']))
    else:
        train_set = Trafficdataset(seq_len=args['seq_len'], win_len=args['win_len'], step=args['step'], d=args['d'],
                                   augs=args['augs'])
        # train_set = Trafficdataset(seq_len=args['seq_len'], win_len=args['win_len'], step=args['step'], d=args['d'],
        #                            augs=None)
        test_set = Trafficdataset(seq_len=args['seq_len'], win_len=args['win_len'], step=args['step'], d=args['d'],
                                  augs=None)
        # for debug:
        print('seq_len = ' + str(args['seq_len']))
    return train_set, test_set


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
    if args['load_path']:
        net.load_state_dict(torch.load(Path(args['load_path']) / 'chkpnt.pt')['model_dict'])
    net.train()
    net.to(Device)
    return net


def train_one_epoch(train_loader, net, opt, cross_entropy, epoch):
    train_loader.dataset.gen_data()
    net.train()
    for iterno, (x, y) in enumerate(train_loader):
        net.zero_grad(set_to_none=True)

        x = x.to(Device)
        y = y.to(Device)

        y_est = net(x)

        loss = cross_entropy(y_est.view_as(y), y)
        # loss.register_hook(lambda grad: print(grad))
        loss.backward()
        opt.step()

        steps = (epoch - 1) * len(train_loader) + iterno
        # ema.update(net, steps)
    return loss


def run_eval(test_loader, net, cross_entropy):
    loss = 0
    test_loader.dataset.gen_data()
    net.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(Device)
            y = y.to(Device)
            y_est = net(x)
            loss += cross_entropy(y_est.view_as(y), y).item()
    loss /= len(test_loader)
    net.train()
    return loss


def train():
    print(Device)
    num_of_GPU = torch.cuda.device_count()
    if Device == 'cuda':
        print("number of GPUs is: " + str(num_of_GPU))

    CheckPoint = None

    args = parse_args()
    with args.cfg.open() as f:
        # args = yaml.load(f, Loader=yaml.FullLoader)
        args = yaml.load(f, Loader=yaml.Loader)  # for Collab
    root = Path(args['save_path'])
    load_root = Path(args['load_path']) if args['load_path'] else None
    print(load_root)
    root.mkdir(parents=True, exist_ok=True)

    train_set, test_set = create_dataset(args, Device)

    if Device == 'cuda':
        train_loader = DataLoader(train_set, batch_size=args['batch_size_cuda'], shuffle=True, drop_last=True,
                                  num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args['batch_size_cuda'], shuffle=False, drop_last=False,
                                 num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, drop_last=True, num_workers=8,
                                  pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, drop_last=False, num_workers=4,
                                 pin_memory=True)
    net = create_model(args)

    ####################################
    # optimizer #
    ####################################

    params = set_wd(net, args['wd'])
    opt = optim.Adam(params,
                     lr=args['max_lr'],
                     betas=args['betas'],
                     weight_decay=0.)

    lr_scheduler = optim.lr_scheduler.OneCycleLR(opt,
                                                 max_lr=args['max_lr'],
                                                 steps_per_epoch=len(train_loader),
                                                 pct_start=0.1,
                                                 epochs=args['n_epochs'])
    from ema import EMA
    # ema = EMA(net)
    ####################################
    # Loss #
    ####################################
    cross_entropy = torch.nn.L1Loss(reduction='mean')
    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    torch.backends.cudnn.benchmark = True
    steps = 0
    best_loss = 999
    net.train()

    if load_root and load_root.exists():
        checkpoint = torch.load(load_root / "chkpnt.pt")
        net.load_state_dict(checkpoint['model_dict'])
        opt.load_state_dict(checkpoint['opt_dict'])
        steps = checkpoint['resume_step'] if 'resume_step' in checkpoint.keys() else 0
        # best_loss = checkpoint['best_loss']
        print('checkpoints loaded')

    for epoch in range(1, args['n_epochs'] + 1):
        train_loader.dataset.gen_data()
        # ema.set_decay_per_step(num_steps_in_epoch=len(train_loader))
        for iterno, (x, y) in enumerate(train_loader):
            net.zero_grad(set_to_none=True)
            x = x.to(Device)
            y = y.to(Device)
            y_est = net(x)
            loss = cross_entropy(y_est.view_as(y), y)
            # loss.register_hook(lambda grad: print(grad))
            loss.backward()
            opt.step()
            lr_scheduler.step()
            # ema.update(net, step=steps)
            ######################
            # Update tensorboard #
            ######################
            writer.add_scalar("lr", opt.param_groups[0]['lr'], steps)
            writer.add_scalar("metric/train", loss.item(), steps)
            steps += 1

            if steps % args['log_interval'] == 0:
                print(
                    "Epoch {} | train: loss {:.8f} | best {:.8f}".format(
                        epoch,
                        loss,
                        best_loss
                    )
                )
            if steps % args['save_interval'] == 0:
                loss_test = run_eval(test_loader, net, cross_entropy)
                writer.add_scalar("metric/test", loss_test, steps)

                if loss_test < best_loss:
                    best_loss = loss_test
                    chkpnt = {
                        'model_dict': net.state_dict(),
                        'opt_dict': opt.state_dict(),
                        'step': steps,
                        'best_loss': best_loss
                    }
                    if CheckPoint is not None:
                        os.remove(CheckPoint)
                        print("Deleted '%s' successfully" % CheckPoint)
                    CheckPoint = os.path.join(root, 'chkpnt_' + args['net_type'] + '_Best_epoch_' + str(epoch) + '.pt')
                    torch.save(chkpnt, CheckPoint)
                    print('chkpnt_' + args['net_type'] + '_Best_epoch_' + str(epoch) + ' is saved')

                    with open(root / "args.yml", "w") as f:
                        args["best_epoch"] = epoch
                        yaml.dump(args, f)

                if steps % args['log_interval'] == 0:
                    print(
                        "Epoch {} | train: loss {:.8f} | test: loss {:.8f} | best {:.8f}".format(
                            epoch,
                            loss,
                            loss_test,
                            best_loss
                        )
                    )
                costs = []
    chkpnt = {
        'model_dict': net.state_dict(),
        'opt_dict': opt.state_dict(),
        'step': steps,
        'best_loss': best_loss
    }

    CheckPoint = os.path.join(root, 'chkpnt_' + args['net_type'] + '_Last_epoch_' + str(epoch) + '.pt')
    torch.save(chkpnt, CheckPoint)
    print('chkpnt_' + args['net_type'] + '_Last_epoch_' + str(epoch) + ' is saved')

    with open(root / "args.yml", "w") as f:
        args["last_epoch"] = epoch
        yaml.dump(args, f)


if __name__ == "__main__":
    train()
