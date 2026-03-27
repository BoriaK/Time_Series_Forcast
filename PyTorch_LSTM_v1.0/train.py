import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
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

# v1 trains the networks over a specific d (from the relevant cfg)
# v2 trains the networks over a random d each epoch (from the relevant cfg_v2)

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """
    Early stopping that tracks test loss history and detects real learning vs noise.
    Considers overall trend, not just single best values.
    Tracks EPOCHS without improvement, not evaluation iterations.
    """
    def __init__(self, patience_epochs=10, min_delta=0.0, window=5):
        self.patience_epochs = patience_epochs
        self.min_delta = min_delta
        self.window = window  # Window size for moving average
        self.epochs_without_improvement = 0
        self.best_loss = None
        self.early_stop = False
        self.loss_history = []
        self.last_improvement_epoch = 0

    def moving_average(self, data, window):
        """Calculate moving average"""
        if len(data) < window:
            return sum(data) / len(data) if data else 0
        return sum(data[-window:]) / window

    def is_real_improvement(self):
        """Check if improvement is real or just noise"""
        if len(self.loss_history) < self.window:
            return True

        # Compare moving average of recent losses with earlier losses
        recent_avg = self.moving_average(self.loss_history[-self.window:], self.window)
        if len(self.loss_history) >= 2 * self.window:
            earlier_avg = self.moving_average(self.loss_history[-2*self.window:-self.window], self.window)
            improvement = earlier_avg - recent_avg
            return improvement > self.min_delta
        return True

    def __call__(self, test_loss, current_epoch):
        self.loss_history.append(test_loss)

        if self.best_loss is None:
            self.best_loss = test_loss
            self.last_improvement_epoch = current_epoch
            print(f"📊 Initial test loss: {test_loss:.8f}")
            return

        # Check overall trend using moving average
        if len(self.loss_history) >= self.window:
            current_avg = self.moving_average(self.loss_history[-self.window:], self.window)

            # Print loss progression
            print(f"📊 Test loss: {test_loss:.8f} | MA({self.window}): {current_avg:.8f} | Best: {self.best_loss:.8f}")

            # Check if moving average shows improvement
            if current_avg < self.best_loss - self.min_delta:
                if self.is_real_improvement():
                    improvement = self.best_loss - current_avg
                    print(f"✓ Real improvement detected: {improvement:.8f}")
                    self.best_loss = current_avg
                    self.last_improvement_epoch = current_epoch
                    self.epochs_without_improvement = 0
                else:
                    print(f"⚠️  Spike detected - not consistent improvement")
            else:
                # Calculate epochs without improvement
                self.epochs_without_improvement = current_epoch - self.last_improvement_epoch
                print(f"⚠️  No improvement: {self.epochs_without_improvement} epochs without improvement (patience: {self.patience_epochs})")
        else:
            # Not enough history yet
            print(f"📊 Test loss: {test_loss:.8f} | Collecting history... ({len(self.loss_history)}/{self.window})")
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.last_improvement_epoch = current_epoch

        # Trigger early stopping based on EPOCHS
        if self.epochs_without_improvement >= self.patience_epochs:
            self.early_stop = True
            print(f"\n🛑 Early stopping triggered!")
            print(f"   No real improvement for {self.epochs_without_improvement} EPOCHS")
            print(f"   Last improvement at epoch {self.last_improvement_epoch}")
            print(f"   Loss history (last 10): {[f'{l:.6f}' for l in self.loss_history[-10:]]}")
            print(f"   Model is NOT learning effectively\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='configs/cfg_convlstm.yml', type=Path)
    parser.add_argument("--optimize", action='store_true',
                        help='Enable parameter optimization (adjust on failure, update on success)')
    args = parser.parse_args()
    print(f"Arguments: {args}")
    return args


def adjust_parameters_on_failure(args, root):
    """
    Automatically adjust hyperparameters when training stops prematurely.
    Saves the new configuration for next run.
    """
    print("\n" + "="*70)
    print("ADJUSTING PARAMETERS FOR NEXT RUN")
    print("="*70)

    old_lr = args.get('max_lr', 0.001)
    old_wd = args.get('wd', 0.0001)
    old_hidden = args.get('hidden_channels', 64)
    old_kernel = args.get('kernel_size', 3)

    # Strategy: If not learning, try to help the model learn better
    # 1. Increase learning rate (conservative → more aggressive)
    new_lr = min(old_lr * 1.5, 0.002)  # Increase by 50%, cap at 0.002

    # 2. Reduce weight decay (less regularization)
    new_wd = max(old_wd * 0.5, 0.00005)  # Reduce by 50%, floor at 0.00005

    # 3. Increase model capacity slightly
    new_hidden = min(int(old_hidden * 1.25), 96)  # Increase by 25%, cap at 96

    # 4. Try different kernel size (alternate between values)
    if old_kernel == 5:
        new_kernel = 3
    elif old_kernel == 3:
        new_kernel = 7
    else:
        new_kernel = 5

    # Update args
    args['max_lr'] = new_lr
    args['wd'] = new_wd
    args['hidden_channels'] = new_hidden
    args['kernel_size'] = new_kernel

    print(f"Parameter Adjustments:")
    print(f"  Learning Rate:    {old_lr:.6f} → {new_lr:.6f}")
    print(f"  Weight Decay:     {old_wd:.6f} → {new_wd:.6f}")
    print(f"  Hidden Channels:  {old_hidden} → {new_hidden}")
    print(f"  Kernel Size:      {old_kernel} → {new_kernel}")

    # Save adjusted config in outputs/convlstm/ directory
    adjusted_cfg_path = root / f"cfg_{args['net_type']}_adjusted.yml"
    with open(adjusted_cfg_path, "w") as f:
        yaml.dump(args, f)

    print(f"\n✓ Adjusted configuration saved to: {adjusted_cfg_path}")
    print(f"  Run next training with: python train.py --cfg {adjusted_cfg_path} --optimize")
    print("="*70 + "\n")

    return args


def update_config_on_success(args, original_cfg_path, root):
    """
    Update the original config file when training completes successfully.
    If an adjusted config exists, use those parameters instead.
    """
    print("\n" + "="*70)
    print("UPDATING ORIGINAL CONFIG WITH SUCCESSFUL PARAMETERS")
    print("="*70)

    # Check if adjusted config exists from previous run
    adjusted_cfg_path = root / f"cfg_{args['net_type']}_adjusted.yml"

    if adjusted_cfg_path.exists():
        print(f"\n✓ Found adjusted config from previous run: {adjusted_cfg_path}")
        print(f"  Loading optimized parameters...")

        # Load the adjusted config
        with open(adjusted_cfg_path, 'r') as f:
            adjusted_args = yaml.load(f, Loader=yaml.Loader)

        # Use adjusted parameters
        print(f"\nApplying adjusted parameters to original config:")
        print(f"  Hidden Channels: {adjusted_args.get('hidden_channels')}")
        print(f"  Kernel Size: {adjusted_args.get('kernel_size')}")
        print(f"  Learning Rate: {adjusted_args.get('max_lr')}")
        print(f"  Weight Decay: {adjusted_args.get('wd')}")

        # Write adjusted config to original
        with open(original_cfg_path, "w") as f:
            yaml.dump(adjusted_args, f)

        print(f"\n✓ Original config updated with adjusted parameters from: {adjusted_cfg_path}")

        # Remove the adjusted config since it's been applied
        adjusted_cfg_path.unlink()
        print(f"✓ Removed adjusted config (parameters now in original)")
    else:
        # No adjusted config exists, just update with current args
        print(f"\nNo adjusted config found. Updating with current parameters:")
        print(f"  Hidden Channels: {args.get('hidden_channels')}")
        print(f"  Kernel Size: {args.get('kernel_size')}")
        print(f"  Learning Rate: {args.get('max_lr')}")
        print(f"  Weight Decay: {args.get('wd')}")

        # Write updated config
        with open(original_cfg_path, "w") as f:
            yaml.dump(args, f)

        print(f"\n✓ Original configuration updated")

    print(f"  Next training will use these optimized settings")
    print("="*70 + "\n")


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
    if device.type == 'cuda':
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
        net = Net()
    elif args['net_type'] == 'lstm':
        from modules import LSTMO as Net
        net = Net()
    elif args['net_type'] == 'cnn_lstm':
        from modules import CNNLSTM as Net
        net = Net()
    elif args['net_type'] == 'convlstm':
        from modules import ConvLSTM as Net
        net = Net(
            input_channels=1,
            hidden_channels=args.get('hidden_channels', 64),
            num_layers=args.get('num_layers', 2),
            kernel_size=args.get('kernel_size', 3),
            device=Device
        )
    elif args['net_type'] == 'transformers':
        from modules import TransformerModelV04 as Net
        net = Net()
    elif args['net_type'] == 'narnn':
        from modules import NARNN_v2 as Net
        net = Net(
            win_len=args['win_len'],
            hidden_size=args['hidden_size'],
            num_layers=args['num_layers'],
            # device=Device
        )
    else:
        raise ValueError("wrong net type, received {}".format(args['net_type']))

    # net = Net()

    # last_epoch = 0  # initialize to 0
    # if args['load_path']:
    #     CheckpointName = 'chkpnt_' + args['net_type'] + '_Last_epoch_1000.pt'
    #     net.load_state_dict(torch.load(Path(args['load_path']) / CheckpointName)['model_dict'])
    #     last_epoch = 1000
    #     print("Loaded last checkpoint from previous training. epoch = " + str(last_epoch))

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
    # # for debug:
    # Device = torch.device("cuda")
    print(Device)
    num_of_GPU = torch.cuda.device_count()
    if Device.type == 'cuda':
        print("number of GPUs is: " + str(num_of_GPU))

    CheckPoint = None

    args_parser = parse_args()
    original_cfg_path = args_parser.cfg  # Store the original config path
    enable_optimization = args_parser.optimize  # Store the optimization flag
    with args_parser.cfg.open() as f:
        # args = yaml.load(f, Loader=yaml.FullLoader)
        args = yaml.load(f, Loader=yaml.Loader)  # for Collab
    root = Path(args['save_path'] + '/' + args['net_type'])
    load_root = Path(args['load_path'] + '/' + args['net_type']) if args['load_path'] else None
    # print(load_root)
    root.mkdir(parents=True, exist_ok=True)

    train_set, test_set = create_dataset(args, Device)

    if Device.type == 'cuda':
        train_loader = DataLoader(train_set, batch_size=args['batch_size_cuda'], shuffle=True, drop_last=True,
                                  num_workers=4*num_of_GPU, pin_memory=True, prefetch_factor=4)
        test_loader = DataLoader(test_set, batch_size=args['batch_size_cuda'], shuffle=False, drop_last=False,
                                 num_workers=4*num_of_GPU, pin_memory=True, prefetch_factor=4)
    else:
        train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, drop_last=True, num_workers=8,
                                  pin_memory=True, prefetch_factor=4)
        test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, drop_last=False, num_workers=4,
                                 pin_memory=True, prefetch_factor=4)
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
    # with open(root / "args.yml", "w") as f:
    #     yaml.dump(args, f)
    # writer = SummaryWriter(str(root))

    torch.backends.cudnn.benchmark = True
    steps = 0
    best_loss = 999
    best_epoch = 0
    last_epoch = 0  # initialize to 0
    net.train()

    # Initialize early stopping with patience=10 epochs
    early_stopping = EarlyStopping(patience_epochs=10, min_delta=0.0)

    if args['load_path']:
        last_epoch = 100
        CheckpointName = 'chkpnt_' + args['net_type'] + '_Last_epoch_' + str(last_epoch) + '.pt'
        net.load_state_dict(torch.load(load_root / CheckpointName)['model_dict'])
        print("Loaded last checkpoint from previous training. epoch = " + str(last_epoch))

    for epoch in range(1, args['n_epochs'] + 1):
        train_loader.dataset.gen_data()
        # ema.set_decay_per_step(num_steps_in_epoch=len(train_loader))
        for iterno, (x, y) in enumerate(train_loader):
            net.zero_grad(set_to_none=True)
            x = x.to(Device)
            # print(x.shape)
            y = y.to(Device)
            # print(y.shape)
            y_est = net(x)
            # print(y_est.shape)
            loss = cross_entropy(y_est.view_as(y), y)
            # loss.register_hook(lambda grad: print(grad))
            loss.backward()
            opt.step()
            lr_scheduler.step()
            # ema.update(net, step=steps)

            ######################
            # Update tensorboard #
            ######################
            # writer.add_scalar("lr", opt.param_groups[0]['lr'], steps)
            # writer.add_scalar("metric/train", loss.item(), steps)
            ###########################################################

            steps += 1

            if steps % args['log_interval'] == 0:
                print(
                    "Epoch {} | train: loss {:.8f} | step {}".format(
                        last_epoch + epoch,
                        loss,
                        steps
                    )
                )
            if steps % args['save_interval'] == 0:
                loss_test = run_eval(test_loader, net, cross_entropy)
                # writer.add_scalar("metric/test", loss_test, steps)

                # Check early stopping - pass current epoch
                early_stopping(loss_test, last_epoch + epoch)
                if early_stopping.early_stop:
                    print(f"\n{'='*70}")
                    print(f"EARLY STOPPING: No improvement for {early_stopping.epochs_without_improvement} EPOCHS")
                    print(f"Best Loss: {best_loss:.8f} at Epoch {best_epoch}")
                    print(f"{'='*70}\n")

                    # Adjust parameters for next run (only if optimization is enabled)
                    if enable_optimization:
                        adjust_parameters_on_failure(args, root)
                    else:
                        print("Parameter optimization disabled. No adjusted config will be created.")
                    break

                if loss_test < best_loss:
                    best_loss = loss_test
                    best_epoch = last_epoch + epoch
                    chkpnt = {
                        'model_dict': net.state_dict(),
                        'opt_dict': opt.state_dict(),
                        'step': steps,
                        'best_loss': best_loss,
                        'best_epoch': best_epoch,
                        'args': args
                    }
                    if CheckPoint is not None and os.path.exists(CheckPoint):
                        os.remove(CheckPoint)
                        print("Deleted '%s' successfully" % CheckPoint)
                    CheckPoint = os.path.join(root, 'chkpnt_' + args['net_type'] + '_Best_epoch_' + str(last_epoch + epoch) + '.pt')
                    torch.save(chkpnt, CheckPoint)
                    print('chkpnt_' + args['net_type'] + '_Best_epoch_' + str(last_epoch + epoch) + ' is saved')

                    with open(root / "args.yml", "w") as f:
                        args["best_epoch"] = last_epoch + epoch
                        args["best_loss"] = best_loss
                        yaml.dump(args, f)

                if steps % args['log_interval'] == 0:
                    print(
                        "Epoch {} | train: loss {:.8f} | test: loss {:.8f} | best {:.8f} | best epoch {}".format(
                            last_epoch + epoch,
                            loss,
                            loss_test,
                            best_loss,
                            best_epoch
                        )
                    )
                # costs = []

        # Check if early stopping was triggered in inner loop
        if early_stopping.early_stop:
            break

    # Training completed - check if it was successful (no early stopping)
    training_completed_successfully = not early_stopping.early_stop

    chkpnt = {
        'model_dict': net.state_dict(),
        'opt_dict': opt.state_dict(),
        'step': steps,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'last_epoch': last_epoch + epoch,
        'args': args
    }

    CheckPoint = os.path.join(root, 'chkpnt_' + args['net_type'] + '_Last_epoch_' + str(last_epoch + epoch) + '.pt')
    torch.save(chkpnt, CheckPoint)
    print('chkpnt_' + args['net_type'] + '_Last_epoch_' + str(last_epoch + epoch) + ' is saved')

    with open(root / "args.yml", "w") as f:
        args["last_epoch"] = last_epoch + epoch
        args["best_epoch"] = best_epoch
        args["best_loss"] = best_loss
        yaml.dump(args, f)

    # If training completed successfully and optimization is enabled, update the original config file
    if training_completed_successfully and enable_optimization:
        update_config_on_success(args, original_cfg_path, root)
    elif training_completed_successfully:
        print("\nTraining completed successfully.")
        print("Parameter optimization disabled. Original config will not be updated.")


if __name__ == "__main__":
    train()
