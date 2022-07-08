from hashlib import new
from typing import OrderedDict
import torch
import torch.nn as nn
from copy import deepcopy
import math


class EMA:
    def __init__(self, model, step_mod_factor=5, decay_per_epoch=0.8):
        # makes a copy of the model parameters for averaging the weights
        # model - network
        # step_mod_factor - number of ema updates per epoch
        # decay_per_epoch - total decay per epoch
        self.decay_per_epoch = decay_per_epoch
        self.step_mod_factor = step_mod_factor
        self.decay_per_step = []
        self.ema = deepcopy(model)
        self.ema.eval()
        self.ema.cuda()
        self.ema_has_module = hasattr(self.ema, 'module')
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def to_gpu(self):
        self.ema.to(device='cuda')

    def to_cpu(self):
        self.ema.to(device='cpu')

    def set_decay_per_step(self, num_steps_in_epoch):
        num_ema_steps_in_epoch = float(num_steps_in_epoch) / self.step_mod_factor
        self.decay_per_step = math.pow(self.decay_per_epoch, 1.0 / num_ema_steps_in_epoch)

    def update(self, model, step):
        if step % self.step_mod_factor != 0:
            return
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                # actual update
                if msd[k].dtype == torch.long:
                    ema_v.copy_(msd[k])
                else:
                    update_ema_jit(ema_v, msd[k], self.decay_per_step, 1. - self.decay_per_step)

    def get_dict(self, model):
        with torch.no_grad():
            needs_module = hasattr(model, 'module') and not self.ema_has_module
            if not needs_module:
                new_state_dict = deepcopy(self.ema.state_dict())
            else:
                new_state_dict = OrderedDict()
                for k, v in self.ema.state_dict().items():
                    name = 'module.' + k
                    new_state_dict[name] = v
                new_state_dict = deepcopy(new_state_dict)
            return new_state_dict


@torch.jit.script
def update_ema_jit(ema_v: torch.Tensor, model_v: torch.Tensor, decay_per_step: float, model_factor: float):
    ema_v.mul_(decay_per_step).add_(model_factor * model_v.float())

