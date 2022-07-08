import torch
import random


class Cutout:
    def __init__(self, p=0.5, max_len=None):
        self.p = p
        self.max_len = max_len

    def __call__(self, x, y):
        if self.p < random.random():
            if self.max_len is None:
                self.max_len = x.shape[-1] // 10
            idx = random.randint(0, self.max_len - 1)
            x[idx:idx + self.max_len] = 0
        return x, y


class Flip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x, y):
        if self.p < random.random():
            x = x.flip(dims=[-1, ])
            y = y.flip(dims=[-1, ])
        return x, y


class Augs:
    def __init__(self, p=0.5, augs=[]) -> None:
        self.augs = []
        for a in augs:
            if a == 'flip':
                self.augs.append(Flip(p=p))
            elif a == 'cutout':
                self.augs.append(Cutout(p=p))
            else:
                raise ValueError('wrong aug reveived {}'.format(a))

    def __call__(self, x, y):
        random.shuffle(self.augs)
        for a in self.augs:
            x, y = a(x, y)
        return x, y