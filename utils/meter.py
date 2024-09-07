import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        super(AverageMeter).__init__()
        self.val = 0
        self.sum = 0
        self.count = 0

    def init_param(self, param_value):
        self.val = torch.zeros_like(param_value)
        self.sum = torch.zeros_like(param_value)
        self.count = 0
        pass

    def update(self, val, ratio=1.0, n=1):
        self.val = val
        self.sum += val * ratio
        self.count += n
