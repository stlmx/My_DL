from typing import Any
import numpy as np

class warmup(object):
    def __init__(self, num_epoch, max_lr) -> None:
        self.num_epoch = num_epoch
        self.max_lr = max_lr
        self.lr = None
    
    def __call__(self, epoch):
        self.lr = self.max_lr / self.num_epoch * epoch
        return self.lr
    
class cos_lr_sche(object):
    def __init__(self, max_epoch) -> None:
        self.lr = None
        self.max_epoch = max_epoch
    
    def __call__(self, epoch) -> Any:
        self.lr = np.cos(np.pi / 2 / self.max_epoch * epoch)
        return self.lr