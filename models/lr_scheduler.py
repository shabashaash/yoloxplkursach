import torch.optim as optim
import numpy as np


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        lr_factor = 0.5 * (1 + np.cos(np.pi * step / self.max_num_iters))
        if step <= self.warmup:
            lr_factor *= (step * 1.0 + 0.00001) / self.warmup
        return lr_factor
