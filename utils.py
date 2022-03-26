from __future__ import annotations

import os
import shutil
from argparse import Namespace

import torch
from tensorboard_logger import log_value


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val: torch.Tensor | float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer: torch.optim.SGD, epoch: int, args: Namespace):
    """Decays LR by 10 after 90, 180 and 270 epochs"""
    lr = args.lr * (0.1 ** (epoch // 90)) * (0.1 ** (epoch // 180)) * (0.1 ** (epoch // 270))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state: dict, is_best: bool, filename: str = 'checkpoint.pth.tar', args: Namespace = None):
    """Saves checkpoint to disk"""
    directory = f"runs/{args.name}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'runs/{args.name}/model_best.pth.tar')


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list[torch.Tensor]:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
