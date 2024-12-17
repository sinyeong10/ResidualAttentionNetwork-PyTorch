import time

import torch.nn as nn
from torch.utils.data import DataLoader

# based https://github.com/liudaizong/Residual-Attention-
#ResidualAttentionModel92U, ResidualAttentionModel92
from residual_attention_network import ResidualAttentionModel56 as ResidualAttentionModel
from utils import *

# used for logging to TensorBoard
from tensorboard_logger import log_value
check_shape = False

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def train(model: ResidualAttentionModel, train_loader: DataLoader, criterion: nn.CrossEntropyLoss,
          optimizer: torch.optim.SGD, epoch: int, args: Namespace):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.perf_counter()
    for i, (images, labels) in enumerate(train_loader):
        if check_shape:
            print("\n\nimages", images.shape) #images torch.Size([8, 3, 32, 32])
        images = images.cuda()
        labels = labels.cuda(non_blocking=True)

        # ForProp
        output = model(images)
        loss = criterion(output, labels)

        # measure accuracy and record loss
        prec1 = accuracy(output, labels, topk=(1,))[0]
        losses.update(loss, images.size(0))
        top1.update(prec1, images.size(0))

        # BackProp + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)
