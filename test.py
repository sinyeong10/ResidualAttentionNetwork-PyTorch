import time

import torch.nn as nn
from torch.utils.data import DataLoader

# based https://github.com/liudaizong/Residual-Attention-Network
from residual_attention_network import ResidualAttentionModel92U as ResidualAttentionModel
from utils import *


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def validate(model: ResidualAttentionModel, val_loader: DataLoader,
             criterion: nn.CrossEntropyLoss, epoch: int, args: Namespace):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.perf_counter()
    for i, (inp, target) in enumerate(val_loader):
        target: torch.Tensor = target.cuda(non_blocking=True)
        inp: torch.Tensor = inp.cuda()

        # compute output
        with torch.no_grad():
            output: torch.Tensor = model(inp)
            loss: torch.Tensor = criterion(output, target)

        # measure accuracy and record loss
        prec1: torch.Tensor = accuracy(output, target, topk=(1,))[0]
        losses.update(loss, inp.size(0))
        top1.update(prec1, inp.size(0))

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


def test(model: ResidualAttentionModel, test_loader: DataLoader):
    """Perform testing on the test set"""
    top1 = AverageMeter()

    model.eval()

    correct = 0
    total = 0
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    for images, labels in test_loader:
        images: torch.Tensor = images.cuda()
        labels: torch.Tensor = labels.cuda(non_blocking=True)

        with torch.no_grad():
            outputs: torch.Tensor = model(images)

        _, predicted = torch.max(outputs.cuda(), 1)
        prec1: torch.Tensor = accuracy(outputs, labels, topk=(1,))[0]
        top1.update(prec1, images.size(0))

        total += labels.size(0)
        correct += (predicted == labels).sum()
        #
        c = (predicted == labels).squeeze()
        for i in range(16):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    print(f'Accuracy of the model on the test images: {top1.avg:.2f}%')
    for i in range(10):
        print(f'Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]: .2f}%')
