import torch
from torch import nn
from torch.nn import functional as F
import logging
import shutil
import math
from datetime import datetime
import os
import numpy as np

def get_log(log_folder, is_test=False):
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    time_str = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if is_test:
        print("log into file %s" % 'test_log_%s.txt' % time_str)
        fh = logging.FileHandler("%s/test_log_%s.txt" % (log_folder, time_str), mode='w')
    else:
        print("log into file %s" % 'log_%s.txt' % time_str)
        fh = logging.FileHandler("%s/log_%s.txt" % (log_folder, time_str), mode='w')

    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def save_checkpoint(state, is_best, folder, filename='checkpoint.pth.tar'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    suffix = filename[filename.find('.'):]
    filename = "%s/%s" % (folder, filename)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/model_best%s' % (folder, suffix))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
