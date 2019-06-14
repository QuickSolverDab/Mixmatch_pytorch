# -*- coding: utf-8 -*-
# This code was created with reference to followings
#   https://github.com/CuriousAI/mean-teacher/tree/master/pytorch
#   https://github.com/pytorch/examples/tree/master/imagenet
#   https://github.com/facebookresearch/mixup-cifar10

import os
import shutil
import torch
import numpy as np
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
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

def make_save_dir(args, save_dir = './ckpt', network=None):
    assert network is not None
    if args.mp:
        save_dir = os.path.join(save_dir, network + '_' + args.dataset, args.time)
    else:
        save_dir = os.path.join(save_dir, 'gpu'+ str(args.gpu)+ '_' + network + '_' + args.dataset,
                                args.repeat + '_' + args.time)
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            pass
    return save_dir

def save_checkpoint(save_dir, args, is_best, state, filename='checkpoint.pth.tar'):
    ckpt_dir = os.path.join(save_dir, filename)
    torch.save(state, ckpt_dir)
    if is_best:
        best_dir = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(ckpt_dir, best_dir)

def record(save_dir, args, state):
    if not os.path.exists(os.path.join(save_dir, '{}_Record_{}.txt'.format(args.repeat, args.time))):
        file = open(os.path.join(save_dir, '{}_Record_{}.txt'.format(args.repeat, args.time)), 'w')
        file.write('gpu : {} \n'.format(args.gpu))
        file.write('Network : {} \n'.format(args.network) )
        file.write('Dataset : {} \n'.format(args.dataset))
        file.write('# of Batch : {} \n'.format(args.batch_size))
        file.write('# of labeled data per classes : {} \n'.format(args.num_labeled_per_classes))
        file.write('# of unlabeled batch: {}\n'.format(args.K))
        file.write('\n')

        file.write('Hyper-prameters\n')
        file.write('initial learning : {}\n'.format(args.lr))
        file.write('exponential moving average decay: {}\n'.format(args.ema))
        file.write('Temperature for sharpening: {}\n'.format(args.T))
        file.write('w for ratio between losses: {}\n'.format(args.w))
        file.write('alpha for beta distributon of mixup: {}\n'.format(args.alpha))
        file.write('\n')
        file.close()
    else:
        file = open(os.path.join(save_dir, '{}_Record_{}.txt'.format(args.repeat, args.time)), 'a')
        file.write('Epoch: {:3d}, Train acc {:.4f}, Test acc {:.4f} \n'
                   .format(state['epoch'], state['train_acc'], state['test_acc']))
        file.close()
    shutil.copy(os.path.join(save_dir, '{}_Record_{}.txt'.format(args.repeat,args.time)),
                os.path.join(save_dir, '../'))

def idx2onehot(idx, n, args):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).cuda(args.gpu)
    onehot.scatter_(1, idx, 1)
    return onehot

"""Mixup
"""
def mixup_data(x, y, alpha=0.2, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lamb'''
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lamb * x + (1 - lamb) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lamb

def mixup_criterion(pred, y_a, y_b, lamb):
    return lamb * F.cross_entropy(pred, y_a) + (1 - lamb) * F.cross_entropy(pred, y_b)


"""Mixmatch
"""
def mixmatch_data(x, y, alpha=0.75, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lamb'''
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha) # For shuffle
        lamb = max(lamb, 1. - lamb) # Different point with org mixup
    else:
        lamb = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lamb * x + (1 - lamb) * x[index, :]
    mixed_target = lamb * y + (1 - lamb) * y[index] # little bit different point with org mixup
    return mixed_x, mixed_target

def mixmatch_criterion(pred_l, pred_u, target_l, target_u, args):
    l_x = cross_entropy(pred_l, target_l)
    l_u = F.mse_loss(F.softmax(pred_u), target_u)
    return l_x + args.w * l_u

def cross_entropy(input, target):
    """ Cross entropy for one-hot labels
    """
    return -torch.mean(torch.sum(target * F.log_softmax(input), dim=1))

def permuted_idx(batch_size, K):
    """ index for permutation
        
    """
    idx_l = torch.randperm(batch_size)
    idx_u = torch.randperm(batch_size*K) + batch_size
    # mix with specific portion
    mix_len = batch_size - (batch_size // K) + 1
    idx = torch.cat((idx_u[:mix_len], idx_l[mix_len:], idx_l[:mix_len], idx_u[mix_len:]), dim=-1)
    per = torch.cat((idx_l,idx_u), dim = -1)
    return idx[per]
