import time
import torch
import math
import random
import numpy as np
import torch.nn.functional as F
from tqdm  import trange
from utils import *

def network_train(train_loader_l, train_loader_u, model,
                  optimizer, ema_optm, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()

    total_step = len(train_loader_u)

    tbar = trange(0, total_step, total=total_step, initial=0)
    train_flow_l = iter(train_loader_l)
    train_flow_u = iter(train_loader_u)
    for steps in tbar:
        # Labeled
        try:
            input_l, target_l = next(train_flow_l)
        except StopIteration:
            train_flow_l      = iter(train_loader_l)
            input_l, target_l = next(train_flow_l)

        # Unlabeled
        try:
            input_u, _   = next(train_flow_u)
        except StopIteration:
            train_flow_u = iter(train_loader_u)
            input_u, _   = next(train_flow_u)


        if args.gpu is not None:
            input_l = input_l.cuda(args.gpu)
            target_l = target_l.cuda(args.gpu, non_blocking=True)
            input_u = torch.cat(input_u, 0)
            input_u = input_u.cuda(args.gpu)

        batch_size = input_l.size(0)

        """Label Guessing
        """
        with torch.no_grad():
            output_u = model(input_u)
            # Algorithm 1. line 7
            guess = F.softmax(output_u).view(args.K, len(output_u)//args.K, -1).sum(dim=0) / args.K
            # Algorithm 1. line 8 (Sharpen)
            pow_guess = torch.pow(guess, 1. / args.T)
            sharpened_guess = pow_guess / pow_guess.sum(dim=1, keepdim=True)
            sharpened_guess = sharpened_guess.detach()

        """Mix match
        """
        # Algorithm 1. line 10 to 11 and concat in laine 12
        # Combine labels and guessed labels
        input = torch.cat((input_l, input_u), dim=0)
        target_l_onehot = idx2onehot(target_l, args.num_classes, args)
        target = torch.cat([target_l_onehot, sharpened_guess, sharpened_guess], dim=0)

        # Algorithm 1. shuffle in line 12 and mixup in line 13, 14
        mixed_input, mixed_target = mixmatch_data(input, target)
        mixed_output = model(mixed_input)
        loss = mixmatch_criterion(mixed_output[:batch_size] , mixed_output[batch_size:],
                                  mixed_target[:batch_size], mixed_target[batch_size:],
                                  args)

        # measure accuracy and record loss
        # only labeled dataset
        output_l = model(input_l)
        acc1, acc5 = accuracy(output_l, target_l, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optm.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if steps == total_step // 5 or steps ==0:
            tbar.set_description('Epoch: [{:3d}] '
                    'Loss  {loss.val:7.4f} ({loss.avg:7.4f}) '
                    'Acc@1 {top1.val:7.3f} ({top1.avg:7.3f}) '
                    'Acc@5 {top5.val:7.3f} ({top5.avg:7.3f})'.format(
                    epoch, loss=losses, top1=top1, top5=top5))

    tbar.set_description('\n')

    ema_optm.step(bn=True)

    return top1.avg


def validate(val_loader, model, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        total_step = len(val_loader)
        val_flow = iter(val_loader)
        tbar = trange(0, total_step, total=total_step, initial=0)
        for steps in tbar:
            try:
                input, target = next(val_flow)
            except StopIteration:
                continue
            if args.gpu is not None:
                input = input.cuda(args.gpu)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if steps == (total_step // 5 ) or steps == 0:
                tbar.set_description('Test'
                            'Loss  {loss.val:7.4f} ({loss.avg:7.4f}) '
                            'Acc@1 {top1.val:7.3f} ({top1.avg:7.3f}) '
                            'Acc@5 {top5.val:7.3f} ({top5.avg:7.3f})'.format(
                            loss=losses, top1=top1, top5=top5))

        tbar.set_description('\n')
        print(' *** Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                            .format(top1=top1, top5=top5))
    return top1.avg


def interleave_offsets(batch, K):
    groups = [batch // K] * K
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(inputs, batch, K):
    K = K + 1
    offsets = interleave_offsets(batch, K)
    inputs = [[v[offsets[p]:offsets[p + 1]] for p in range(K)] for v in inputs]
    for i in range(1, K):
        inputs[0][i], inputs[i][i] = inputs[i][i], inputs[0][i]
    return [torch.cat(v, dim=0) for v in inputs]
