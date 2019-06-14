from __future__ import division
from __future__ import unicode_literals
import torch

# Refer from https://github.com/CuriousAI/mean-teacher/tree/master/pytorch
#            https://github.com/YU1ut/MixMatch-pytorch
class WeightEMA:
    """
    Maintains (exponential) moving average of a set of parameters.
    """
    def __init__(self, net, ema_net, args):
        """ Args:
            net: updated by training
            ema_net: updated by ema
            args.ema: decaying parameters(0.999)
            args.lr: initial learning rate(0.002)
        """
        self.alpha = args.ema
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.model = net
        self.ema_model = ema_net
        self.wd = 0.02 * args.lr # Hyper parameters of Mixmatch

        # start from equal network parameters
        model_statedict = self.model.state_dict()
        self.ema_model.load_state_dict(model_statedict)

    def step(self, bn=False):
        if bn:
            # related mention in the official code from https://github.com/google-research/mixmatch/blob/master/mixmatch.py
            model_statedict = self.model.state_dict()
            ema_statedict = self.ema_model.state_dict()
            params = [name for name, param in self.ema_model.named_parameters()]
            for i in params:
                del model_statedict[i]
            ema_statedict.update(model_statedict)
            self.ema_model.load_state_dict(ema_statedict)
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay for model
                param.data.mul_(1 - self.wd)
