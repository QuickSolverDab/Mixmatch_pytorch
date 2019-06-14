from models.resnet import *
from models.vgg import *
from models.wrn import *
from models.preresnet import *
from models.pyramidnet import *

def networks(network,**kwargs):
    # ResNet
    if 'resnet' in network and 'pre' not in network:
        depth =  int(network[6:])
        return resnet(depth, **kwargs)

    elif 'vgg' in network:
        depth = int(network[3:5])
        if 'bn' in network:
            return vgg_bn(depth, **kwargs)
        else:
            return vgg(depth, **kwargs)

    elif 'wideResNet' in network:
        depth = int(network[10:12])
        widen_factor = int(network[13:])
        return wideResNet(depth, widen_factor, **kwargs)

    elif 'preresnet' in network:
        depth = int(network[9:])
        return preresnet(depth, **kwargs)

    elif 'pyramidnet' in network:
        depth = int(network[10:])
        return pyramidnet(depth, **kwargs)
