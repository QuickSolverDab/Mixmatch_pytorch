# Mixmatch_pytorch
This code is pytorch version of the work introduced from ["MixMatch- A Holistic Approach to Semi-Supervised Learning"](https://arxiv.org/abs/1905.02249) by David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver and Colin Raffel. This is not official code, and you can see the official code in [here](https://github.com/google-research/mixmatch).

### Requirements
* python 3.5.2
* pytorch 1.0.1
* torchvision 0.2.2
* tqdm
* numpy
- optional for visualizing and summary network
  * tensorboardX
  * tensorflow 1.12.0
  * torchsummary

### Running
    python main.py --gpu <gpu_id> --dataset <dataset name> --network <network name> --data_dir <data_directory>

### Informantion
This code does not support multi-gpu process but It will be supported soon.
When you run the code, ckpt, logs, and summary file will be saved in ckpt folder.

### Code reference
- https://github.com/pytorch/examples/tree/master/imagenet
- https://github.com/CuriousAI/mean-teacher/tree/master/pytorch
- https://github.com/facebookresearch/mixup-cifar10
- https://github.com/YU1ut/MixMatch-pytorch

# Hello
