import torch
from torch.optim import lr_scheduler
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
import math
import random
import os
from torch.nn import MaxPool1d,AvgPool1d
from torch import Tensor
from typing import Iterable, Set, Tuple


def seed_torch(seed=2020):

    # 加入以下随机种子，数据输入，随机扩充等保持一致
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 加入所有随机种子后，模型更新后，中间结果还是不一样，
    # 发现这一的现象：前两轮，的结果还是一样；随着模型更新结果会变；
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'poly':
        max_step = opt.niter+opt.niter_decay
        power = 0.9
        def lambda_rule(epoch):
            current_step = epoch + opt.epoch_count
            lr_l = (1.0 - current_step / (max_step+1)) ** float(power)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, init_step=0, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = init_step
        print(self.global_step)
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class SGDROptimizer(torch.optim.SGD):

    def __init__(self, params, steps_per_epoch, lr=0, weight_decay=0, epoch_start=1, restart_mult=2):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.local_step = 0
        self.total_restart = 0

        self.max_step = steps_per_epoch * epoch_start
        self.restart_mult = restart_mult

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.local_step >= self.max_step:
            self.local_step = 0
            self.max_step *= self.restart_mult
            self.total_restart += 1

        lr_mult = (1 + math.cos(math.pi * self.local_step / self.max_step))/2 / (self.total_restart + 1)

        for i in range(len(self.param_groups)):
            self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.local_step += 1
        self.global_step += 1


def split_dataset(dataset, n_splits):

    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out


def decode_seg(label_mask, toTensor=False):
    """
    :param label_mask: mask (np.ndarray): (M, N)/  tensor: N*C*H*W
    :return: color label: (M, N, 3),
    """
    if not isinstance(label_mask, np.ndarray):
        if isinstance(label_mask, torch.Tensor):  # get the data from a variable
            image_tensor = label_mask.data
        else:
            return label_mask
        label_mask = image_tensor[0][0].cpu().numpy()

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3),dtype=np.float)
    r = label_mask % 6
    g = (label_mask % 36) // 6
    b = label_mask // 36
    # 归一化到[0-1]
    rgb[:, :, 0] = r / 6
    rgb[:, :, 1] = g / 6
    rgb[:, :, 2] = b / 6
    if toTensor:
        rgb = torch.from_numpy(rgb.transpose([2,0,1])).unsqueeze(0)

    return rgb


def tensor2im(input_image, imtype=np.uint8, normalize=True):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        if image_numpy.shape[0] == 3:  # if RGB
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            if normalize:
                image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2np(input_image, if_normalize=False):
    """
    :param input_image: C*H*W / H*W
    :return: ndarray, H*W*C / H*W
    """
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array

    else:
        image_numpy = input_image
    if image_numpy.ndim == 2:
        return image_numpy
    elif image_numpy.ndim == 3:
        C, H, W = image_numpy.shape
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        #  如果输入为灰度图C==1，则输出array，ndim==2；
        if C == 1:
            image_numpy = image_numpy[:, :, 0]
        if if_normalize:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
            #  add to prevent extreme noises in visual images
            image_numpy[image_numpy<0]=0
            image_numpy[image_numpy>255]=255
    return image_numpy


import ntpath
from misc.imutils import save_image
def save_visuals(visuals, img_dir, name, save_one=True, iter='0'):
    """
    """
    # save images to the disk
    for label, image in visuals.items():
        N = image.shape[0]
        if save_one:
            N = 1
        # 保存各个bz的数据
        for j in range(N):
            name_ = ntpath.basename(name[j])
            name_ = name_.split(".")[0]
            # print(name_)
            image_numpy = tensor2np(image[j], if_normalize=True).astype(np.uint8)
            # print(image_numpy)
            img_path = os.path.join(img_dir, iter+'_%s_%s.png' % (name_, label))
            save_image(image_numpy, img_path)