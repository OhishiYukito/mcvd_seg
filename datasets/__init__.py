# https://github.com/voletiv/mcvd-pytorch/blob/master/datasets/__init__.py

from datasets.bair import BAIRDataset
from datasets.cityscapes import CityscapesDataset
from datasets.kth64 import KTHDataset
from datasets.stochastic_moving_mnist import StochasticMovingMNIST
from datasets.davis import DavisHDF5Dataset
from datasets.ucf101 import UCF101Dataset

import os
import torch

DATASETS = ['BAIR64', 'KTH64', 'STOCHASTICMOVINGMNIST', 'UCF101']

def get_dataset(config, segmentation=False, data_path=None):
    
    assert config.data.dataset.upper() in DATASETS, \
        f"datasets/__init__.py: dataset can be only in {DATASETS}! config.data.dataset is {config.data.dataset}!"
    
    if segmentation:
        if config.data.seg_dataset.upper() == "DAVIS":
            if data_path is None:
                data_path = 'datasets/DAVIS64_h5'
            if getattr(config.data, 'size', 64)==64:
                data_path = 'datasets/DAVIS64_h5'
            else:
                data_path = 'datasets/DAVIS_h5'
            seq_len = config.data.num_frames
            grayscale = (config.data.channels==1)
            train_dataset = DavisHDF5Dataset(data_path=os.path.join(data_path, 'train'), batch_size=config.train.batch_size, frames_per_sample=seq_len, image_size=getattr(config.data, 'size', 64),
                                            random_time=True, random_horizontal_flip=getattr(config.data, 'random_flip', True), grayscale=grayscale)
            test_dataset = DavisHDF5Dataset(data_path=os.path.join(data_path, 'test'),  batch_size=config.train.batch_size, frames_per_sample=seq_len, image_size=getattr(config.data, 'size', 64),
                                            random_time=True, random_horizontal_flip=False, grayscale=grayscale)
    
    else:
        if config.data.dataset.upper() == "BAIR64":
            if data_path is None:
                data_path = 'datasets/BAIR_h5'
            frames_per_sample = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + config.data.num_frames
            train_dataset = BAIRDataset(os.path.join(data_path, "train"), frames_per_sample=frames_per_sample, random_time=True,
                                random_horizontal_flip=getattr(config.data, 'random_flip', True), color_jitter=getattr(config.data, 'color_jitter', 0.0))
            test_dataset = BAIRDataset(os.path.join(data_path, "test"), frames_per_sample=frames_per_sample, random_time=True,
                                    random_horizontal_flip=False, color_jitter=0.0)
        
        elif config.data.dataset.upper() == "KTH64":
            # KTH64_h5 (data_path)
            # |-- shard_0001.hdf5
            if data_path is None:
                data_path = 'datasets/KTH64_h5'
            frames_per_sample = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + config.data.num_frames
            train_dataset = KTHDataset(data_path, frames_per_sample=frames_per_sample, train=True,
                                    random_time=True, random_horizontal_flip=getattr(config.data, 'random_flip', True), with_target=False)
            test_dataset = KTHDataset(data_path, frames_per_sample=frames_per_sample, train=False,
                                    random_time=True, random_horizontal_flip=False, with_target=False, total_videos=256)
            
        
        elif config.data.dataset.upper() == "STOCHASTICMOVINGMNIST":
            if data_path is None:
                data_path = 'datasets/MNIST'
            seq_len = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + config.data.num_frames
            train_dataset = StochasticMovingMNIST(data_path, train=True, seq_len=seq_len, num_digits=getattr(config.data, "num_digits", 2),
                                                #step_length=config.data.step_length, 
                                                with_target=False)
            test_dataset = StochasticMovingMNIST(data_path, train=False, seq_len=seq_len, num_digits=getattr(config.data, "num_digits", 2),
                                                #step_length=config.data.step_length, 
                                                with_target=False, total_videos=256)
        

        elif config.data.dataset.upper() == "UCF101":
            # UCF101_h5 (data_path)
            # |-- shard_0001.hdf5
            if data_path is None:
                data_path = 'datasets/UCF101_h5'
            frames_per_sample = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0) + config.data.num_frames
            train_dataset = UCF101Dataset(data_path, frames_per_sample=frames_per_sample, image_size=getattr(config.data, 'image_size', 64), train=True, random_time=True,
                                    random_horizontal_flip=getattr(config.data, 'random_flip', True), with_target=False)
            test_dataset = UCF101Dataset(data_path, frames_per_sample=frames_per_sample, image_size=getattr(config.data, 'image_size', 64), train=False, random_time=True,
                                        random_horizontal_flip=False, total_videos=256, with_target=False)
    
    return train_dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    """ transform X range
    
    data.uniform_dequantization: conduct uniform dequantization that convert discrete to continuous [0,256]
    data.gaussian_dequantization: conduct gaussian dequantization that convert discrete to continuous [0,256]
    data.rescaled: [0,1] -> [-1,1]
    data.logit_transform: x -> log(x / (1-x)) ([0,1] -> [-inf, inf])
    """
    if getattr(config.data, 'uniform_dequantization', False):
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    """ transform X range to [0.0, 1.0]

    """
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)