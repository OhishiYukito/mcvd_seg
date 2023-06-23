# https://github.com/voletiv/mcvd-pytorch/blob/master/models/__init__.py

# SMLD: s = -1/sigma * z
# DDPM: s = -1/sqrt(1 - alpha) * z
# All `scorenet` models return z, not s!

import torch
import logging
import numpy as np

from functools import partial
from scipy.stats import hmean
from torch.distributions.gamma import Gamma
from tqdm import tqdm
#from . import pndm


def get_sigmas(config):

    T = getattr(config.model, 'num_forward_steps')

    if config.model.sigma_dist == 'geometric':
        return torch.logspace(np.log10(config.model.sigma_begin), np.log10(config.model.sigma_end),
                              T).to(config.device)

    elif config.model.sigma_dist == 'linear':
        return torch.linspace(config.model.sigma_begin, config.model.sigma_end,
                              T).to(config.device)

    elif config.model.sigma_dist == 'cosine':
        t = torch.linspace(T, 0, T+1)/T
        s = 0.008
        f = torch.cos((t + s)/(1 + s) * np.pi/2)**2
        return f[:-1]/f[-1]

    else:
        raise NotImplementedError('sigma distribution not supported')
    