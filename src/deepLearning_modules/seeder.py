import os

import torch
import numpy as np

import random

def seed_everything(seed: int) -> None:
    """
    Fixes a seed to ensure reproducibility.

    Parameters:
        seed (int): the seed that will be used

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True