import torch
from torch.utils.data import Sampler
import math
import random

class RASampler(Sampler):
    """Repeated Augmentation Sampler (for repeated augmentation training)"""
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # repeat 3x for repeated augmentation
        indices = list(range(len(self.dataset))) * 3
        if self.shuffle:
            random.shuffle(indices)
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples