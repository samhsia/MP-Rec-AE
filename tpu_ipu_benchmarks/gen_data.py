import numpy as np
from numpy import random as ra

import sys

import torch
from torch.utils.data import Dataset

from utils import fix_seed

# Generate IDs for a given sparse feature
def generate_ids(
    table_size,
    num_ids,
    distribution
):
    if distribution == "uniform":
        ids = ra.random(num_ids)
        ids = np.round(ids * (table_size - 1)).astype(np.int64)
    elif distribution == "gaussian":
        mu    = (table_size - 1) / 2.0
        sigma = mu / 3
        ids   = ra.normal(mu, sigma, num_ids)
        ids   = np.clip(ids, 0, table_size-1).astype(np.int64)
    else:
        sys.exit("ERROR: Specified distribution for Sparse ID generation is not supported.")
    return ids
    
# Generate a batch of random data (both dense and sparse inputs)
def generate_data_batch(
    batch_size, 
    dense_dim, 
    table_sizes, 
    num_lookups,
    distribution,
    precision
):
    # Generate dense input
    x_dense = torch.tensor(ra.rand(batch_size, dense_dim).astype(precision))

    # Generate sparse input
    x_offsets = []
    x_indices = []
    
    for size in table_sizes:
        batch_offsets = []
        batch_indices = []
        offset = 0
        
        for _ in range(batch_size):
            ids = generate_ids(size, num_lookups, distribution)
            batch_offsets += [offset]
            batch_indices += ids.tolist()
            offset        += num_lookups
        
        x_offsets.append(torch.tensor(batch_offsets))
        x_indices.append(torch.tensor(batch_indices))
        
    return (x_dense, x_offsets, x_indices)

# Generate a batch of outputs (click probabilities)
def generate_output_batch(
    batch_size, 
    precision
):
    outputs = np.round(ra.rand(batch_size, 1)).astype(precision) # round to {0, 1}
    return torch.tensor(outputs)

# Class for generated dataset
class GeneratedDataset(Dataset):
    def __init__(
        self,
        num_batches,
        batch_size,
        dense_dim,
        table_sizes,
        num_lookups,
        precision,
        distribution="uniform",
        seed=1234
    ):
        self.num_batches  = num_batches
        self.batch_size   = batch_size
        self.dense_dim    = dense_dim
        self.table_sizes  = table_sizes
        self.num_lookups  = num_lookups
        self.precision    = precision
        self.distribution = distribution
        self.seed         = seed
        
        fix_seed(self.seed)
        
    def __getitem__(self, index):
        # Handle case when indices into dataset form a slice
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]
        
        # Generate batch of inputs
        (x_dense, x_offsets, x_indices) = generate_data_batch(
            self.batch_size,
            self.dense_dim,
            self.table_sizes,
            self.num_lookups,
            self.distribution,
            self.precision
        )
        
        # Generate batch of outputs
        y = generate_output_batch(self.batch_size, self.precision)
        
        return (x_dense, x_offsets, x_indices, y)
    
    def __len__(self):
        return self.num_batches

# Make a PyTorch dataloader out of dataset
def make_dataloader(dataset, sampler=None):
    def collate_wrapper_fn(list_of_tuples):
        # where each tuple is (x_dense, x_offsets, x_indices, y)
        (x_dense, x_offsets, x_indices, y) = list_of_tuples[0]
        return (x_dense,
                torch.stack(x_offsets),
                torch.stack(x_indices),
                y)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, # fixed at 1 here
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wrapper_fn,
        pin_memory=False,
        drop_last=False,
        sampler=sampler,
    )
    
    return dataloader