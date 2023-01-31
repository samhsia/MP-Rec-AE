import argparse

import numpy as np
from numpy import random as ra
import random

import torch

# Base configurations
def parse_configurations():
    parser = argparse.ArgumentParser(description='MetaRec Benchmarks')

    # Dataset Arguments
    parser.add_argument('--num-batches', type=int, default=100, help='Number of batches')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--dense-dim', type=int, default=13, help='Dense feature dimension')
    parser.add_argument('--table-sizes', type=str, default='10-100-1000-10000-100000', help='Embedding table sizes')
    parser.add_argument('--num-lookups', type=int, default=1, help='Number of lookups per embedding table')

    # Model Architecture Arguments
    parser.add_argument('--emb-dim', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--mlp-bot-dims', type=str, default='13-512-256-64-16', help='Bottom MLP Dimensions')
    parser.add_argument('--mlp-top-dims', type=str, default='512-256-1', help='Top MLP Dimensions')
    parser.add_argument('--precision', type=str, default='np.float32', help='Precision')

    # DHE Arguments
    parser.add_argument('--dhe-activation', type=str, default='sigmoid', help='DHE Decoder MLP Activation Function')
    parser.add_argument('--dhe-batch-norm', action="store_true", default=False, help='DHE Decoder MLP Batch Norm')
    parser.add_argument('--dhe-hash-fn', type=str, default='universal', help='DHE Encoder Hash Function')
    parser.add_argument('--dhe-k', type=int, default=1024, help='DHE Encoder Parameter k')
    parser.add_argument('--dhe-m', type=int, default=1000000, help='DHE Encoder Parameter m')
    parser.add_argument('--dhe-mlp-dims', type=str, default='1024-128-16', help='Top MLP Dimensions')
    parser.add_argument('--dhe-transform', type=str, default='uniform', help='DHE Encoder Transform Distribution')

    # Training Arguments
    parser.add_argument('--num-epochs', type=int, default=2, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning Rate')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    # Printing Arguments
    parser.add_argument('--print-freq', type=int, default=10, help='Printing Frequency')
    parser.add_argument('--print-model-stats', action='store_true', default=False, help='Print model statistics')

    # Select DLRM Mode
    parser.add_argument('--device', type=str, default='cpu', help='Target Device Type')
    parser.add_argument('--train', action="store_true", default=False, help='Training benchmark')
    parser.add_argument('--inference', action="store_true", default=False, help='Inference benchmark')

    # IPU Options
    parser.add_argument('--ipu-offload', action="store_true", default=False, help='Offload weights to Streaming Memory')
    parser.add_argument('--offload-threshold', type=int, default=1000, help='Parameter size threshold for Streaming Memory offloading.')
    parser.add_argument('--ipu-device-iterations', type=int, default=1, help='Number of batches in one loop.')
    parser.add_argument('--ipu-replicas', type=int, default=1, help='Number of IPUs for one copy of model.')

    # TPU Options
    parser.add_argument('--tpu-debug', action="store_true", default=False, help='Print TPU Debugging Information')
    parser.add_argument('--num-tpus', type=int, default=1, help='Number of TPU cores to use.')

    return parser.parse_args()

# Configurations for microbenchmarking
def parse_configurations_sweeps():
    parser = argparse.ArgumentParser(description='MetaRec Sweeps')
    parser.add_argument('--results-dir', type=str, default='./results', help='Directory for sweep results')
    return parser.parse_args()

def estimate_config_size(table_sizes, emb_dim, bytes_per_param = 4):
    total_sizes = np.sum(np.array(parse_dims(table_sizes)))
    num_params  = total_sizes * emb_dim
    total_bytes = num_params * bytes_per_param
    return total_bytes

# Fix seed for reproducibility
def fix_seed(seed):
    random.seed(seed)
    ra.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.deterministic = True

def ipu_unpack_batch(inputs):
    x_dense, x_offsets, x_indices, y = inputs
    x_dense   = torch.squeeze(x_dense, dim=0)
    x_offsets = torch.squeeze(torch.stack(x_offsets), dim=1)
    x_indices = torch.squeeze(torch.stack(x_indices), dim=1)
    y         = torch.squeeze(y, dim=0)
    return x_dense, x_offsets, x_indices, y

def multi_ipu_unpack_batch(inputs):
    x_dense, x_offsets, x_indices, y = ipu_unpack_batch(inputs)
    x_dense   = torch.flatten(x_dense, start_dim=0, end_dim=1)
    x_offsets = torch.flatten(x_offsets, start_dim=1, end_dim=-1).T
    x_indices = torch.flatten(x_indices, start_dim=1, end_dim=-1).T
    y         = torch.flatten(y, start_dim=0, end_dim=1)
    return x_dense, x_offsets, x_indices, y

# Parse dimensions string
def parse_dims(dims):
    dims_parsed = [int(i) for i in dims.split('-')]
    return dims_parsed

# Convert (offsets, indices) to KeyedJaggedTensor 
def convert_to_kjt(offsets, indices):
    num_tables  = len(offsets)
    num_batches = len(offsets[0])
    num_lookups = len(indices[0])//num_batches

    keys    = ["sparse_fea_{}".format(num) for num in range(num_tables)]
    values  = torch.flatten(indices)
    offsets = torch.arange(num_tables * num_batches + 1) * num_lookups

    features = KeyedJaggedTensor(
        keys    = keys,
        values  = values,
        offsets = offsets
    )

    return features