##############################  IMPORTS ##############################
import math
import numpy as np
from numpy import random as ra
from primesieve import nth_prime

import sys
from random import randint

import torch
import torch.nn as nn
# from torch.autograd.profiler import record_function

import time
from typing import List
from typing_extensions import TypedDict

from utils import fix_seed
##############################  IMPORTS ##############################

class DHEBag(nn.Module):
    def __init__(
        self, 
        table_sizes: List, 
        args_dhe: TypedDict
    ):
        super(DHEBag, self).__init__()

        self.table_sizes = table_sizes
        self.args_dhe    = args_dhe
        self.emb_dim     = self.args_dhe['mlp_dims'][-1]

        self.hash_params, self.hash_params_size, self.hash_params_flops = \
            self.generate_hash_params(
                self.table_sizes, 
                self.args_dhe
            )
        self.dhe_mlps, self.dhe_mlps_size, self.dhe_mlps_flops = \
            self.create_mlp(
                self.table_sizes, 
                self.args_dhe
            )

    ##############################  ACCOUNTING / DHE STATISTICS ##############################
    def get_sizes(
        self
    ):
        dhe_size_enc = self.hash_params_size
        dhe_size_dec = self.dhe_mlps_size
        return dhe_size_enc, dhe_size_dec

    def get_flops(
        self,
    ):
        dhe_flops_enc_hash = self.hash_params_flops
        if self.args_dhe["transform"] == "uniform":
            dhe_flops_enc_tran = len(self.table_sizes) * self.args_dhe['k'] * 4
        elif self.args_dhe["transform"] == 'gaussian':
            dhe_flops_enc_tran = len(self.table_sizes) * self.args_dhe['k'] * 10
        dhe_flops_dec_mlps = self.dhe_mlps_flops
        dhe_flops_dec_redu = len(self.table_sizes) * self.emb_dim * (self.args_dhe['num_lookups'] - 1)

        return dhe_flops_enc_hash, dhe_flops_enc_tran, dhe_flops_dec_mlps, dhe_flops_dec_redu

    ##############################  SETUP ##############################
    # Returns the prime number after num (use this function if primesieve is not available)
    def next_prime(
        self,
        num: int,
    ):
        i = num + 1
        while True:
            isPrime = True
            for j in range(2, int(math.sqrt(i) + 1)):
                if i % j == 0:
                    isPrime = False
                    break
            if isPrime:
                return i # returns prime number larger than num
            i += 1

    # Generates encoder hash function parameters
    def generate_hash_params(
        self,
        table_sizes: List,
        args_dhe: TypedDict,
    ):
        num_tables  = len(table_sizes)
        hash_params = nn.ParameterList()

        # We have four hash function parameters to generate per hash function: m, p, a, b.

        # m and p are determinsitic
        m = args_dhe["m"]
        p = nth_prime(1, args_dhe["m"])  # p can be any prime integer larger than m.
        # p = self.next_prime(args_dhe['m'])

        fix_seed(args_dhe["seed"]) # a and b are non-deterministic (here we fix seed for reproducibility)
        
        # each embedding table is replaced with k hash functions, leading to a total of num_tables * k hash functions.
        for _ in range(num_tables):
            hash_params_table_i = []
            for _ in range(args_dhe["k"]):
                if args_dhe["hash_fn"] == "universal":
                    # Random a, b
                    a = randint(1, p - 1)
                    b = randint(0, p - 1)
                    hash_params_table_i.append([a, b, p, m])
                else:
                    sys.exit("DHE: Please use valid hash function!")
            hash_params_table_i = torch.tensor(
                hash_params_table_i
            )
            hash_params.append(nn.Parameter(hash_params_table_i, requires_grad=False))
        # hash_params can be indexed by [table #, hash function #, hash function parameter]

        if args_dhe["hash_fn"] == "universal":
            hash_params_size  = num_tables * args_dhe["k"] * 4 * 4 # the last four represents 4 bytes / int32
            hash_params_flops = num_tables * args_dhe["k"] * 4 # estimate the four operations as 4 FLOPS

        return hash_params, hash_params_size, hash_params_flops

    # Creates decoder MLP stacks
    def create_mlp(
        self,
        table_sizes: List,
        args_dhe: TypedDict,
    ):
        if args_dhe['precision'] == np.float32:
            bytes_per_param = 4
        elif args_dhe['precision'] == np.float16:
            bytes_per_param = 2
        else:
            sys.exit('Unsupported Datatype')

        mlp_dims   = args_dhe["mlp_dims"]
        num_tables = len(table_sizes)
        mlp_stacks = nn.ModuleList()

        mlp_size  = 0
        mlp_flops = 0

        # For each sparse feature, build a DHE Decoder MLP stack
        for _ in range(num_tables):
            # build MLP layer by layer
            layers = nn.ModuleList()
            for i in range(len(mlp_dims) - 1):
                dim_in  = int(mlp_dims[i])
                dim_out = int(mlp_dims[i + 1])
                layer   = nn.Linear(dim_in, dim_out, bias = True)

                mu      = 0.0
                sigma_w = np.sqrt(2 / (dim_in + dim_out))
                sigma_b = np.sqrt(1 / dim_out)
                w       = ra.normal(mu, sigma_w, size=(dim_out, dim_in)).astype(args_dhe['precision'])
                b       = ra.normal(mu, sigma_b, size=dim_out).astype(args_dhe['precision'])

                layer.weight.data = torch.tensor(w, requires_grad = True)
                layer.bias.data   = torch.tensor(b, requires_grad = True)

                layers.append(layer)

                # construct decoder MLP activation function
                if i < len(mlp_dims) - 2:
                    layers.append(nn.ReLU())
                else:
                    if args_dhe["activation"] == 'relu':
                        layers.append(nn.ReLU())
                    if args_dhe["activation"] == 'sigmoid':
                        layers.append(nn.Sigmoid())
                    elif args_dhe["activation"] == 'mish':
                        layers.append(nn.Mish())
                    else:
                        sys.exit('Unsupported DHE MLP Activation Function')
                    
                    # Batch Normalization
                    if args_dhe["batch_norm"]:
                        layers.append(nn.BatchNorm1d(num_features=dim_out))
                
                mlp_size  += (dim_in + 1) * dim_out * bytes_per_param
                mlp_flops += 2 * dim_in * dim_out

            mlp_stacks.append(torch.nn.Sequential(*layers))

        # returns a list of MLP Stacks, where mlp_stacks[i] is the decoder MLP stack for table i.
        return mlp_stacks, mlp_size, mlp_flops

    ##############################  ENCODER ##############################
    def hash(
        self,
        x: torch.Tensor,
        hash_params: torch.Tensor,
        args_dhe: TypedDict,
    ):  
    
        # Parallel Hash
        if args_dhe["hash_fn"] == "universal":
            a = hash_params[:, 0]
            b = hash_params[:, 1]
            p = hash_params[:, 2]
            m = hash_params[:, 3]
            # Outer product compute of all input IDs and hash parameter "a"
            x_hashed = torch.outer(x, a).add(b).fmod(p).fmod(m)
        else:
            sys.exit("DHE: Please use valid hash function!")

        # returns Tensor of (# lookups, # hash functions) hashed lookups
        return x_hashed

    def transform(
        self,
        x: torch.Tensor,
        args_dhe: TypedDict,
    ):
        # Option 1: Uniform Transform
        if args_dhe["transform"] == "uniform":
            x_transformed = (x / (args_dhe["m"]-1)) * 2 - 1
        # Option 2:  Box-Muller Transform (convert uniformly distributed samples to standard normal distribution)
        elif args_dhe["transform"] == 'gaussian':
            x_transformed = (x+1) / (args_dhe["m"])
            U1    = x_transformed[:,::2]
            U2    = x_transformed[:,1::2]
            R     = torch.sqrt(-2 * torch.log(U1))
            Theta = 2 * torch.tensor(math.pi) * U2
            x_transformed[:,::2]  = R * torch.cos(Theta)
            x_transformed[:,1::2] = R * torch.sin(Theta)
        else:
            sys.exit("DHE: Please use valid transform function!")

        # (for parallel execution) returns (# lookups, # hash functions) hashed lookups (post-transformation)
        return x_transformed

    ##############################  DECODER ##############################
    def reduce(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
    ):
        x_reduced = []

        # Offsets are encoded as the starting index for reduction in a multi-hot lookup
        for offset_num in range(len(offsets)):
            offset_start = offsets[offset_num]
            if offset_num == len(offsets) - 1:
                offset_end = len(x)
            else:
                offset_end = offsets[offset_num + 1]
            x_reduced.append(torch.sum(x[offset_start:offset_end], dim=0))

        # returns lookups that have been reduced in shape of (batch size, embedding dimenson)
        return torch.squeeze(torch.stack(x_reduced), 1)

    ##############################  DHE ##############################
    def encode(
        self,
        x: torch.Tensor,
        hash_params: torch.Tensor,
        args_dhe: TypedDict,
    ):
        '''
        t0 = time.time()
        # Encoder stack is composed of two parts: hash and transform
        with record_function("Encoder: Hash"):
            x_hashed = self.hash(x, hash_params, args_dhe)
        t1 = time.time()
        with record_function("Encoder: Transform"):
            x_encoded = self.transform(x_hashed, args_dhe)
        t2 = time.time()
        # print('Encode - Hash: {} ms, Transform: {} ms'.format((t1-t0)*1000, (t2-t1)*1000))
        '''
        x_hashed  = self.hash(x, hash_params, args_dhe)
        x_encoded = self.transform(x_hashed, args_dhe)

        # returns (# lookups, # hash functions) encoded lookups
        return x_encoded


    def decode(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        weights: torch.Tensor,
        dhe_l: nn.Sequential,
    ):
        '''
        t0 = time.time()
        # Decoder stack is composed of two parts: MLP and reduce
        with record_function("Decoder: MLP"):
            x_decoded = dhe_l(x)
        t1 = time.time()
        if weights is not None:
            with record_function("Decoder: Per-Sample Scaling"):
                x_decoded = x_decoded * weights[:, None]
        with record_function("Decoder: Reduce"):
            x_decoded = (
                self.reduce(x_decoded, offsets) if x_decoded.shape[0] != len(offsets) else x_decoded
            )  # Don't use reduce logic if one-hot lookup
        t2 = time.time()
        # print('Decode - MLP: {} ms, Reduce: {} ms'.format((t1-t0)*1000, (t2-t1)*1000))
        '''
        x_decoded = dhe_l(x)
        if weights is not None:
            x_decoded = x_decoded * weights[:, None]
        x_decoded = (
            self.reduce(x_decoded, offsets) if self.args_dhe['num_lookups'] > 1 else x_decoded
        )  # Don't use reduce logic if one-hot lookup

        # returns (batch size, embedding dimension) decoded lookups
        return x_decoded


    def forward(
        self,
        lS_o: torch.Tensor,
        lS_i: torch.Tensor,
        lS_w: torch.Tensor = None,
    ):
        '''
        time_encode = 0
        time_decode = 0
        ly = []  # list of resulting embeddings from each table
        # iterate through each table's required lookup IDs
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]  # find reduction offsets of table k

            if lS_w is not None:
                per_sample_weights = lS_w[k]
                assert (per_sample_weights.shape == sparse_index_group_batch.shape)
            else:
                per_sample_weights = None
                
            params_table_k = self.hash_params[k]  # load hash parameters of table k

            torch.cuda.synchronize()
            t0 = time.time()
            # encoder stack
            with record_function("Table {} Encoder".format(k)):
                sparse_index_group_batch_encoded = self.encode(
                    sparse_index_group_batch, params_table_k, self.args_dhe
                )

            torch.cuda.synchronize()
            t1 = time.time()
            # decoder stack
            with record_function("Table {} Decoder".format(k)):
                sparse_index_group_batch_decoded = self.decode(
                    sparse_index_group_batch_encoded, sparse_offset_group_batch, per_sample_weights,
                    self.dhe_mlps[k],
                )
            torch.cuda.synchronize()
            t2 = time.time()
            # append decoded embeddings of this stack to aggregate
            ly.append(sparse_index_group_batch_decoded)
            time_encode += (t1-t0)*1000
            time_decode += (t2-t1)*1000
        print('DHE Encode: {} ms, DHE Decode: {} ms'.format(time_encode, time_decode))
        '''
        ly = []  # list of resulting embeddings from each table
        # iterate through each table's required lookup IDs
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]  # find reduction offsets of table k

            if lS_w is not None:
                per_sample_weights = lS_w[k]
                assert (per_sample_weights.shape == sparse_index_group_batch.shape)
            else:
                per_sample_weights = None
                
            params_table_k = self.hash_params[k]  # load hash parameters of table k
            # encoder stack
            sparse_index_group_batch_encoded = self.encode(
                sparse_index_group_batch, params_table_k, self.args_dhe
            )
            # decoder stack
            sparse_index_group_batch_decoded = self.decode(
                sparse_index_group_batch_encoded, sparse_offset_group_batch, per_sample_weights,
                self.dhe_mlps[k],
            )
            # append decoded embeddings of this stack to aggregate
            ly.append(sparse_index_group_batch_decoded)

        # return is in a list indexed by table number, same return format as apply_emb found in dlrm_s_pytorch.py
        return ly
