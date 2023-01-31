import numpy as np
from numpy import random as ra

import sys

import torch
import torch.nn as nn

from models.dhebag import DHEBag

class Hybrid(nn.Module):
    
    # Create MLP layers
    def create_mlp(
        self, 
        mlp_dims,
        en_sigmoid,
        precision=np.float32
    ):
        if precision == np.float32:
            bytes_per_param = 4
        elif precision == np.float16:
            bytes_per_param = 2
        else:
            sys.exit('Unsupported Datatype')
        
        layers    = nn.ModuleList()
        mlp_size  = 0
        mlp_flops = 0
        
        for i in range(len(mlp_dims) - 1):
            dim_in  = int(mlp_dims[i])
            dim_out = int(mlp_dims[i + 1])
            layer   = nn.Linear(dim_in, dim_out, bias = True)
            
            mu      = 0.0
            sigma_w = np.sqrt(2 / (dim_in + dim_out))
            sigma_b = np.sqrt(1 / dim_out)
            w       = ra.normal(mu, sigma_w, size=(dim_out, dim_in)).astype(precision)
            b       = ra.normal(mu, sigma_b, size=dim_out).astype(precision)
            
            layer.weight.data = torch.tensor(w, requires_grad = True)
            layer.bias.data   = torch.tensor(b, requires_grad = True)

            layers.append(layer)
            
            if en_sigmoid and i == len(mlp_dims) - 2:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
                
            mlp_size  += (dim_in + 1) * dim_out * bytes_per_param
            mlp_flops += 2 * dim_in * dim_out
                
        return nn.Sequential(*layers), mlp_size, mlp_flops
            
    # Create Embedding Tables
    def create_embedding_tables(
        self,
        emb_dim,
        table_sizes,
        precision=np.float32
    ):
        if precision == np.float32:
            bytes_per_param = 4
        elif precision == np.float16:
            bytes_per_param = 2
        else:
            sys.exit('Unsupported Datatype')
        
        embedding_tables = nn.ModuleList()
        
        for size in table_sizes:
            table = nn.EmbeddingBag(size, emb_dim, mode='sum', sparse=True)
            
            w = ra.uniform(
                low  = -np.sqrt(1/size),
                high = np.sqrt(1/size),
                size = (size, emb_dim)
            ).astype(precision)
            
            table.weight.data = torch.tensor(w, requires_grad=True)
            
            embedding_tables.append(table)
        
        tables_size = emb_dim * np.sum(table_sizes) * bytes_per_param
            
        return embedding_tables, tables_size

    def create_dhe_stacks(
        self,
        table_sizes,
        args_dhe=None
    ):
        dhe_stacks = DHEBag(table_sizes, args_dhe)
        return dhe_stacks

    def apply_mlp(self, inputs, mlp):
        return mlp(inputs)
    
    def apply_embedding_tables(self, offsets, indices, tables):
        embeddings = []
        for i in range(len(indices)):
            index_group_batch  = indices[i]
            offset_group_batch = offsets[i]
            table              = tables[i]
            
            embedding = table(index_group_batch, offset_group_batch)
            embeddings.append(embedding)
            
        return embeddings

    def apply_dhe_stacks(self, offsets, indices):
        return self.dhe_stacks(offsets, indices)
    
    def __init__(
        self,
        dense_dim,
        mlp_bot_dims,
        mlp_top_dims,
        emb_dim,
        table_sizes,
        num_lookups,
        precision,
        args_dhe
    ):
        super(Hybrid, self).__init__()
        
        self.dense_dim     = dense_dim
        self.mlp_bot_dims  = mlp_bot_dims
        self.mlp_top_dims  = mlp_top_dims
        self.emb_dim       = emb_dim
        self.table_sizes   = table_sizes
        self.num_lookups   = num_lookups
        self.precision     = precision
        self.args_dhe      = args_dhe
        
        # Modify MLP Top Architecture to account for feature interaction
        self.mlp_top_dims = [emb_dim * (1 + 2 * len(table_sizes))] + self.mlp_top_dims
        
        # Make MLPs (MLP Bot, MLP Top)
        self.mlp_bot, self.mlp_bot_size, self.mlp_bot_flops = self.create_mlp(self.mlp_bot_dims, en_sigmoid=False, precision=self.precision)
        self.mlp_top, self.mlp_top_size, self.mlp_top_flops = self.create_mlp(self.mlp_top_dims, en_sigmoid=True, precision=self.precision)
        
        # Loss Function
        self.loss_fn = nn.BCELoss(reduction="mean")

        # Make Embedding Tables
        self.embedding_tables, self.tables_size = self.create_embedding_tables(self.emb_dim, self.table_sizes, self.precision)
        self.tables_flops = len(self.table_sizes) * self.emb_dim * (self.num_lookups - 1)

        # Make DHE Stacks
        self.dhe_stacks = self.create_dhe_stacks(table_sizes, self.args_dhe)

        # DHE Stacks Statistics
        self.dhe_sizes  = self.dhe_stacks.get_sizes()
        self.dhe_flops  = self.dhe_stacks.get_flops()

        self.dhe_size_enc, self.dhe_size_dec = self.dhe_sizes
        self.dhe_flops_enc_hash, self.dhe_flops_enc_tran, self.dhe_flops_dec_mlps, self.dhe_flops_dec_redu \
            = self.dhe_flops
        
        # Summary Print Messages
        self.model_size  = self.mlp_bot_size + self.mlp_top_size + self.tables_size + self.dhe_size_enc + self.dhe_size_dec
        self.model_flops = self.mlp_bot_flops + self.mlp_top_flops + self.tables_flops \
            + self.dhe_flops_enc_hash + self.dhe_flops_enc_tran + self.dhe_flops_dec_mlps + self.dhe_flops_dec_redu

        
        print('Model Statistics')
        print('DataType: {}'.format(self.precision))

        print('========== Size ==========')
        print('- MLP Bot Size: {:.3f} MB'.format(self.mlp_bot_size/1e6))
        print('- MLP Top Size: {:.3f} MB'.format(self.mlp_top_size/1e6))
        print('- Embedding Tables Size: {:.3f} MB'.format(self.tables_size/1e6))
        print('- DHE Encoder Size: {:.3f} MB'.format(self.dhe_size_enc/1e6))
        print('- DHE Decoder Size: {:.3f} MB'.format(self.dhe_size_dec/1e6))
        print('- Model Size: {:.3f} MB'.format(self.model_size/1e6))

        print('\t({:.3f}% MLP, {:.3f}% Table, {:.3f}% DHE Enc, {:.3f}% DHE Dec)' \
            .format(100*(self.mlp_bot_size + self.mlp_top_size)/self.model_size, 100*self.tables_size/self.model_size,  \
                    100*self.dhe_size_enc/self.model_size, 100*self.dhe_size_dec/self.model_size))
        
        print('========== FLOPs ==========')
        print('- MLP Bot FLOPs: {:.3f} MFLOPs'.format(self.mlp_bot_flops/1e6))
        print('- MLP Top FLOPs: {:.3f} MFLOPs'.format(self.mlp_top_flops/1e6))
        print('- Embeddings Reduction FLOPs: {:.3f} MFLOPs'.format(self.tables_flops/1e6))
        print('- DHE Encoder Hash FLOPs: {:.3f} MFLOPs'.format(self.dhe_flops_enc_hash/1e6))
        print('- DHE Encoder Transform FLOPs: {:.3f} MFLOPs'.format(self.dhe_flops_enc_tran/1e6))
        print('- DHE Decoder MLP FLOPs: {:.3f} MFLOPs'.format(self.dhe_flops_dec_mlps/1e6))
        print('- DHE Decoder Reduction FLOPs: {:.3f} MFLOPs'.format(self.dhe_flops_dec_redu/1e6))
        print('- Model FLOPs: {:.3f} MFLOPs'.format(self.model_flops/1e6))

        print('\t({:.3f}% Bot MLP, {:.3f}% Top MLP, , {:.3f}% Embeddings, {:.3f}% DHE Enc-Hash, {:.3f}% DHE Enc-Tran, {:.3f}% DHE Dec-MLPs, {:.3f}% DHE Dec-Redu)'\
            .format(100*self.mlp_bot_flops/self.model_flops, 100*self.mlp_top_flops/self.model_flops, 100*self.tables_flops/self.model_flops, \
                    100*self.dhe_flops_enc_hash/self.model_flops, 100*self.dhe_flops_enc_tran/self.model_flops, \
                    100*self.dhe_flops_dec_mlps/self.model_flops, 100*self.dhe_flops_dec_redu/self.model_flops))
        

    def forward(self, x_dense, x_offsets, x_indices):
        
        mlp_bot_out    = self.apply_mlp(x_dense, self.mlp_bot)
        embeddings_tbl = self.apply_embedding_tables(x_offsets, x_indices, self.embedding_tables)
        embeddings_dhe = self.apply_dhe_stacks(x_offsets, x_indices)
        fea_int_out    = torch.cat([mlp_bot_out] + embeddings_tbl + embeddings_dhe, dim=1)
        mlp_top_out    = self.apply_mlp(fea_int_out, self.mlp_top)
        
        return mlp_top_out