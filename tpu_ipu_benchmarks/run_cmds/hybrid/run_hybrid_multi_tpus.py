import numpy as np

import sys
from time import time

import torch

 # Import PyTorch/XLA
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp 

from gen_data import GeneratedDataset, make_dataloader
from models.hybrid.hybrid_tpu import Hybrid_TPU
from utils import fix_seed, parse_configurations, parse_dims

def _mp_fn_inference(index, args, args_dhe):

    list_of_ordinals = np.arange(args.num_tpus)
    if xm.get_local_ordinal() in list_of_ordinals:
        # Fix Seed
        fix_seed(args.seed)
        
        device      = xm.xla_device()
        device_name = xm.xla_real_devices([str(device)])[0]

        # Acquiring TPU
        print('Process {} has acquired {}'.format(index, device_name))

        xm.rendezvous('TPU Acquisiton Complete', replicas=list_of_ordinals)
        if xm.is_master_ordinal():
            print('========================================')

        # Data Generation
        dataset = GeneratedDataset(
            num_batches = args.num_batches,
            batch_size  = args.batch_size,
            dense_dim   = args.dense_dim,
            table_sizes = args.table_sizes,
            num_lookups = args.num_lookups,
            precision   = args.precision
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=args.num_tpus, # xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False
        )
        dataloader = make_dataloader(dataset, sampler)

        # Model and Optimizer Setup
        model = Hybrid_TPU(
            dense_dim    = args.dense_dim,
            mlp_bot_dims = args.mlp_bot_dims,
            mlp_top_dims = args.mlp_top_dims,
            emb_dim      = args.emb_dim,
            table_sizes  = args.table_sizes,
            num_lookups  = args.num_lookups,
            precision    = args.precision,
            args_dhe     = args_dhe
        )
        model.to(device)
        print('Process {} successfully created Hybrid on {}'.format(index, device_name))

        xm.rendezvous('Hybrid Creation Complete', replicas=list_of_ordinals)
        if xm.is_master_ordinal():
            print('========================================')

        model.eval()
        xm.rendezvous('Inference Start', replicas=list_of_ordinals)

        if xm.is_master_ordinal():
            with torch.no_grad():
                for epoch in range(args.num_epochs):
                    parallel_loader = pl.ParallelLoader(dataloader, [device]).per_device_loader(device)
                    print('Process {} starting epoch {}'.format(index, epoch))
                    for i, inputBatch in enumerate(parallel_loader):
                        x_dense, x_offsets, x_indices, _ = inputBatch
                        outputs = model(x_dense.to(device), x_offsets.to(device), x_indices.to(device))
            if args.tpu_debug:
                    print(met.metrics_report())

        else:
            with torch.no_grad():
                for epoch in range(args.num_epochs):
                    parallel_loader = pl.ParallelLoader(dataloader, [device]).per_device_loader(device)
                    print('*** Process {} starting epoch {}'.format(index, epoch))
                    for i, inputBatch in enumerate(parallel_loader):
                        x_dense, x_offsets, x_indices, _ = inputBatch
                        outputs = model(x_dense.to(device), x_offsets.to(device), x_indices.to(device))

        xm.rendezvous('Inference End', replicas=list_of_ordinals)
                            
        # Barrier to prevent master from exiting before workers finish.
    xm.rendezvous('finish')

def main():
    # Ensure TPU Execution
    assert args.device == 'tpu'

    # Experiment Arguments
    args = parse_configurations()
    args.mlp_bot_dims = parse_dims(args.mlp_bot_dims)
    args.mlp_top_dims = parse_dims(args.mlp_top_dims)
    args.table_sizes  = parse_dims(args.table_sizes)
    args.precision    = eval(args.precision)

    print('-------------------- Enabling Hybrid --------------------')
    # DHE Parameters
    args_dhe = {}
    args_dhe['activation']  = args.dhe_activation
    args_dhe['batch_norm']  = args.dhe_batch_norm
    args_dhe['hash_fn']     = args.dhe_hash_fn
    args_dhe['k']           = args.dhe_k
    args_dhe['m']           = args.dhe_m
    args_dhe['mlp_dims']    = parse_dims(args.dhe_mlp_dims)
    args_dhe['num_lookups'] = args.num_lookups
    args_dhe['precision']   = args.precision
    args_dhe['seed']        = args.seed
    args_dhe['transform']   = args.dhe_transform
    
    print('---------- DHE Parameters: ----------')
    for key, val in args_dhe.items():
        print('{} : {}'.format(key, val))
    print('--------------------')

    # Training
    if args.train == True:
        sys.exit('Multi TPU Training not implemented yet.')
            
    # Inference
    if args.inference == True:
        xmp.spawn(_mp_fn_inference, args=(args, args_dhe,), nprocs=8, start_method='spawn')

if __name__ == '__main__':
    main()