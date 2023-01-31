import numpy as np

import sys
from time import time

import poptorch

import torch

from gen_data import GeneratedDataset
from models.dlrm.dlrm_multi_ipu import DLRM_MULTI_IPU
from utils import fix_seed, multi_ipu_unpack_batch, parse_configurations, parse_dims

def main():
    # Ensure IPU Execution
    assert args.device == 'ipu'

    # Experiment Arguments
    args = parse_configurations()
    args.mlp_bot_dims = parse_dims(args.mlp_bot_dims)
    args.mlp_top_dims = parse_dims(args.mlp_top_dims)
    args.table_sizes  = parse_dims(args.table_sizes)
    args.precision    = eval(args.precision)

    # Partitioning Tables for Multi IPUs
    # Custom Sharding for Criteo Kaggle
    table_sizes_1 = args.table_sizes[:-3]
    table_sizes_2 = [args.table_sizes[-3]]
    table_sizes_3 = [args.table_sizes[-2]]
    table_sizes_4 = [args.table_sizes[-1]]

    # Fix Seed
    fix_seed(args.seed)

    # Data Generation
    dataset = GeneratedDataset(
        num_batches = args.num_batches,
        batch_size  = args.batch_size,
        dense_dim   = args.dense_dim,
        table_sizes = args.table_sizes,
        num_lookups = args.num_lookups,
        precision   = args.precision
    )

    # Model and Optimizer Setup
    constant_offsets = torch.tensor(np.arange(args.batch_size)*args.num_lookups)
    model = DLRM_MULTI_IPU(
        dense_dim        = args.dense_dim,
        mlp_bot_dims     = args.mlp_bot_dims,
        mlp_top_dims     = args.mlp_top_dims,
        emb_dim          = args.emb_dim,
        table_sizes      = args.table_sizes,
        table_sizes_1    = table_sizes_1,
        table_sizes_2    = table_sizes_2,
        table_sizes_3    = table_sizes_3,
        table_sizes_4    = table_sizes_4,
        num_lookups      = args.num_lookups,
        precision        = args.precision,
        constant_offsets = constant_offsets,
    )
    # Create Optimizer
    optimizer = poptorch.optim.SGD(model.parameters(), lr=args.lr)

    print(model)

    # Training
    if args.train == True:
        sys.exit('Multi IPU Training not implemented yet.')

    # Inference
    if args.inference == True:
        # Model Compilation
        model = model.eval()
        opts  = poptorch.Options()
        opts.deviceIterations(args.ipu_device_iterations)
        opts.replicationFactor(args.ipu_replicas)
        if args.ipu_offload:
            print('<IPU> Offloading weights w/ > {} parameters to Streaming Memory>'.format(args.offload_threshold))
            opts.TensorLocations.setWeightLocation(
                poptorch.TensorLocationSettings().minElementsForOffChip(args.offload_threshold).useOnChipStorage(False))
        opts._Popart.set("saveInitializersToFile", "my_file.onnx") # Disable ONNX limitation

        poptorch_dataloader = poptorch.DataLoader(
            opts,
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0)
        poptorch_model_inference = poptorch.inferenceModel(model,
                                                        options=opts)
        # Pre-Compile Model
        x_dense_compile, _, x_indices_compile, _ = multi_ipu_unpack_batch(next(iter(poptorch_dataloader)))

        t0 = time()
        poptorch_model_inference.compile(x_dense_compile, x_indices_compile)
        t1 = time()
        print('<IPU> Compilation Time: {:.3f} s'.format(t1-t0))

        # Inference Flow
        for epoch in range(args.num_epochs):
            print('Epoch {} ===================='.format(epoch))
            for i, inputBatch in enumerate(poptorch_dataloader):
                x_dense, _, x_indices, _ = multi_ipu_unpack_batch(inputBatch)
                output = poptorch_model_inference(x_dense, x_indices)
        poptorch_model_inference.detachFromDevice()

if __name__ == '__main__':
    main()