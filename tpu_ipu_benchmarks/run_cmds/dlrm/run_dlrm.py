import numpy as np

import sys
from time import time

import torch

from gen_data import GeneratedDataset, make_dataloader
from utils import fix_seed, ipu_unpack_batch, parse_configurations, parse_dims

def main():
    # Experiment Arguments
    args = parse_configurations()
    args.mlp_bot_dims = parse_dims(args.mlp_bot_dims)
    args.mlp_top_dims = parse_dims(args.mlp_top_dims)
    args.table_sizes  = parse_dims(args.table_sizes)
    args.precision    = eval(args.precision)

    # IPU Imports
    if args.device == 'ipu':
        from models.dlrm.dlrm_ipu import DLRM_IPU
        import poptorch
        print('Using IPU...')
    elif args.device == 'ipu_manual':
        from models.dlrm.dlrm_ipu_manual import DLRM_IPU_MANUAL_BAG
        import poptorch
        print('Using IPU (Manual)...')

    # TPU Imports
    elif args.device == 'tpu':
        from models.dlrm.dlrm_tpu import DLRM_TPU
        import torch_xla.core.xla_model as xm
        if args.tpu_debug:
            import torch_xla.debug.metrics as met
        device = xm.xla_device()
        print('Using TPU...')

    # Default: CPU, GPU
    if args.device == 'cpu':
        device = 'cpu'
        print('Using CPU...')
    elif args.device == 'gpu' and torch.cuda.is_available():
        device = 'cuda:0'
        print('Using GPU...')
    from models.dlrm.dlrm import DLRM
        
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

    dataloader = make_dataloader(dataset)

    # Model and Optimizer Setup
    if args.device == 'cpu' or args.device == 'gpu':
        model = DLRM(
            dense_dim    = args.dense_dim,
            mlp_bot_dims = args.mlp_bot_dims,
            mlp_top_dims = args.mlp_top_dims,
            emb_dim      = args.emb_dim,
            table_sizes  = args.table_sizes,
            num_lookups  = args.num_lookups,
            precision    = args.precision
        )
        model.to(device)
        # Create Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.device == 'ipu':
        constant_offsets = torch.tensor(np.arange(args.batch_size)*args.num_lookups)
        model = DLRM_IPU(
            dense_dim        = args.dense_dim,
            mlp_bot_dims     = args.mlp_bot_dims,
            mlp_top_dims     = args.mlp_top_dims,
            emb_dim          = args.emb_dim,
            table_sizes      = args.table_sizes,
            num_lookups      = args.num_lookups,
            precision        = args.precision,
            constant_offsets = constant_offsets
        )
        # Create Optimizer
        optimizer = poptorch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.device == 'ipu_manual':
        constant_offsets = torch.tensor(np.arange(args.batch_size)*args.num_lookups)
        model = DLRM_IPU_MANUAL_BAG(
            dense_dim        = args.dense_dim,
            mlp_bot_dims     = args.mlp_bot_dims,
            mlp_top_dims     = args.mlp_top_dims,
            emb_dim          = args.emb_dim,
            table_sizes      = args.table_sizes,
            num_lookups      = args.num_lookups,
            precision        = args.precision,
            constant_offsets = constant_offsets
        )
        # Create Optimizer
        optimizer = poptorch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.device == 'tpu':
        model = DLRM_TPU(
            dense_dim    = args.dense_dim,
            mlp_bot_dims = args.mlp_bot_dims,
            mlp_top_dims = args.mlp_top_dims,
            emb_dim      = args.emb_dim,
            table_sizes  = args.table_sizes,
            num_lookups  = args.num_lookups,
            precision    = args.precision
        )
        model.to(device)
        # Create Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    print(model)

    # Training
    if args.train == True:
        if args.device == 'cpu' or args.device == 'gpu':
            model.train()
            timing_vector     = []
            timing_vector_fwd = []
            timing_vector_bwd = []
            for epoch in range(args.num_epochs):
                print('Epoch {} ===================='.format(epoch))
                for i, inputBatch in enumerate(dataloader):
                    x_dense, x_offsets, x_indices, y = inputBatch
                    t0      = time()
                    outputs = model(x_dense.to(device), x_offsets.to(device), x_indices.to(device))
                    t1      = time()
                    loss    = model.loss_fn(outputs, y.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    t2 = time()
                    timing_vector.append(1000*(t2 - t0))
                    timing_vector_fwd.append(1000*(t1 - t0))
                    timing_vector_bwd.append(1000*(t2 - t1))
                    if (i%args.print_freq == 0):
                        print('Epoch {} Batch {}: {:.3f} ({:.3f}, {:.3f}) ms'.format(
                            epoch, i, timing_vector[-1], timing_vector_fwd[-1], timing_vector_bwd[-1]))
            print('\nTotal Training Time (Stable): {:.3f} ms/it'.format(np.median(timing_vector)*len(timing_vector)))
            print('Total Training Time: {:.3f} ms/it'.format(np.sum(timing_vector)))
            print('Median Training Time: {:.3f} ms/it'.format(np.median(timing_vector)))
            print('Mean Training Time: {:.3f} ms/it\n'.format(np.mean(timing_vector)))

            print('\nTotal FWD Time (Stable): {:.3f} ms/it'.format(np.median(timing_vector_fwd)*len(timing_vector_fwd)))
            print('Total FWD Time: {:.3f} ms/it'.format(np.sum(timing_vector_fwd)))
            print('Median FWD Time: {:.3f} ms/it'.format(np.median(timing_vector_fwd)))
            print('Mean FWD Time: {:.3f} ms/it\n'.format(np.mean(timing_vector_fwd)))

            print('\nTotal BWD Time (Stable): {:.3f} ms/it'.format(np.median(timing_vector_bwd)*len(timing_vector_bwd)))
            print('Total BWD Time: {:.3f} ms/it'.format(np.sum(timing_vector_bwd)))
            print('Median BWD Time: {:.3f} ms/it'.format(np.median(timing_vector_bwd)))
            print('Mean BWD Time: {:.3f} ms/it\n'.format(np.mean(timing_vector_bwd)))

        elif args.device == 'ipu' or args.device == 'ipu_manual':
            # Model Compilation
            model.train()
            opts = poptorch.Options()
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
            poptorch_model_training = poptorch.trainingModel(model, 
                                                            options=opts, 
                                                            optimizer=optimizer)

            # Pre-Compile Model
            x_dense_compile, _, x_indices_compile, y_compile = ipu_unpack_batch(next(iter(poptorch_dataloader)))
            t0 = time()
            poptorch_model_training.compile(x_dense_compile, x_indices_compile, y_compile)
            t1 = time()
            print('<IPU> Compilation Time: {:.3f} s'.format(t1-t0))

            # Training Flow
            timing_vector_ipu = []
            for epoch in range(args.num_epochs):
                print('Epoch {} ===================='.format(epoch))
                for i, inputBatch in enumerate(poptorch_dataloader):
                    x_dense, _, x_indices, y = ipu_unpack_batch(inputBatch)
                    t0           = time()
                    output, loss = poptorch_model_training(x_dense, x_indices, y)
                    t1           = time()
                    timing_vector_ipu.append(1000*(t1-t0))
                    if (i%args.print_freq == 0):
                        print('Epoch {} Batch {}: {:.3f} ms'.format(epoch, i, timing_vector_ipu[-1]))
            print('\nTotal Training Time (Stable): {:.3f} ms/it'.format(np.median(timing_vector_ipu)*len(timing_vector_ipu)))
            print('Total Training Time: {:.3f} ms/it'.format(np.sum(timing_vector_ipu)))
            print('Median Training Time: {:.3f} ms/it'.format(np.median(timing_vector_ipu)))
            print('Mean Training Time: {:.3f} ms/it\n'.format(np.mean(timing_vector_ipu)))
            poptorch_model_training.detachFromDevice()

        elif args.device == 'tpu':
            sys.exit('TPU Training not implemented yet.')

    # Inference
    if args.inference == True:
        if args.device == 'cpu' or args.device == 'gpu':
            model.eval()
            timing_vector = []
            for epoch in range(args.num_epochs):
                print('Epoch {} ===================='.format(epoch))
                for i, inputBatch in enumerate(dataloader):
                    x_dense, x_offsets, x_indices, _ = inputBatch
                    t0      = time()
                    outputs = model(x_dense.to(device), x_offsets.to(device), x_indices.to(device))
                    t1      = time()
                    timing_vector.append(1000*(t1 - t0))
                    if (i%args.print_freq == 0):
                        print('Epoch {} Batch {}: {:.3f} ms'.format(epoch, i, timing_vector[-1]))
            print('\nTotal Inference Time (Stable): {:.3f} ms/it'.format(np.median(timing_vector)*len(timing_vector)))
            print('Total Inference Time: {:.3f} ms/it'.format(np.sum(timing_vector)))
            print('Median Inference Time: {:.3f} ms/it'.format(np.median(timing_vector)))
            print('Mean Inference Time: {:.3f} ms/it\n'.format(np.mean(timing_vector)))

        elif args.device == 'ipu' or args.device == 'ipu_manual':
            # Model Compilation
            model = model.eval()
            opts  = poptorch.Options()
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
            x_dense_compile, _, x_indices_compile, _ = ipu_unpack_batch(next(iter(poptorch_dataloader)))

            t0 = time()
            poptorch_model_inference.compile(x_dense_compile, x_indices_compile)
            t1 = time()
            print('<IPU> Compilation Time: {:.3f} s'.format(t1-t0))

            # Inference Flow
            timing_vector_ipu = []
            for epoch in range(args.num_epochs):
                print('Epoch {} ===================='.format(epoch))
                for i, inputBatch in enumerate(poptorch_dataloader):
                    x_dense, _, x_indices, _ = ipu_unpack_batch(inputBatch)
                    t0     = time()
                    output = poptorch_model_inference(x_dense, x_indices)
                    t1     = time()
                    timing_vector_ipu.append(1000*(t1-t0))
                    if (i%args.print_freq == 0):
                        print('Epoch {} Batch {}: {:.3f} ms'.format(epoch, i, timing_vector_ipu[-1]))
            print('\nTotal Inference Time (Stable): {:.3f} ms/it'.format(np.median(timing_vector_ipu)*len(timing_vector_ipu)))
            print('Total Inference Time: {:.3f} ms/it'.format(np.sum(timing_vector_ipu)))
            print('Median Inference Time: {:.3f} ms/it'.format(np.median(timing_vector_ipu)))
            print('Mean Inference Time: {:.3f} ms/it\n'.format(np.mean(timing_vector_ipu)))
            poptorch_model_inference.detachFromDevice()

        elif args.device == 'tpu':
            model.eval()
            timing_vector_tpu = []
            for epoch in range(args.num_epochs):
                print('Epoch {} ===================='.format(epoch))
                for i, inputBatch in enumerate(dataloader):
                    x_dense, x_offsets, x_indices, _ = inputBatch
                    t0      = time()
                    outputs = model(x_dense.to(device), x_offsets.to(device), x_indices.to(device))
                    t1      = time()
                    timing_vector_tpu.append(1000*(t1 - t0))
                    if (i%args.print_freq == 0):
                        print('Epoch {} Batch {}: {:.3f} ms'.format(epoch, i, timing_vector_tpu[-1]))
                if args.tpu_debug:
                    print(met.metrics_report())
            print('\nTotal Inference Time (Stable): {:.3f} ms/it'.format(np.median(timing_vector_tpu)*len(timing_vector_tpu)))
            print('Total Inference Time: {:.3f} ms/it'.format(np.sum(timing_vector_tpu)))
            print('Median Inference Time: {:.3f} ms/it'.format(np.median(timing_vector_tpu)))
            print('Mean Inference Time: {:.3f} ms/it\n'.format(np.mean(timing_vector_tpu)))

if __name__ == '__main__':
    main()