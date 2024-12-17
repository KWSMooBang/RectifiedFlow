import os
import gc
import io
import time
import logging

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import torch


from absl import flags
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from models import ddpm, ncsnv2
from models import sdes
from models.rectified_flow import RectifiedFlow
from models import utils as mutils
from utils import losses, likelihood, evaluation, sampling
from utils.ema import ExponentialMovingAverage
from utils.checkpoint import save_checkpoint, load_checkpoint
from data import datasets

FLAGS = flags.FLAGS


def finetune_reflow(config, work_dir):
    """Runs the rematching finetune pipeline
    
    Args:
        config: Configuration to use
        work_dir: Working directory for checkpoints and TF summaries.
            If this contains checkpoint training will be resumed from the latest checkpoint
    """
    
    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    tf.io.gfile.makedirs(sample_dir)
    
    tb_dir = os.path.join(work_dir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)
    
    # Initialize model
    model = mutils.create_model(config)
    
    # Load pre-trained checkpoint
    checkpoint_dir = config.reflow.last_flow_checkpoint
    loaded_state = torch.load(checkpoint_dir, map_location=config.device)
    model.load_state_dict(loaded_state['model'], strict=False)
    loaded_ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    loaded_ema.load_state_dict(loaded_state['ema'])
    ema_model = mutils.create_model(config)
    loaded_ema.copy_to(ema_model.parameters())
    print("Loaded:", checkpoint_dir, "Step:", loaded_state['step'])
    
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
    
    # Create checkpoints directory
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = load_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    
    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)
    
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sdes.VPSDE(
            model=state['model'], 
            beta_min=config.model.beta_min, 
            beta_max=config.model.beta_max, 
            N=config.model.num_scales
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sdes.subVPSDE(
            model=state['model'],
            beta_min=config.model.beta_min, 
            beta_max=config.model.beta_max, 
            N=config.model.num_scales
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sdes.VESDE(
            model=state['model'],
            sigma_min=config.model.sigma_min, 
            sigma_max=config.model.sigma_max, 
            N=config.model.num_scale
        )
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'rectified_flow':
        sde = RectifiedFlow(
            model=state['model'], 
            init_type=config.sampling.init_type, 
            noise_scale=config.sampling.init_noise_scale, 
            reflow_flag=True,
            reflow_t_schedule=config.reflow.reflow_t_schedule,
            reflow_loss=config.reflow.reflow_loss,
            use_ode_sampler=config.sampling.use_ode_sampler
        )
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown")
    
    # Build one-step training and evaluation functions
    optimize_fn = losses.get_optimize_fn(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        sde, train=True, optimize_fn=optimize_fn,
        reduce_mean=reduce_mean, continuous=continuous,
        likelihood_weighting=likelihood_weighting
    )
    eval_step_fn = losses.get_step_fn(
        sde, train=False, optimize_fn=optimize_fn,
        reduce_mean=reduce_mean, continuous=continuous,
        likelihood_weighting=likelihood_weighting
    )
    
    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (
            config.training.batch_size,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size
        )
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
        
    num_train_steps = config.training.n_iters
    
    data_root = config.reflow.data_root
    print("Data Path:", data_root)
    print("t schedule:", config.reflow.reflow_t_schedule, "Loss:", config.reflow.reflow_loss)
    if config.reflow.reflow_type == 'generate_data_from_z0':
        # Note: prepare reflow dataset with ODE
        print("Start generating data with ODE from z0", ", seed:", config.seed)
        
        loaded_ema.copy_to(model.paramters())
        data_collection = []
        z0_collection = []
        for data_step in range(config.reflow.total_number_of_samples // config.training.batch_size):
            print(data_step)
            z0 = sde.get_z0(torch.zeros((config.training.batch_size, 3, config.data.image_size, config.data.image_size), device=config.device), train=False).to(config.device)
            batch = sde.ode(z0)
            
            print(batch.shape, batch.max(), batch.min(), z0.mean(), z0.std())
            
            z0_collection.append(z0.cpu())
            data_collection.append(batch.cpu())
            
        z0_collection = torch.cat(z0_collection)
        data_collection = torch.cat(data_collection)
        print(data_collection.shape, z0_collection.shape)
        print(z0_collection.mean(), z0_collection.std())
        if not os.path.exists(os.path.join(data_root, str(config.seed))):
            os.mkdir(os.path.join(data_root, str(config.seed)))
        np.save(os.path.join(data_root, str(config.seed), 'z1.npy'), data_collection.numpy())
        np.save(os.path.join(data_root, str(config.seed), 'z0.npy'), z0_collection.numpy())
        
        import sys
        print("Successfully generated z1 from random z0 with random seed:", config.seed, "Total number of pairs:", (data_step+1)*config.training.batch_size)
        sys.exit(0)
    elif config.reflow.reflow_type == 'train_reflow':
        # Note: load existing dataset
        print("Start training with (Z0, Z1) pair")
        
        z0_collection = []
        data_collection = []
        folder_list = os.listdir(data_root)
        for folder in folder_list:
            print("Folder:", folder)
            z0 = np.load(os.path.join(data_root, folder, 'z0.npy'))
            print("Loaded z0")
            data = np.load(os.path.join(data_root, folder, 'z1.npy'))
            print("Loaded z1")
            z0 = torch.from_numpy(z0).cpu()
            data = torch.from_numpy(data).cpu()
            
            z0_collection.append(z0)
            data_collection.append(data)
            print("z0 shape:", z0.shape, "z0 min:", z0.min(), "z0 max:", z0.max())
            print("z1 shape:", data.sahpe, "z1 min:", data.min(), "z1 max:", data.max())
            
        print("Successfully loaded (z0, z1) pairs")
        z0_collection = torch.cat(z0_collection)
        data_collection = torch.cat(data_collection)
        print("Shape of z0:", z0_collection.shape, "Shape of z1:", data_collection.shape)
        
    elif config.reflow.reflow_type == 'train_online_reflow':
        pass
    else:
        raise NotImplementedError(f"Reflow type {config.reflow.reflow_type} is not implemented")
    
    print("Initial step of the model:", initial_step)
    # In case there are multiple hosts (e.g., TPU pods), only log to host0
    logging.info(f"Starting reflow training loop at step {initial_step}")
    
    for step in range(initial_step, num_train_steps + 1):
        # Conver data to Jax arrays and normalize them. Use ._numpy() to avoid copy
        if config.reflow.reflow_type == 'train_reflow':
            indices = torch.randperm(len(data_collection))[:config.training.batch_size]
            data = data_collection[indices].to(config.device).float()
            z0 = z0_collection[indices].to(config.device).float()
        elif config.reflow.reflow_type == 'train_online_reflow':
            z0 = sde.get_z0(torch.zeros((config.training.batch_size, 3, config.data.image_size, config.data.image_size), device=config.device), train=False).to(config.device)
            data = sde.euler_ode(z0, N=20)
            z0 = z0.to(config.device).float()
            data = data.to(config.device).float()
            
        batch = [z0, data]
        
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info(f"step: {step}, training_loss: {loss.item():.5e}")
            writer.add_scalar("training_loss", loss, step)
            
        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)
            
        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            if config.reflow.reflow_type == 'train_reflow':
                indices = torch.randperm(len(data_collection))[:config.training.batch_size]
                data = data_collection[indices].to(config.device).float()
                z0 = z0_collection[indices].to(config.device).float()
            elif config.reflow.reflow_type == 'train_online_reflow':
                z0 = sde.get_z0(torch.zeros((config.training.batch_size, 3, config.data.image_size, config.data.image_size), device=config.device), train=False).to(config.device)
                data = sde.euler_ode(z0, ema_model, N=20)
                z0 = z0.to(config.device).float()
                data = data.to(config.device).float()
                
            eval_batch = [z0, data]
            
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info(f"step: {step}, eval_loss: {eval_loss.item():.5e}")
            writer.add_scalar("eval_loss", eval_loss.item(), step)
            
        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{save_step}.pth"), state)
            
            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                sample, n = sampling_fn()
                ema.restore(model.parameters())
                this_sampler_dir = os.path.join(sample_dir, f"iter_{step}")
                tf.io.gfile.makedirs(this_sampler_dir)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                with tf.io.gfile.GFile(
                    os.path.join(this_sampler_dir, "sample.np"), 'wb'
                ) as fout:
                    np.save(fout, sample)
                
                with tf.io.gfile.GFile(
                    os.path.join(this_sampler_dir, "sample.png"), 'wb'
                ) as fout:
                    save_image(image_grid, fout)