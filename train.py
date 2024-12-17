"""Training for score-based generative models"""

import gc
import io
import os
import time
import logging

import numpy as np
import tensorflow as tf
import tensorflow_gan as gan
import torch

from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from data import datasets
from models.rectified_flow import RectifiedFlow
from models import sdes
from models import utils as mutils
from utils.ema import ExponentialMovingAverage
from utils import losses, sampling
from utils.checkpoint import load_checkpoint, save_checkpoint



def train(config, work_dir):
    """Runs the training pipeline

    Args:
        config: Configuation to use
        work_dir: Working directory for checkpoints and TF summaries.
            If this contains checkpoint, training will be resumed from the latest checkpoint
    """
    # create directories for exprimental logs
    sample_dir = os.path.join(work_dir, "samples")
    tf.io.gfile.makedirs(sample_dir)
    
    tb_dir = os.path.join(work_dir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)
    
    # Initialize model
    model = mutils.create_model(config)
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
    train_dataset, eval_dataset = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_dataset)
    eval_iter = iter(eval_dataset)
    
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sdes.VPSDE(
            beta_min=config.model.beta_min, 
            beta_max=config.model.beta_max, 
            N=config.model.num_scales
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sdes.subVPSDE(
            beta_min=config.model.beta_min, 
            beta_max=config.model.beta_max, 
            N=config.model.num_scales
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sdes.VESDE(
            sigma_min=config.model.sigma_min, 
            sigma_max=config.model.sigma_max, 
            N=config.model.num_scales
        )
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'rectified_flow':
        sde = RectifiedFlow(
            model=state['model'],
            init_type=config.sampling.init_type, 
            noise_scale=config.sampling.init_noise_scale, 
            use_ode_sampler=config.sampling.use_ode_sampler
        ) 
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
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
    
    # Build sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
        
    num_train_steps = config.training.n_iters
    
    # In case there are multiple hosts (e.g. TPU pods), only log to host 0
    logging.info(f"Starting training loop at step {initial_step}")
    
    for step in range(initial_step, num_train_steps + 1):
        # Convert data to Jax arrays and normalize them. Use ._numpy() to avoid copy
        batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
        batch = batch.permute(0, 3, 1, 2)
        batch = scaler(batch)
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info(f"step: {step}, training_loss: {loss.item()}.5e")
            writer.add_scaler("training_loss", loss, step)
            
        # Save a temporary checkpoint to resum training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)
        
        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
            eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info(f"step: {step}, eval_loss: {eval_loss.item()}.5e")
            writer.add_scaler("eval_loss", eval_loss.item(), step)
            
        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{save_step}.pth"), state)
            
            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(sde.model.parameters())
                ema.copy_to(sde.model.parameters())
                sample, n = sampling_fn()
                ema.restore(sde.model.parameters())
                this_sample_dir = os.path.join(sample_dir, f"iter_{step}")
                tf.io.gfile.makedirs(this_sample_dir)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                
                with tf.to.gfile.GFile(
                    os.path.join(this_sample_dir, "sample.np"), 'wb'
                ) as fout:
                    np.save(fout, sample)
                
                with tf.io.gfile.GFile(
                    os.path.join(this_sample_dir, "sample.png"), 'wb'
                ) as fout:
                    save_image(image_grid, fout)