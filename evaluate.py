import gc
import io
import os
import time
import logging
import numpy as np
import torch
import tensorflow as tf
import tensorflow_gan as tfgan

from data import datasets
from models import utils as mutils
from models import sdes
from models.rectified_flow import RectifiedFlow
from utils import losses, likelihood, evaluation, sampling
from utils.ema import ExponentialMovingAverage
from utils.checkpoint import load_checkpoint, save_checkpoint


def evaluate(config, work_dir, eval_folder="eval"):
    """Evaluate trained models

    Args:
        config: Configuration to use.
        work_dir: Working directory for checkpoints
        eval_folder: The subfolder for storing evaluation results. Default to 'eval'
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(work_dir, eval_folder)
    tf.io.gfile.makedirs(eval_dir)
    
    # Build data pipeline
    train_ds, eval_ds, _ = datasets.get_dataset(config, 
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)
    
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    # Initialize model
    model = mutils.create_model(config)
    optimizer = losses.get_optimize(config, model.parameters())
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
    
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    
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
            N=config.model.num_scales
        )
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'rectified_flow':
        sde = RectifiedFlow(
            model=state['model'], 
            init_type=config.sampling.init_type, 
            noise_scale=config.sampling.init_noise_scale,
            use_ode_sampler=config.sampling.use_ode_sampler, 
            sigma_var=config.sampling.sigma_variance,
            sde_tol=config.sampling.ode_tol, 
            sample_N=config.sampling.sample_N
        )
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.get_optimize_fn(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting
        
        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous, likelihood_weighting=likelihood_weighting)
    
    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config, uniform_dequantization=True, evaluation=True)
    
    if config.eval.bqd_dataset.lower() == 'train':
        ds_bqd = train_ds_bpd
        bpd_num_repeats = 1
    elif config.eval.bqd_dataset.lower() == 'test':
        # Go over the dataset 5 times when computing likelihood on the test dataset
        ds_bpd = eval_ds_bpd
        bpd_num_repeats = 5
    else:
        raise ValueError(f"No bpd dataset {config.eval.bqd_dataset} recognized")
    
    # Build the likelihood computation function when likelihood is enabled
    if config.eval.enable_bpd:
        if config.training.sde.lower() == 'mixup':
            likelihood_fn = likelihood.get_likelihood_fn_rf(sde, inverse_scaler)
        else:
            likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)
            
    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (
            config.eval.batch_size,
            config.data.num_channels,
            config.data.image_size, config.data.image_size
        )
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
        
    # Use inceptionV3 for images with resolution higher than 256
    inceptionv3 = config.data.image_size >= 256
    inception_model = None
    
    begin_checkpoint = config.data.image_size >= 256
    logging.info(f"begin checkpoint: {begin_checkpoint}")
    for checkpoint in range(begin_checkpoint, config.eval.end_checkpoint + 1):
        # Wait if the target checkpint doesn't exist yet
        waiting_message_printed = False
        checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint}.pth")
        while not tf.io.gfile.exists(checkpoint_filename):
            if not waiting_message_printed:
                logging.warning(f"Waiting for the arrival of checkpoint_{checkpoint}")
                waiting_message_printed = True
            time.sleep(60)
            
    # Wait for 2 additional mins in caes the file exists but is not ready for reading
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint}.pth")
    try:
        state = load_checkpoint(checkpoint_path, state, device=config.device)
    except:
        time.sleep(60)
        try: 
            state = load_checkpoint(checkpoint_path, state, device=config.device)
        except:
            time.sleep(120)
            state = load_checkpoint(checkpoint_path, state, device=config.device)
    ema.copy_to(sde.model.parameters())
    
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
        all_losses = []
        eval_iter = iter(eval_ds)
        for i, batch in enumerate(eval_iter):
            eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
            eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step(state, eval_batch)
            all_losses.append(eval_loss.item())
            if (i + 1) % 1000 == 0:
                logging.info(f"Finished {i+1}th step loss evaluation")
        
        all_losses = np.asarray(all_losses)
        with tf.io.gfile.GFile(os.path.join(eval_dir, f"checkpoint_{checkpoint}_loss.npz"), 'wb') as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
            fout.write(io_buffer.getvalue())
            
    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
        bpds = []
        for repeat in range(bpd_num_repeats):
            bpd_iter = iter(ds_bpd)
            for batch_id in range(len(ds_bpd)):
                batch = next(bpd_iter)
                eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
                eval_batch = eval_batch.permute(0, 3, 1, 2)
                eval_batch = scaler(eval_batch)
                bpd = likelihood_fn(sde.model, eval_batch)[0]
                bpd = bpd.detach().cpu().numpy().reshape(-1)
                bpds.extend(bpd)
                logging.info(
                    f"checkpoint: {checkpoint}, repeat: {repeat}, batch: {batch_id}, mean bpd: {np.mean(np.asarray(bpds)):6f}"
                )
                bpd_round_id = batch_id + len(ds_bpd) * repeat
                # Save bits/dim to disk or Google Cloud Storage
                with tf.io.gfile.GFile(os.path.join(eval_dir, f"{config.eval.bpd_dataset}_checkpint_{checkpoint}_bpd_{bpd_round_id}.npz"), 'wb') as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, bpd)
                    fout.write(io_buffer.getvalue())
                    
    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
        for round in range(num_sampling_rounds):
            logging.info(f"sampling -- checkpoint: {checkpoint}, round: {round}")
            
            # Directory to save samples. Different for each host to avoid writing conflicts
            this_sample_dir = os.path.join(
                eval_dir, f"checkpoint_{checkpoint}"
            )
            tf.io.gfile.makedirs(this_sample_dir)
            samples, n = sampling_fn()
            samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            samples = samples.reshape(
                (-1, config.data.image_size, config.data.image_size, config.data.num_channels)
            )            
            # Write samples to disk or Google Cloud Storage
            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, f"samples_{round}.npz"), 'wb'
            ) as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, samples=samples)
                fout.write(io_buffer.getvalue())
              
            # Force garbage collection before calling TensorFlow code for Inception network  
            gc.collect()
            latents = evaluation.run_inception_distributed(samples, inception_model, inceptionv3=inceptionv3)
            
            # Force garbage collection again before returning to JAX code
            gc.collect()
            # Save latent represents of teh Inception network to disk or Google Cloud Storage
            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, f"statistics_{round}.npz"), 'wb'
            ) as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(
                    io_buffer, pool_3=latents['pool_3'], logits=latents['logits']
                )
                fout.write(io_buffer.getvalue())
                
        # Compute inception scores, FIDs and KIDs
        # Load all statistics that have been previously computed and saved for each host
        all_logits = []
        all_pools = []
        this_sample_dir = os.path.join(eval_dir, f"checkpoint_{checkpoint}")
        stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
        for stat_file in stats:
            with tf.io.gfile.GFile(stat_file, 'rb') as fin:
                stat = np.load(fin)
                if not inceptionv3:
                    all_logits.append(stat['logits'])
                all_pools.append(stat['pool_3'])
                
        if not inceptionv3:
            all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
        all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]
        
        # Load pre-computed dataset statistics
        data_stats = evalu
    