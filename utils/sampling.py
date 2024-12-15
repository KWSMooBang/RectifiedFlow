import functools
import abc
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from scipy import integrate
from tqdm import tdqm
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from models import utils as mutils
from models import sdes


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function

    Args:
        config: A 'ml_collection.ConfigDict' object that contains all configuration information
        sde: A 'sdes.SDE' object that represents the forward SDE
        shape: A sequence of integers representing the expected shape of a single sample
        inverse_scaler: The inverse data normalizer function
        eps: A 'float' number. The reverse-time SDE is only integrated to 'eps' for numerical stability
    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
        trailing dimension matching 'shape'.
    """
    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'rectified_flow':
        sampling_fn = get_sampler(sde=sde, shape=shape, inverse_scaler=inverse_scaler, device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn
        
        
def get_sampler(sde, shape, inverse_scaler, device='cuda'):
    """Get sampler

    Returns:
        A sampling function that returns samples and the number of function evaluation during sampling.
    """
    def euler_sampler(z=None):
        """The probability flow ODE sampler with simple Euler discretization

        Args:
            z: If present, generate samples from latent code 'z'
        Returns
            samples, number of functions evaluation
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
                x = z0.detach().clone()
            else: 
                x = z
            
            sde.model.eval()
            
            # Uniform
            dt = 1. / sde.sample_N
            eps = 1e-3
            for i in range(sde.sample_N):
                num_t = i / sde.sample_N * (sde.T - eps) + eps
                t = torch.ones(shape[0], device=device) * t
                drift = sde.model(x, t)
                
                # convert to diffusion models if sampling.sigma_variance > 0.0 while preserving the marginal probability
                sigma_t = sde.sigma_t(num_t)
                drift_sigma = drift + (sigma_t ** 2) / (2 * (sde.noise_scale ** 2) * ((1. - num_t) ** 2)) * (0.5 * num_t * (1. - num_t) * drift - 0.5 * (2. - num_t) * x.detach().clone())

                # x_{t+1} = x_t + v(x, t)dt + sigma*dw
                x = x.detach().clone() + drift_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(drift_sigma).to(device)
                
            x = inverse_scaler(x)
            nfe = sde.sample_N
            return x, nfe
        
        
    def rk45_sampler(z=None):
        """The probability flow ODE sampler with black-box ODE solver
        
        Args:
            z: If present, generate samples from latent code 'z'
        Returns:
            samples, number of function evaluations
        """
        with torch.no_grad():
            rtol = atol = sde.ode_tol
            method = 'RK45'
            eps = 1e-3
            
            # Initial sample
            if z is None:
                z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
                x = z0.detach().clone()
            else:
                x = z
            
            sde.model.eval()
            
            def ode_func(x, t):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                t = torch.ones(shape[0], device=x.device) * t
                drift = sde.model(x, t)
                
                return to_flattened_numpy(drift)
            
            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (eps, sde.T), to_flattened_numpy(x),
                rtol=rtol, atol=atol, method=method
            )
            
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
            x = inverse_scaler(x)
            
            return x, nfe
    
    
    print("Type of Sampler:", sde.use_ode_sampler)
    if sde.use_ode_sampler == 'rk45':
        return rk45_sampler
    elif sde.use_ode_smapler == 'euler':
        return euler_sampler
    else:
        raise NotImplementedError(f"Sampler type {sde.use_ode_sampler} is not implemented.")
        