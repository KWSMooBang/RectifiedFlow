import numpy as np
import torch
import sdes

from torch import nn


_MODELS = {}


def register_model(cls=None, *, name=None):
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls
    
    if cls is None:
        return _register
    else:
        return _register(cls)
    

def get_model(name):
    return _MODELS[name]


def create_model(config):
    """
    Create the score model
    """
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)
    
    num_params = 0
    for p in score_model.parametere():
        num_params += p.numel()
    print(f"Number of parameters in the score model: {num_params}")
    
    score_model = nn.DataParallel(score_model)
    return score_model


def get_sigmas(config):
    """
    Get sigmas(the set of noise levels for SMLD from config files)
    Args:
        config: A ConfigDict object parsed from the config file
    Returns:
        sigmas: a jax numpy array of noise levels
    """
    sigmas = np.exp(
        np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales)
    )
    
    return sigmas


def get_ddpm_params(config):
    """
    Get betas and alphas(parameters used in the original DDPM)
    """
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
        'beta_min': beta_start * (num_diffusion_timesteps - 1),
        'beta_max': beta_end * (num_diffusion_timesteps - 1),
        'num_diffusion_timesteps': num_diffusion_timesteps
    }


def get_score_fn(sde, model, train=False, continuous=False):
    """
    Wraps 'score_fn' so that the model output corresponds to a real time-dependent score function
    Args:
        sde: An 'sde_lib.SDE' object that represents the forward SDE
        model: score model
        train: 'True' for training and 'False' for evaluation
        continuous: if 'True' the score-based model is expected to directly take continuous time steps
    Returns:
        score function
    """
    if train:
        model.train()
    else:
        model.eval()
    
    if isinstance(sde, sdes.VPSDE) or isinstance(sde, sdes.subVPSDE):
        def score_fn(x, t):
            if continuous or isinstance(sde, sdes.subVPSDE):
                labels = t * 999
                score = model(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                labels = t * (sde.N - 1)
                score = model(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score
    elif isinstance(sde, sdes.VESDE):
        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()
                
            score = model(x, labels)
            return score
    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name___} not yet supported.")
    
    return score_fn
        
def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1, ))

def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))