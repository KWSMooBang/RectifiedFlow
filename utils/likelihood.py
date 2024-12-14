import numpy as np
import torch
import torchvision

from scipy import integrate
from models import utils as mutils


def get_div_fn(fn):
    """Create the divergence function of 'fn' using the Hutchinson-Skilling trace estimator."""
    
    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


def get_likelihood_fn(sde, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point

    Args:
        sde: A 'sdes.SDE' object that represents the forward SDE
        inverse_scaler: The inverse data normalizer
        hutchinson_type: 'Rademacher' or 'Gaussian'. The type of noise for Hutchinson-Skilling trace estimator
        rtol: A 'float' number. The relative tolerance level of the black-box ODE solver.
        atol: A 'float' number. The absolute tolerance level of the black-box ODE solver.
        method (str, optional): The algorithm for the black-box ODE solver 
        eps (float, optional): The probability flow ODE is integrated to 'eps' for numerical stability
    Returns:
        A function that a batch of data points an returns the log_likelihoods in bits/dim,
            the latent code, and the number of function evaluation cost by computation
    """
    
    def drift_fn(model, x, t):
        """The drift function of the reverse-time SDE"""
        score_fn = mutils.get_score_fn(sde, model, train=False, continuous=False)
        # Probability flow ODE is a special case of Reverse SDE
        reversed_sde = sde.reverse(score_fn, probability_flow=True)
        return reversed_sde(x, t)[0]
    
    def div_fn(model, x, t, noise):
        return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)
    
    def likelihood_fn(model, data):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
            model: A score model
            data: A pytorch tensor
        Returns:
            bpd: A pytorch tensor of shape [batch_size]. The log-likelihoods on 'data' in bits/dim.
            z: A pytorch tensor of the same shape as 'data'. The latent representation of 'data' under the
                prbability flow ODE
            nfe: An integer. The number of function evaluations used for running the black-box ODE solver
        """
        with torch.no_grad():
            shape = data.shape
            if hutchinson_type == 'Gaussian':
                epsilon = torch.randn_like(data)
            elif hutchinson_type == 'Rademacher':
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 -1,
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
        
            def ode_func(t, x):
                sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = mutils.to_flattened_numpy(drift_fn(model, sample, vec_t))
                logp_grad = mutils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
                return np.concatenate([drift, logp_grad], axis=0)
            
            init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0], ))], axis=0)
            solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
            delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0], )).to(data.device).type(torch.float32)
            prior_logp = sde.prior_logp(z)
            bpd = -(prior_logp + delta_logp) / np.log(2)
            N = np.prod(shape[1:])
            bpd = bpd / N
            offset = 7. - inverse_scaler(-1.)
            bpd = bpd + offset
            return bpd, z, nfe
    
    return likelihood_fn


def get_likelihood_fn_rf(sde, inverse_scaler, hutchinson_type='Rademacher',
                         rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point for Rectified Flow.

    Args:
        sde: A 'sdes.SDE' object that represents the forward SDE.
        inverse_scaler: The inverse data normalizer
        hutchinson_type: 'Rademacher' or 'Gaussian'. The type of noise for Hutchinson-Skilling trace estimator
        rtol: A 'float' number. The relative tolerance level of the black-box ODE solver.
        atol: A 'float' number. The absolute tolerance level of the black-box ODE solver.
        method (str, optional): The algorithm for the black-box ODE solver 
        eps (float, optional): The probability flow ODE is integrated to 'eps' for numerical stability
    Returns:
        A function that a batch of data points an returns the log_likelihoods in bits/dim,
            the latent code, and the number of function evaluation cost by computation
    """
    def div_fn(x, t, noise):
        return get_div_fn(lambda xx, tt: sde.model(xx, tt))(x, t, noise)
    
    def get_prior_logp(z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps
    
    def likelihood_fn(model, data):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
            model: A score model
            data: A pytorch tensor
        Returns:
            bpd: A pytorch tensor of shape [batch_size]. The log-likelihoods on 'data' in bits/dim.
            z: A pytorch tensor of the same shape as 'data'. The latent representation of 'data' under the
                prbability flow ODE
            nfe: An integer. The number of function evaluations used for running the black-box ODE solver
        """
        with torch.no_grad():
            shape = data.shape
            if hutchinson_type == 'Gaussian':
                epsilon = torch.randn_like(data)
            elif hutchinson_type == 'Rademacher':
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
            else: 
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
            
            def ode_func(t, x):
                sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = mutils.to_flattened_numpy(sde.model(sample, vec_t))
                logp_grad = mutils.to_flattened_numpy(div_fn(sample, vec_t, epsilon))
                return np.concatenate([drift, logp_grad], axix=0)
            
            init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0], ))], axis=0)
            solution = integrate.solve_ivp(
                ode_func, (sde.T, eps), init, 
                rtol=rtol, atol=atol, method=method
            )
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
            
            delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
            prior_logp = get_prior_logp(z)
            bpd = -(prior_logp + delta_logp) / np.log(2)
            N = np.prod(shape[1:])
            bpd = bpd / N
            offset = 7. - inverse_scaler(-1.)
            bpd = bpd + offset
            return bpd, z, nfe
        
        return likelihood_fn