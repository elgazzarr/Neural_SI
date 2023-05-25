import jax.numpy as jnp
from jax.scipy.special import gammaln
import equinox as eqx

#input−target∗log(input)+log(target!)

def poisson_nll(input, target, eps=1e-5, stirling_approx=False):
    #input−target∗log(input)+log(target!)
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)
    """
    
    loss = input - target * jnp.log(input + eps)
    
    if stirling_approx:
        loss += -(target * jnp.log(target + eps) - target + 0.5 * jnp.log(2 * jnp.pi * target + eps))

    return jnp.mean(loss)

def logprob(y, mean, sd):
    # defines log likelihood between observed data y and predicted data mean and sd
    return -0.5 * ((y - mean) / sd) ** 2 - jnp.log(sd) - 0.5 * jnp.log(2 * jnp.pi)

def normal_kl_divergence(loc1, scale1, loc2, scale2):
    # KL divergence between two normal distributions
    return jnp.log(scale2 / scale1) + (scale1 ** 2 + (loc1 - loc2) ** 2) / (2 * scale2 ** 2) - 0.5

