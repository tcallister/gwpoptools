import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def sampleGaussianProcess(name, grid, sigma, l, mu): 
    """ 
    Helper funcion to sample a Gaussian Process over a grid 

    Parameters 
    ----------
    name : `str`
        Name of parameter over which we're building GP. Will be
        inserted into names of sampling sites
    grid : `jax.numpy.array`
        Grid of values over which we're constructing the GP
    sigma : `float`
        Amplitude of the GP 
    l : `float`
        Length scale of the GP  
    mu : `float`
        Mean of the GP 
    
    Returns
    -------
    f_grid : `jax.numpy.array`
        GP evaluated across grid
    """

    N = len(grid) 

    # Build covariance matrix 
    Sigma = sigma**2 * jnp.exp(-(grid[:,jnp.newaxis] - grid[jnp.newaxis,:])**2 / (2*l**2)) 

    # Cholesky decomposition
    jitter = 1e-6
    L = jnp.linalg.cholesky(Sigma + jitter * jnp.eye(N))
    
    # Draw the samples 
    z = numpyro.sample("latent_f_"+name, dist.Normal(jnp.zeros(N), jnp.ones(N)))
    
    # Reparameterization
    f_grid = numpyro.deterministic("f_"+name, L @ z + mu)
    
    return f_grid