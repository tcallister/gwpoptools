import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import scipy
import numpy as np

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


def compute_gp_prior_params(dPsi_max, dPsi_event, deltaX, N_events):

    """
    Function to compute quantities appearing in priors on GP variances and autocorrelation lengths

    Parameters
    ----------
    dPsi_max : float
        Estimate of the maximum allowed variation in the GP across the domain
    dPsi_event : float
        Estimate of the maximum allowed variation in the GP between event locations
    deltaX : float
        Domain width
    N_events : int
        Number of observations in our sample

    Returns
    -------
    Sigma_sig : float
        Standard deviation to be used in a Gaussian prior on GP standard deviation `sigma`
    Mu_ln_tau : float
        Mean to be used in a Gaussian prior on GP's log-autocorrelation length
    Sig_ln_tau : float
        Standard deviation to be used in a Gaussian prior on GPs' log-autocorrelation length
    Sigma_ratio : float
        Standard deviation to be used in a Gaussian regularization prior on the ratio `sigma/tau`
    """

    # Compute the 99th percentile of a chi-squared distribution
    q_99 = scipy.stats.chi2.ppf(0.99, 1)

    # Compute standard deviation on `sigma` prior
    Sigma_sig = dPsi_max/(2.*q_99**0.5*scipy.special.erfinv(0.99))

    # Expected minimum spacing between events in large N limit
    dx_min = deltaX/N_events**2

    # Mean and standard deviation on `ln_tau` prior
    Mu_ln_tau = np.log(deltaX/2.)
    Sigma_ln_tau = np.log(2/N_events**2) / (np.sqrt(2.) * scipy.special.erfinv(1-2*0.99))

    # Standard deviation on ratio
    Sigma_ratio = N_events * dPsi_event / (deltaX * np.sqrt(2*q_99) * scipy.special.erfinv(0.99))

    return Sigma_sig, Mu_ln_tau, Sigma_ln_tau, Sigma_ratio

