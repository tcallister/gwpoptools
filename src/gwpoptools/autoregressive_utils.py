import numpyro
import jax.numpy as jnp
from jax import lax
import scipy
import numpy as np


def build_ar1(total, new_element):

    """
    Helper function to iteratively construct an AR process, given a previous value and a new parameter/innovation pair. Used together with `jax.lax.scan`

    Parameters
    ----------
    total : float
        Processes' value at the previous iteration
    new_element : tuple
        Tuple `(c,w)` containing new parameter/innovation; see Eq. 4 of the associated paper

    Returns
    -------
    total : float
        AR process value at new point
    """

    c, w = new_element
    total = c*total+w
    return total, total


def sampleAutoregressiveProcess(name, reference_grid, reference_index, std_std, ln_tau_mu, ln_tau_std, regularization_std):

    """
    Helper function to sample AR process over a regular grid.

    Parameters
    ----------
    name : `str`
        Name of parameter over which we're building AR process. Will be
        inserted into names of sampling sites
    reference_grid : `jax.numpy.array`
        Regular grid of values over which we're constructing process
    reference_index : `int`
        Index of site from which we will begin sampling left and right
    std_std : `float`
        Standard deviation of prior on AR standard deviation
    ln_tau_mu : `float`
        Mean of prior on log-scale length of AR process
    ln_tau_std : `float`
        Standard deviation of prior on log-scale length
    regularization_std : `float`
        Standard deviation associated with regularization prior on AR hyperparameters 

    Returns
    -------
    fs : `jax.numpy.array`
        AR process evaluated across reference grid
    """

    # First get variance of the process
    ar_std = numpyro.sample("ar_"+name+"_std", numpyro.distributions.HalfNormal(std_std))

    # Next the autocorrelation length
    log_ar_tau = numpyro.sample("log_ar_"+name+"_tau", numpyro.distributions.Normal(ln_tau_mu, ln_tau_std))
    ar_tau = numpyro.deterministic("ar_"+name+"_tau", jnp.exp(log_ar_tau))

    # As discussed in Appendix B, we need a regularizing log-likelihood factor to help stabilize our inference
    numpyro.factor(name+"_regularization", -(ar_std/jnp.sqrt(ar_tau))**2/(2.*regularization_std**2))

    # Sample an initial rate density at reference point
    ln_f_ref_unscaled = numpyro.sample("ln_f_"+name+"_ref_unscaled", numpyro.distributions.Normal(0, 1))
    ln_f_ref = ln_f_ref_unscaled*ar_std

    # Generate forward steps and join to reference value, following the procedure outlined in Appendix A
    # First generate a sequence of unnormalized steps from N(0, 1), then rescale to compute weights and innovations
    deltas = jnp.diff(reference_grid)[0]
    steps_forward = numpyro.sample(name+"_steps_forward", numpyro.distributions.Normal(0, 1), sample_shape=(reference_grid[reference_index:].size-1, ))
    ws_forward = jnp.sqrt(1.-jnp.exp(-2.*deltas/ar_tau))*ar_std*steps_forward
    phis_forward = jnp.ones(ws_forward.size)*jnp.exp(-deltas/ar_tau)
    final, ln_f_high = lax.scan(build_ar1, ln_f_ref, jnp.transpose(jnp.array([phis_forward, ws_forward])))
    ln_fs = jnp.append(ln_f_ref, ln_f_high)

    # Generate backward steps and prepend to forward steps above following an analogous procedure
    steps_backward = numpyro.sample(name+"_steps_backward", numpyro.distributions.Normal(0, 1), sample_shape=(reference_grid[:reference_index].size, ))
    ws_backward = jnp.sqrt(1.-jnp.exp(-2.*deltas/ar_tau))*ar_std*steps_backward
    phis_backward = jnp.ones(ws_backward.size)*jnp.exp(-deltas/ar_tau)
    final, ln_f_low = lax.scan(build_ar1, ln_f_ref, jnp.transpose(jnp.array([phis_backward, ws_backward])))
    ln_fs = jnp.append(ln_f_low[::-1], ln_fs)

    fs = jnp.exp(ln_fs)
    numpyro.deterministic("fs_"+name,fs)

    return fs


def compute_ar_prior_params(dR_max, dR_event, deltaX, N_events):

    """
    Function to compute quantities appearing in our prior on AR(1) process variances and autocorrelation lengths,
    following discussion in Appendix B

    Parameters
    ----------
    dR_max : float
        Estimate of the maximum allowed variation in the merger rate across the domain
    dR_event : float
        Estimate of the maximum allowed variation in the merger rate between event locations
    deltaX : float
        Domain width
    N_events : int
        Number of observations in our sample

    Returns
    -------
    Sigma_sig : float
        Standard deviation to be used in a Gaussian prior on AR(1) process standard deviation `sigma`
    Mu_ln_tau : float
        Mean to be used in a Gaussian prior on AR(1) process' log-autocorrelation length
    Sig_ln_tau : float
        Standard deviation to be used in a Gaussian prior on AR(1) process' log-autocorrelation length
    Sigma_ratio : float
        Standard deviation to be used in a Gaussian regularization prior on the ratio `sigma/sqrt(tau)`
    """

    # Compute the 99th percentile of a chi-squared distribution
    q_99 = scipy.special.gammaincinv(1/2, 0.99)

    # Compute standard deviation on `sigma` prior, see Eq. B21
    Sigma_sig = np.log(dR_max)/(2.*q_99**0.5*scipy.special.erfinv(0.99))

    # Expected minimum spacing between events; see Eq. B29
    dx_min = -(deltaX/N_events)*np.log(1.-(1.-np.exp(-N_events))/N_events)

    # Mean and standard deviation on `ln_tau` prior, see Eqs. B26 and B30
    Mu_ln_tau = np.log(deltaX/2.)
    Sigma_ln_tau = (np.log(dx_min) - Mu_ln_tau)/(2**0.5*scipy.special.erfinv(1.-2*0.99))

    # Standard deviation on ratio, see Eq. B25
    Sigma_ratio = (np.log(dR_event)/(2.*scipy.special.erfinv(0.99)))*np.sqrt(N_events/(q_99*deltaX))

    return Sigma_sig, Mu_ln_tau, Sigma_ln_tau, Sigma_ratio


