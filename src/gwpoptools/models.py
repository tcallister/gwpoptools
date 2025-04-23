import jax
import jax.numpy as jnp
from jax.scipy.special import erf


def truncatedNormal(samples, mu, sigma, lowCutoff, highCutoff):

    """
    Jax-enabled truncated normal distribution

    Parameters
    ----------
    samples : `jax.numpy.array` or float
        Locations at which to evaluate probability density
    mu : float
        Mean of truncated normal
    sigma : float
        Standard deviation of truncated normal
    lowCutoff : float
        Lower truncation bound
    highCutoff : float
        Upper truncation bound

    Returns
    -------
    ps : jax.numpy.array or float
        Probability density at the locations of `samples`
    """

    a = (lowCutoff-mu)/jnp.sqrt(2*sigma**2)
    b = (highCutoff-mu)/jnp.sqrt(2*sigma**2)
    norm = jnp.sqrt(sigma**2*jnp.pi/2)*(-erf(a) + erf(b))
    ps = jnp.exp(-(samples-mu)**2/(2.*sigma**2))/norm

    # Truncate
    ps = jnp.where(samples < lowCutoff, 0, ps)
    ps = jnp.where(samples > highCutoff, 0, ps)

    return ps


def truncatedPowerLaw(samples, beta, minValue, maxValue):

    """
    Normalized truncated power-law model

    Parameters
    ----------
    samples : float or array
        Values at which to evaluate probability density
    beta : float
        Power-law index
    minValue : float
        Lower bound on power law
    maxValue : float
        Upper bound on power law

    Returns
    -------
    ps : float or array
        Normalized probability densities
    """

    ps = (1.+beta)*samples**beta/(maxValue**(1.+beta)-minValue**(1.+beta))
    ps = jnp.where(samples > maxValue, 0, ps)
    ps = jnp.where(samples < minValue, 0, ps)

    return ps


def truncatedBrokenPowerLaw(samples, alpha1, alpha2, minValue, break_location, maxValue):

    """
    Normalized and truncated broken power-law model

    Parameters
    ----------
    samples : float or array
        Values at which to evaluate probability density
    alpha1 : float
        Power-law index below `break_location`
    alpha2 : float
        Power-law index above `break_location`
    minValue : float
        Lower bound on broken power law
    break_location : float
        Break point at which power-law index changes
    maxValue : float
        Upper bound on broken power law

    Returns
    -------
    ps : float or array
        Normalized probability densities
    """

    # Compute piecewise cases
    p_broken_pl = jnp.where(samples < break_location, (samples/break_location)**alpha1, (samples/break_location)**alpha2)
    p_broken_pl = jnp.where(samples < minValue, 0, p_broken_pl)

    # Normalization constant
    if maxValue:
        normalization = (break_location**(1.+alpha1)-minValue**(1.+alpha1))/break_location**alpha1/(1.+alpha1) \
            + (maxValue**(1.+alpha2) - break_location**(1.+alpha2))/break_location**alpha2/(1.+alpha2)
        p_broken_pl = jnp.where(samples > maxValue, 0, p_broken_pl)
    else:
        normalization = (break_location**(1.+alpha1)-minValue**(1.+alpha1))/break_location**alpha1/(1.+alpha1) \
            + (- break_location**(1.+alpha2))/break_location**alpha2/(1.+alpha2)

    p_broken_pl /= normalization

    return p_broken_pl


def sigmoid(xs, ymin, ymax, xc, dx):

    """
    Function defining a sigmoid.

    Parameters
    ----------
    xs : `array`
        Array of values at which to evaluate sigmoid
    ymin : `float`
        Min value as `xs` approaches negative infinity
    ymax : `float`
        Max value as `xs` approaches infinity
    xc : `float`
        Central value about which the transition occurs
    dx : `float`
        Scale width of the transition

    Returns
    -------
    sigmoid_vals : `array`
        Array of sigmoid values
    """

    sigmoid_vals = ymin + (ymax-ymin)/(1. + jnp.exp(-(xs-xc)/dx))
    return sigmoid_vals
