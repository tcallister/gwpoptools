import numpyro
import jax.numpy as jnp
import numpy as np

logit_std = 2.5


def monteCarloIntegralAndDiagnostics(weights, n):

    """
    Function to compute Monte Carlo integrals and return variance and effective
    counts.

    Parameters
    ----------
    weights : array
        Monte Carlo weights
    n : int
        Total number of samples, to divide sum of `weights`

    Returns
    -------
    mean : float
        Estimated mean
    n_effective : float
        Effective number of samples
    variance : float
        Variance on the mean
    """

    # Do sums once
    sum_weights = jnp.sum(weights)
    sum_weights_squared = jnp.sum(weights**2)

    # Estimate of the mean
    mean = sum_weights/n

    # Effective sample count
    n_effective = sum_weights**2/sum_weights_squared

    # Variance
    variance = sum_weights_squared/n**2 - mean**2/n

    return mean, n_effective, variance


def logit_transform(x, x_min, x_max):

    """
    Transform bounded data to an unbounded domain using a logit transform
    and compute the Jacobian of the transformation.
    
    Parameters
    ----------
    x : jnp.ndarray
        Input data in the range [xmin, xmax].
    xmin : float
        Minimum value of the bounded domain.
    xmax : float
        Maximum value of the bounded domain.
        
    Returns
    -------
    logit_x : jnp.ndarray
        Transformed data in the unbounded domain
    jacobian : jnp.ndarray
        Jacobian of the transformation, d(logit x)/dx
    """

    # Compute logit
    logit_x = jnp.log((x-x_min) / (x_max - x))
    
    # Compute the Jacobian
    dlogit_dx = 1./(x-x_min) + 1./(x_max-x)

    return logit_x, dlogit_dx


def get_value_from_logit(logit_x, x_min, x_max):

    """
    Function to map a variable `logit_x`, defined on `(-inf,+inf)`, to a quantity `x`
    defined on the interval `(x_min,x_max)`.

    Parameters
    ----------
    logit_x : float
        Quantity to inverse-logit transform
    x_min : float
        Lower bound of `x`
    x_max : float
        Upper bound of `x`

    Returns
    -------
    x : float
       The inverse logit transform of `logit_x`
    dlogit_dx : float
       The Jacobian between `logit_x` and `x`; divide by this quantity to convert a uniform prior on `logit_x` to a uniform prior on `x`
    """

    exp_logit = jnp.exp(logit_x)
    x = (exp_logit*x_max + x_min)/(1.+exp_logit)
    dlogit_dx = 1./(x-x_min) + 1./(x_max-x)

    return x, dlogit_dx


def sampleUniformFromLogit(name, minValue, maxValue, logit_std=logit_std):

    """
    Function to sample a uniform bounded prior via an unbounded prior in a transformed
    logit space.

    Parameters
    ----------
    name : `str`
        Name for target parameter
    minValue : `float`
        Minimum allowed value
    maxValue : `float`
        Maximum allowed value
    logit_std : `float`
        Standard deviation for initial sampling prior in logit space.
        This is later undone, but may effect sampling efficiency.
        Default 2.5.

    Returns
    -------
    param : `float`
        Sampled parameter value
    """

    # Draw from unconstrained logit space
    logit_param = numpyro.sample("logit_"+name,
        numpyro.distributions.Normal(0, logit_std))

    # Transform to physical value and get Jacobian
    param, jacobian = get_value_from_logit(logit_param, minValue, maxValue)

    # Undo sampling prior and record result
    numpyro.factor("p_"+name, logit_param**2/(2.*logit_std**2) - jnp.log(jacobian))
    numpyro.deterministic(name, param)
    
    return param

