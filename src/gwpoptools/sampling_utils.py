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


def sampleUniformFromLogit(name, minValue, maxValue, logit_std=logit_std, sample_shape=()):

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
        numpyro.distributions.Normal(0, logit_std), sample_shape=sample_shape)

    # Transform to physical value and get Jacobian
    param, jacobian = get_value_from_logit(logit_param, minValue, maxValue)

    # Undo sampling prior and record result
    numpyro.factor("p_"+name, logit_param**2/(2.*logit_std**2) - jnp.log(jacobian))
    numpyro.deterministic(name, param)
    
    return param


def build_interpolation_map(x_grid, x_query):

    """
    Helper fuction to turn linear interpolation into a matrix operation.
    Specifically, computes a matrix of weights that can be applied to function values
    f(x) defined over a grid, in order to interpolate appropriate values of f(x_query).

    Parameters
    ----------
    x_grid : `array`
        Array of grid points to serve as interpolation reference.
        Must be strictly increasing. Size (m,)
    x_query : `array`
        Array of points at which we want to interpolate f(x). Must be one-dimensional.
        Size (n,)

    Returns
    -------
    interpolation_weights : `array`
        (n, m) array that, when multiplied by f(x), yields linearly interpolated values
    """

    # Complain if insertion points are outside the grid
    if np.any(x_query < x_grid[0]) or np.any(x_query > x_grid[-1]):
        raise ValueError(f"Query points lie outside the grid!")

    # Find where every queried sample sits inside x_grid
    # The bizarre logic is to deal with the special case of a query point sitting directly
    # on the lowest grid point. In this case, we ensure that `index_below = 0` and
    # `index_above = 1`
    index_below = np.maximum(np.searchsorted(x_grid, x_query, side='left') - 1, 0)
    index_above = index_below + 1

    # Calculate fractional positions of query points in their respective intervals
    ws = (x_query - x_grid[index_below])/(x_grid[index_above] - x_grid[index_below])

    # Assign fractional linearized weights to grid values immediately before and after insertion points
    interpolation_weights = np.zeros((x_query.size, x_grid.size))
    interpolation_weights[np.arange(x_query.size), index_below] = 1. - ws
    interpolation_weights[np.arange(x_query.size), index_above] = ws

    return interpolation_weights


