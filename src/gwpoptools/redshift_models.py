import numpyro
import jax
import jax.numpy as jnp


def brokenPowerLaw(z, kappa1, kappa2, zBreak):

    """
    Broken power-law model, to be primarily used as a model for the BBH rate evolution with redshift

    Parameters
    ----------
    z : float or array
        Location at which to evaluate broken power law
    kappa1 : float
        Power law index at `z<zBreak`    
    kappa2 : float
        Power law index at `z>zBreak`
    zBreak : float
        Break point

    Returns
    -------
    f_z : float or array
        Unnormalized broken power law evaluated across `z` 
    """

    return jnp.where(z<zBreak, ((1.+z)/(1.+zBreak))**kappa1, ((1.+z)/(1.+zBreak))**kappa2)


def madauDickinsonModel(z, alpha, beta, zp):

    """
    Madau-Dickinson-like SFR model for the CBC volumetric merger rate

    Parameters
    z : float or array
        Location at which to evaluate model
    alpha : float
        Power-law index governing low-redshift behavior, which grows as `(1+z)**alpha`
    beta : float
        Power-law index governing high-redshift behavior, which falls as `(1+z)**(-beta)`
    zp : float
        Transition point marking approximate location of "cosmic noon"

    Returns
    -------
    f_z : float or array
        Unnormalized MD model evaluated across `z`
    """

    return (1.+z)**alpha/(1.+((1.+z)/(1.+zp))**(alpha+beta))
