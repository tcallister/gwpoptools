import jax
import jax.numpy as jnp
from gwpoptools.models import truncatedNormal


def spinTiltModel_isotropic_plus_gaussian_IID(cos_tilts, f_iso, mu_cost, sig_cost):

    """
    Normalized probability distribution for component spin tilts, in which cosine-tilt
    distribution is a mixture between an isotropic component and a freely moving peak.

    Parameters
    ----------
    cos_tilts : float or array
        Values at which to evaluate distribution
    f_iso : float
        Fraction of systems in isotropic component
    mu_cost : float
        Mean of freely moving peak
    sig_cost : float
        Standard deviation of freely moving peak

    Returns
    -------
    ps : float or array
        Normalized probability distributions
    """

    return f_iso*(1./2.) + (1.-f_iso)*truncatedNormal(cos_tilts, mu_cost, sig_cost, -1, 1)


def spinTiltModel_isotropic_plus_gaussian_nonIID(cos_tilts1, cos_tilts2, f_iso, mu_cost, sig_cost):

    """
    Normalized probability distribution for component spin tilts, in which cosine-tilt
    distribution is a mixture between an isotropic component and a freely moving peak.
    This is the default spin tilt model for the O4A Astro Dist paper, in which spins are
    *not* independently distributed, but instead each live in the isotropic or gaussian
    subpopulations

    Parameters
    ----------
    cos_tilts1 : float or array
        Cosine spin-orbit tilts of primary mass
    cos_tilts2 : float or array
        Cosine spin-orbit tilts of secondary mass
    f_iso : float
        Fraction of systems in isotropic component
    mu_cost : float
        Mean of freely moving peak
    sig_cost : float
        Standard deviation of freely moving peak

    Returns
    -------
    ps : float or array
        Normalized probability distributions
    """

    return f_iso*(1./4.) \
            + (1.-f_iso)*truncatedNormal(cos_tilts1, mu_cost, sig_cost, -1, 1)\
                        *truncatedNormal(cos_tilts2, mu_cost, sig_cost, -1, 1)


