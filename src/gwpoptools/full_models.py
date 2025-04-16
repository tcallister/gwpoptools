import numpyro
import jax
import jax.numpy as jnp
from jax.scipy.special import erf
import scipy
import numpy as np

tmp_max = 100.
tmp_min = 2.


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
    norm = jnp.sqrt(sigma**2*np.pi/2)*(-erf(a) + erf(b))
    ps = jnp.exp(-(samples-mu)**2/(2.*sigma**2))/norm

    # Truncate
    ps = jnp.where(samples<lowCutoff, 0, ps)
    ps = jnp.where(samples>highCutoff, 0, ps)

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
    ps = jnp.where(samples>maxValue, 0, ps)
    ps = jnp.where(samples<minValue, 0, ps)

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
    p_broken_pl = jnp.where(samples<break_location, (samples/break_location)**alpha1, (samples/break_location)**alpha2)
    p_broken_pl = jnp.where(samples<minValue, 0, p_broken_pl)

    # Normalization constant
    if maxValue:
        normalization = (break_location**(1.+alpha1)-minValue**(1.+alpha1))/break_location**alpha1/(1.+alpha1) \
            + (maxValue**(1.+alpha2) - break_location**(1.+alpha2))/break_location**alpha2/(1.+alpha2)
        p_broken_pl = jnp.where(samples>maxValue, 0, p_broken_pl)
    else:
        normalization = (break_location**(1.+alpha1)-minValue**(1.+alpha1))/break_location**alpha1/(1.+alpha1) \
            + (- break_location**(1.+alpha2))/break_location**alpha2/(1.+alpha2)

    p_broken_pl /= normalization

    return p_broken_pl

def gwpopulationTapering(samples, low, delta):

    """
    Tapering function that goes to zero for values less than `low`.
    Follows the gwpopulation tapering convention.

    Parameters
    ----------
    samples : array or float
        Sample locations
    low : float
        Value below which tapering function returns zero
    delta : float
        Length of taper above `low`

    Returns
    -------
    taper : array or float
        Tapering function
    """

    # To prevent infinities, we need to clip denominators appearing below.
    # Note that denom1 approaches zero from above as samples -> low,
    # while denom2 approaches zero from below as samples -> low+delta
    clip = 1e-1
    denom1 = samples-low
    denom2 = samples-low-delta
    denom1 = np.clip(denom1, a_min=clip, a_max=None)
    denom2 = np.clip(denom2, a_min=None, a_max=-clip)
    
    # Compute tapering
    f = jnp.exp(delta/denom1 + delta/denom2)
    taper = 1./(1 + f)

    # Apply bounds
    taper = jnp.where(samples<low, 0, taper)
    taper = jnp.where(samples>=low+delta, 1, taper)

    return taper


def exponentialTapering(samples, onset, dx):

    """
    Tapering function that exponentially suppresses values below `onset`

    Parameters
    ----------
    samples : array or float
        Sample locations
    onset : float
        Value below which tapering function turns on
    dx : float
        Scale length of exponential suppression

    Returns
    -------
    taper : array or float
        Tapering function
    """

    taper = jnp.exp(-(samples-onset)**2/(2*dx**2))
    taper = jnp.where(samples<onset, taper, 1)
    return taper


def transformParams_gwpopulationTapering_to_exponentialTapering(minValue, delta):

    """
    Transforms minimum value and lengthscale characterizing `gwpopulationTapering`
    into the equivalent onset and scale length values defining the best fit
    `exponentialTapering` function
    """

    shift = 0.8273
    rescaling = 0.2681
    exponentialOnset = minValue + shift*delta
    exponentialWidth = rescaling*delta

    return exponentialOnset, exponentialWidth


def massModel(m1, alpha, mu_m1, sig_m1, f_peak, mMax, mMin, dmMax, dmMin, tmp_max=tmp_max):

    """
    Baseline primary mass model, described as a mixture between a power law
    and gaussian, with exponential tapering functions at high and low masses

    Parameters
    ----------
    m1 : array or float
        Primary masses at which to evaluate probability densities
    alpha : float
        Power-law index
    mu_m1 : float
        Location of possible Gaussian peak
    sig_m1 : float
        Stanard deviation of possible Gaussian peak
    f_peak : float
        Approximate fraction of events contained within Gaussian peak (not exact due to tapering)
    mMax : float
        Location at which high-mass tapering begins
    mMin : float
        Location at which low-mass tapering begins
    dmMax : float
        Scale width of high-mass tapering function
    dmMin : float
        Scale width of low-mass tapering function

    Returns
    -------
    p_m1s : jax.numpy.array
        Unnormalized array of probability densities
    """

    # Define power-law and peak
    p_m1_pl = (1.+alpha)*m1**(alpha)/(tmp_max**(1.+alpha) - tmp_min**(1.+alpha))
    p_m1_peak = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin, low_filter, 1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax, high_filter, 1.)

    # Apply filters to combined power-law and peak
    return (f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)*low_filter*high_filter


def primaryMassModel_O4A(m1, alpha1, alpha2, mBreak, mu_m1, sig_m1, f_peak1, mu_m2, sig_m2, f_peak2, mMin, dmMin, mMax, dmMax):

    """
    Baseline primary mass model for O4a Astro Distributions paper, composed of a mixture
    between a broken power law and two Gaussian peaks, with tapering at low masses.

    Parameters
    ----------
    m1 : array or float
        Primary masses at which to evaluate probability densities
    alpha : float
        Power-law index
    mu_m1 : float
        Location of possible Gaussian peak
    sig_m1 : float
        Stanard deviation of possible Gaussian peak
    f_peak : float
        Approximate fraction of events contained within Gaussian peak (not exact due to tapering)
    mMax : float
        Location at which high-mass tapering begins
    mMin : float
        Location at which low-mass tapering begins
    dmMax : float
        Scale width of high-mass tapering function
    dmMin : float
        Scale width of low-mass tapering function

    Returns
    -------
    f_m1s : jax.numpy.array
        Unnormalized array of probability densities
    """

    # Compute broken power law and Gaussian peaks
    lower_truncation = 3.
    p_bpl = truncatedBrokenPowerLaw(m1, alpha1, alpha2, lower_truncation, mBreak, False)
    p_m1_peak1 = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)
    p_m1_peak2 = jnp.exp(-(m1-mu_m2)**2/(2.*sig_m2**2))/jnp.sqrt(2.*np.pi*sig_m2**2)

    # Tapering function
    exponential_onset, exponential_scale = transformParams_gwpopulationTapering_to_exponentialTapering(mMin, dmMin)
    taper = exponentialTapering(m1, exponential_onset, exponential_scale)

    taperHigh = jnp.exp(-(m1-mMax)**2/(2*dmMax**2))
    taperHigh = jnp.where(m1>mMax, taperHigh, 1)

    # Form full mixture and return
    f_m1 = ((1-f_peak1-f_peak2)*p_bpl + f_peak1*p_m1_peak1 + f_peak2*p_m1_peak2)*taper*taperHigh
    return f_m1
    

def massRatioModel_O4A(q, m1, beta, m2_low, delta_m2):

    """
    Default mass ratio distribution for the O4A Astro Dist paper, given
    by a tapered power law.

    Parameters
    ----------
    q : float or array
        Mass ratios at which to evaluate distribution
    m1 : float or array
        Primary mass values, used for tapering over m2
    beta : float
        Power-law index
    m2_low : float
        Minimum secondary mass
    delta_m2 : float
        Scale over which secondary mass tapering takes place

    Returns
    -------
    f_m2 : float or array
        Unnormalized array of probability densities, conditioned on `m1`
    """

    # Compute tapering, which is performed in m2 space
    m2 = q*m1
    exponential_onset, exponential_scale = transformParams_gwpopulationTapering_to_exponentialTapering(m2_low, delta_m2)
    taper = exponentialTapering(m2, exponential_onset, exponential_scale)

    # Combine power law and tapering, return
    f_m2 = q**beta * taper
    return f_m2


def TEST_massRatioModel_O4A(q, beta):

    """
    Default mass ratio distribution for the O4A Astro Dist paper, given
    by a tapered power law.

    Parameters
    ----------
    q : float or array
        Mass ratios at which to evaluate distribution
    m1 : float or array
        Primary mass values, used for tapering over m2
    beta : float
        Power-law index
    m2_low : float
        Minimum secondary mass
    delta_m2 : float
        Scale over which secondary mass tapering takes place

    Returns
    -------
    f_m2 : float or array
        Unnormalized array of probability densities, conditioned on `m1`
    """

    # Compute tapering, which is performed in m2 space
    q_min = 3./200.
    p_m2 = (1.+beta)*q**beta/(1 - q_min**(1.+beta))

    return p_m2


def primaryMassModel_brokenPL_twoPeaks_independentBetas(
        m1,
        q,
        alpha1,
        alpha2,
        mBreak,
        mu_m1,
        sig_m1,
        f_peak1,
        mu_m2,
        sig_m2,
        f_peak2,
        mMin,
        dmMin,
        mMax,
        dmMax,
        beta_peak1,
        m2_low_peak1,
        delta_m2_peak1,
        beta_peak2,
        m2_low_peak2,
        delta_m2_peak2,
        beta_pl,
        m2_low_pl,
        delta_m2_pl,
        *,
        cached_normalization_data=None,
        return_cache=False):

    """
    Baseline primary mass model for O4a Astro Distributions paper, composed of a mixture
    between a broken power law and two Gaussian peaks, with tapering at low masses.

    Parameters
    ----------
    m1 : array or float
        Primary masses at which to evaluate probability densities
    q : array or float
        Mass ratios at which to evaluate probability densities
    alpha : float
        Power-law index
    mu_m1 : float
        Location of possible Gaussian peak
    sig_m1 : float
        Stanard deviation of possible Gaussian peak
    f_peak : float
        Approximate fraction of events contained within Gaussian peak (not exact due to tapering)
    mMax : float
        Location at which high-mass tapering begins
    mMin : float
        Location at which low-mass tapering begins
    dmMax : float
        Scale width of high-mass tapering function
    dmMin : float
        Scale width of low-mass tapering function
    beta_peak1 : float
        Power-law index for mass ratio distribution in lower peak
    m2_low_peak1 : float
        Minimum secondary mass in low-mass peak
    delta_m2_peak1 : float
        Scale over which secondary mass tapering takes place
    beta_peak2 : float
        Power-law index for mass ratio distribution in upper peak
    m2_low_peak2 : float
        Minimum secondary mass in high-mass peak
    delta_m2_peak2 : float
        Scale over which secondary mass tapering takes place
    beta_pl : float
        Power-law index for mass ratio distribution in power-law continuum
    m2_low_pl : float
        Minimum secondary mass in power-law
    delta_m2_pl : float
        Scale over which secondary mass tapering takes place

    Returns
    -------
    f_m1s_qs : jax.numpy.array
        Unnormalized array of probability densities
    """

    # Compute broken power law and Gaussian peaks
    lower_truncation = 3.
    p_bpl = truncatedBrokenPowerLaw(m1, alpha1, alpha2, lower_truncation, mBreak, False)
    p_m1_peak1 = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)
    p_m1_peak2 = jnp.exp(-(m1-mu_m2)**2/(2.*sig_m2**2))/jnp.sqrt(2.*np.pi*sig_m2**2)

    # Mass ratio distributions in each component
    p_q_peak1 = massRatioModel_O4A(q, m1, beta_peak1, m2_low_peak1, delta_m2_peak1)
    p_q_peak2 = massRatioModel_O4A(q, m1, beta_peak2, m2_low_peak2, delta_m2_peak2)
    p_q_pl = massRatioModel_O4A(q, m1, beta_pl, m2_low_pl, delta_m2_pl)

    # Normalize
    if cached_normalization_data is None:

        m_grid = jnp.logspace(jnp.log10(3.), jnp.log10(200.), 200)
        q_grid = jnp.linspace(1e-3, 1, 199)
        M_GRID, Q_GRID = jnp.meshgrid(m_grid, q_grid)

        F_Q_peak1 = massRatioModel_O4A(Q_GRID, M_GRID, beta_peak1, m2_low_peak1, delta_m2_peak1)
        F_Q_peak2 = massRatioModel_O4A(Q_GRID, M_GRID, beta_peak2, m2_low_peak2, delta_m2_peak2)
        F_Q_pl = massRatioModel_O4A(Q_GRID, M_GRID, beta_pl, m2_low_pl, delta_m2_pl)
        f_q_peak1_norms = jnp.trapezoid(F_Q_peak1, q_grid, axis=0)
        f_q_peak2_norms = jnp.trapezoid(F_Q_peak2, q_grid, axis=0)
        f_q_pl_norms = jnp.trapezoid(F_Q_pl, q_grid, axis=0)

        cached_normalization_data = {
            'm_grid':m_grid,
            'q_grid':q_grid,
            'f_q_peak1_norms':f_q_peak1_norms,
            'f_q_peak2_norms':f_q_peak2_norms,
            'f_q_pl_norms':f_q_pl_norms}

    p_q_peak1 /= jnp.interp(jnp.log10(m1), jnp.log10(cached_normalization_data['m_grid']), cached_normalization_data['f_q_peak1_norms'])
    p_q_peak2 /= jnp.interp(jnp.log10(m1), jnp.log10(cached_normalization_data['m_grid']), cached_normalization_data['f_q_peak2_norms'])
    p_q_pl /= jnp.interp(jnp.log10(m1), jnp.log10(cached_normalization_data['m_grid']), cached_normalization_data['f_q_pl_norms'])

    # Tapering function
    exponential_onset, exponential_scale = transformParams_gwpopulationTapering_to_exponentialTapering(mMin, dmMin)
    taper = exponentialTapering(m1, exponential_onset, exponential_scale)

    taperHigh = jnp.exp(-(m1-mMax)**2/(2*dmMax**2))
    taperHigh = jnp.where(m1>mMax, taperHigh, 1)

    # Form full mixture and return
    f_m1 = ((1-f_peak1-f_peak2)*p_bpl*p_q_pl + f_peak1*p_m1_peak1*p_q_peak1 + f_peak2*p_m1_peak2*p_q_peak2)*taper*taperHigh

    if return_cache:
        return f_m1, cached_normalization_data

    else:
        return f_m1


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
