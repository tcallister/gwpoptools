import pytest
import numpy as np
from pytest import approx
import jax.numpy as jnp
import gwpoptools.models as models
import gwpoptools.spin_models as spin_models


# Test expected numerical results from spin_models.spinTiltModel_isotropic_plus_gaussian_IID
@pytest.mark.parametrize("params, expected_result", [
    ([0.2, 0.5, 0.9, 1.1], 0.5496993),
    ([jnp.array([-0.5, 1.]), 0.5, 0.9, 1.1], [0.41325963, 0.615449]),
    ])
def test_evaluate_spinTiltModel_isotropic_plus_gaussian_IID(params, expected_result):
    assert spin_models.spinTiltModel_isotropic_plus_gaussian_IID(*params) == approx(expected_result)


# Test expected numerical results from spin_models.spinTiltModel_isotropic_plus_gaussian_nonIID
@pytest.mark.parametrize("params, expected_result", [
    ([0.2, -1, 0.5, 0.9, 1.1], 0.17448625),
    ([jnp.array([0, 0.5]), jnp.array([-1, 0.3]), 0.5, 0.9, 1.1], [0.1683568 , 0.34224674]),
    ])
def test_evaluate_spinTiltModel_isotropic_plus_gaussian_nonIID(params, expected_result):
    assert spin_models.spinTiltModel_isotropic_plus_gaussian_nonIID(*params) == approx(expected_result)


# Test that integration over non-IID model yields marginal model
def test_marginalize_spinTiltModel_isotropic_plus_gaussian_nonIID():
    f_iso, mu, sigma = 0.5, 0.9, 1.1
    costilt_grid = jnp.linspace(-1, 1, 600)
    p_cost1_cost2 = spin_models.spinTiltModel_isotropic_plus_gaussian_nonIID(costilt_grid[:, jnp.newaxis], costilt_grid[jnp.newaxis, :], f_iso, mu, sigma)
    p_cost1_marginalized = jnp.trapezoid(p_cost1_cost2, costilt_grid, axis=1)
    p_cost1_direct = spin_models.spinTiltModel_isotropic_plus_gaussian_IID(costilt_grid, f_iso, mu, sigma)
    assert jnp.allclose(p_cost1_marginalized, p_cost1_direct)


# Test that spin models properly reduce to truncated normal
def test_limit_spinTiltModel_isotropic_plus_gaussian_IID():
    value, f_iso, mu, sigma = -0.3, 0, 0.5, 0.9
    a = spin_models.spinTiltModel_isotropic_plus_gaussian_IID(value, f_iso, mu, sigma)
    b = models.truncatedNormal(value, mu, sigma, -1, 1)
    assert a == b
