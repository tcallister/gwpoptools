import pytest
import numpy as np
from pytest import approx
import jax.numpy as jnp
import gwpoptools.models as models


# Test expected numerical results from models.truncatedPowerLaw
@pytest.mark.parametrize("params, expected_result", [
    ([5., 1., 1., 10.], 10./99.),
    ([3., 2., 1., 5.], 27./124.),
    ([jnp.array([3., 5.]), 1., 1., 10.], [6./99., 10./99.]),
    ])
def test_evaluate_truncatedPowerLaw(params, expected_result):
    assert models.truncatedPowerLaw(*params) == approx(expected_result)


# Test that truncatedPowerLaw returns zero outside bounds
def test_truncatedPowerLaw_below_min():
    assert models.truncatedPowerLaw(0.5, 1., 1., 10.) == approx(0.)

def test_truncatedPowerLaw_above_max():
    assert models.truncatedPowerLaw(11., 1., 1., 10.) == approx(0.)


# Test that truncatedPowerLaw integrates to unity
def test_normalize_truncatedPowerLaw():
    xs = jnp.linspace(1., 10., 10000)
    ps = models.truncatedPowerLaw(xs, 1., 1., 10.)
    assert jnp.trapezoid(ps, xs) == approx(1., rel=1e-4)


# Test expected numerical results from models.truncatedBrokenPowerLaw
@pytest.mark.parametrize("params, expected_result", [
    ([1.5, 1., -2., 1., 2., 4.], 3./7.),
    ([3., 1., -2., 1., 2., 4.], 16./63.),
    ([jnp.array([1.5, 3.]), 1., -2., 1., 2., 4.], [3./7., 16./63.]),
    ])
def test_evaluate_truncatedBrokenPowerLaw(params, expected_result):
    assert models.truncatedBrokenPowerLaw(*params) == approx(expected_result)


# Test that truncatedBrokenPowerLaw returns zero outside bounds
def test_truncatedBrokenPowerLaw_below_min():
    assert models.truncatedBrokenPowerLaw(0.5, 1., -2., 1., 2., 4.) == approx(0.)

def test_truncatedBrokenPowerLaw_above_max():
    assert models.truncatedBrokenPowerLaw(5., 1., -2., 1., 2., 4.) == approx(0.)


# Test that truncatedBrokenPowerLaw integrates to unity
def test_normalize_truncatedBrokenPowerLaw():
    xs = jnp.linspace(1., 4., 10000)
    ps = models.truncatedBrokenPowerLaw(xs, 1., -2., 1., 2., 4.)
    assert jnp.trapezoid(ps, xs) == approx(1., rel=1e-4)


# Test that truncatedBrokenPowerLaw reduces to truncatedPowerLaw when alpha1==alpha2
def test_truncatedBrokenPowerLaw_consistent_with_truncatedPowerLaw():
    xs = jnp.linspace(1., 10., 1000)
    p_broken = models.truncatedBrokenPowerLaw(xs, 1., 1., 1., 5., 10.)
    p_simple = models.truncatedPowerLaw(xs, 1., 1., 10.)
    assert jnp.allclose(p_broken, p_simple)


# Test expected numerical results from models.sigmoid
@pytest.mark.parametrize("params, expected_result", [
    ([1, -1, 4, 2, 0.5], -0.40398538),
    ([jnp.array([0, 1, 2]), -17, 10, -12, 5], [7.7543373, 8.133263, 8.452246]),
    ])
def test_evaluate_sigmoid(params, expected_result):
    assert models.sigmoid(*params) == approx(expected_result)


# Test asymptotic limits of models.sigmoid
def test_limit_sigmoid_high():
    ymin, ymax, xc, dx = -1, 4, 2, 0.5
    assert models.sigmoid(jnp.inf, ymin, ymax, xc, dx) == ymax


# Test asymptotic limits of models.sigmoid
def test_limit_sigmoid_low():
    ymin, ymax, xc, dx = -1, 4, 2, 0.5
    assert models.sigmoid(-jnp.inf, ymin, ymax, xc, dx) == ymin
