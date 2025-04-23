import pytest
import numpy as np
from pytest import approx
import jax.numpy as jnp
import gwpoptools.models as models


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
