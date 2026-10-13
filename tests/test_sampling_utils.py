import pytest
import numpy as np
import jax.numpy as jnp
import gwpoptools.sampling_utils as sampling_utils


# Test that perEventMonteCarloIntegralAndDiagnostics matches
# monteCarloIntegralAndDiagnostics applied to each event in turn
@pytest.mark.parametrize("n_per_event", [
    [100, 100, 100],
    [300, 400, 500],
    [250, 1, 1000, 30],
    ])
def test_perEventMonteCarloIntegralAndDiagnostics_consistent(n_per_event):

    # Ragged synthetic weights, with sorted event labels
    rng = np.random.default_rng(0)
    weights = jnp.array(rng.exponential(size=sum(n_per_event)))
    event_ids = jnp.array(np.repeat(np.arange(len(n_per_event)), n_per_event))
    bounds = np.cumsum([0] + n_per_event)

    # Per-event results from the single-event function
    expected = [sampling_utils.monteCarloIntegralAndDiagnostics(weights[i:j], j-i)
                for i, j in zip(bounds[:-1], bounds[1:])]
    expected = [jnp.array(x) for x in zip(*expected)]

    result = sampling_utils.perEventMonteCarloIntegralAndDiagnostics(
        weights, event_ids, jnp.array(n_per_event, dtype=float))

    for r, e in zip(result, expected):
        assert jnp.allclose(r, e)
