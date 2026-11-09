import jax
import jax.numpy as jnp
import numpy as np

from morphjax.rl import PPOConfig, init_params
from morphjax.verify import check_designs, check_gains, dense_grid, policy_command, policy_gains

# Two points strictly between design points (airspeed, altitude and span all off-grid).
AXES = (np.array([85.0, 105.0]), np.array([2500.0]), np.array([-6.25]))


def test_designs_hold_between_grid_points(setup):
    checks = check_designs(setup, AXES)
    assert checks["gs"].off_grid == 2
    assert checks["gs"].stable == 2
    assert checks["lpv"].stable == 2
    assert checks["lpv"].certified == 2  # the common Lyapunov certificate extends here


def test_policy_gain_is_the_local_slope_of_the_command(setup):
    params = init_params(jax.random.PRNGKey(0), PPOConfig())
    dense, _ = dense_grid(setup, AXES)
    gains, saturated = policy_gains(params, dense)
    assert gains.shape == (2, 1, 1, 3, 8) and not saturated.any()
    rho = jnp.array([dense.v_syn[0], dense.alt[0], dense.eta_s[0]])
    dx = jnp.zeros(8).at[1].set(1e-6)  # small roll-rate deviation
    slope = (policy_command(params, dx, rho) - policy_command(params, -dx, rho)) / 2e-6
    np.testing.assert_allclose(slope, gains[0, 0, 0][:, 1], rtol=1e-5, atol=1e-8)
    checks = check_gains(setup, dense, gains)
    assert checks.max_real_eig.shape == (2, 1, 1)


def test_history_delay_check_reduces_to_the_plain_one():
    from morphjax.design import delayed_spectral_radius
    from morphjax.verify import spectral_radius_hist

    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8)) * 0.3 - np.eye(8)
    b, k = rng.normal(size=(8, 3)), rng.normal(size=(3, 8)) * 0.1
    for d in range(5):
        np.testing.assert_allclose(
            spectral_radius_hist(a, b, k, np.zeros((3, 3)), 0.02, d),
            delayed_spectral_radius(a, b, k, 0.02, d),
            rtol=1e-12,
        )
