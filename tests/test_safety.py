import jax
import jax.numpy as jnp
import numpy as np

from morphjax.rl import RESIDUAL_SCALE
from morphjax.safety import (
    _relative_error,
    halfspace,
    make_safety_filter,
    project,
    sampled_decrease,
)
from morphjax.verify import dense_grid, interpolate_gains


def test_projection_enforces_the_halfspace_exactly(setup):
    f = make_safety_filter(setup)
    rng = np.random.default_rng(0)
    rho = (jnp.asarray(87.0), jnp.asarray(3300.0), jnp.asarray(-9.0))  # off-grid point
    for _ in range(50):
        dx = jnp.asarray(rng.normal(size=8)) * f.xs
        du = jnp.asarray(rng.uniform(-1, 1, size=3)) * RESIDUAL_SCALE
        a, b = halfspace(f, rho, dx)
        assert float(b) > 0  # the certificate holds here, so du = 0 is always feasible
        out = project(f, rho, dx, du) / f.us
        assert float(a @ out - b) <= 1e-9 * max(1.0, float(b))
        if float(a @ (du / f.us) - b) <= 0:
            np.testing.assert_allclose(out * f.us, du)  # inactive: untouched


def test_projection_is_jittable_and_zero_residual_passes(setup):
    f = make_safety_filter(setup)
    rho = (jnp.asarray(90.0), jnp.asarray(5000.0), jnp.asarray(-12.5))
    out = jax.jit(project)(f, rho, jnp.ones(8) * f.xs, jnp.zeros(3))
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_filtered_residual_keeps_the_decrease_on_the_true_model(setup):
    f = make_safety_filter(setup)
    axes = (np.array([90.0]), np.array([5000.0]), np.array([-12.5]))  # a design point
    dense, _ = dense_grid(setup, axes)
    gains = interpolate_gains(setup.grid, setup.lat_gains["lpv"], dense)
    out = sampled_decrease(f, dense, gains, {"proj": project, "raw": None}, samples=512)
    assert out["proj"]["decay"].min() == 1.0
    assert out["raw"]["decay"].min() < 1.0  # without the filter some residuals break it


def test_margin_makes_the_decrease_hold_off_grid(setup):
    # Off the design grid the plain projection can fail on the true model; the
    # margin covers the interpolation error there (B carries none: it is the
    # actuator input matrix, identical at every flight condition).
    f = make_safety_filter(setup)
    f_safe = f._replace(margin=0.35)
    axes = (np.array([65.0]), np.array([2500.0]), np.array([-18.75]))  # a cell centre
    dense, _ = dense_grid(setup, axes)
    gamma, b_err = _relative_error(f, dense)
    assert gamma.max() < f_safe.margin and b_err.max() < 1e-9
    gains = interpolate_gains(setup.grid, setup.lat_gains["lpv"], dense)
    out = sampled_decrease(f_safe, dense, gains, {"safe": project}, samples=512)
    assert out["safe"]["decay"].min() == 1.0
