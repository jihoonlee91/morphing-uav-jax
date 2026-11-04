import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from morphjax.data import Table
from morphjax.interp import interp, slice_table


def _random_table(rng, sizes, n_out=3):
    grids = tuple(np.sort(rng.choice(np.linspace(-10, 10, 50), n, replace=False)) for n in sizes)
    data = rng.normal(size=(*sizes, n_out))
    return Table(tuple(jnp.asarray(g) for g in grids), jnp.asarray(data)), grids, data


def test_matches_scipy_inside_grid():
    rng = np.random.default_rng(0)
    for sizes in [(7,), (5, 6), (4, 5, 3)]:
        table, grids, data = _random_table(rng, sizes)
        ref = RegularGridInterpolator(grids, data)
        f = jax.jit(jax.vmap(lambda p, t=table: interp(t, *p)))
        pts = np.stack([rng.uniform(g[0], g[-1], 200) for g in grids], axis=-1)
        np.testing.assert_allclose(f(jnp.asarray(pts)), ref(pts), atol=1e-12)


def test_linear_extrapolation_and_clipping():
    g = jnp.array([0.0, 1.0, 3.0])
    table = Table((g,), (2.0 * g + 1.0)[:, None])
    assert float(interp(table, 5.0)[0]) == 11.0  # linear data extrapolates exactly
    assert float(interp(table, -2.0)[0]) == -3.0
    assert float(interp(table, 5.0, extrap="clip")[0]) == 7.0
    assert float(interp(table, -2.0, extrap="clip")[0]) == 1.0


def test_gradient_is_the_cell_slope():
    g = jnp.array([0.0, 1.0, 3.0])
    table = Table((g,), jnp.array([[0.0], [1.0], [5.0]]))
    slope = jax.grad(lambda x: interp(table, x)[0])
    assert float(slope(0.5)) == 1.0
    assert float(slope(2.0)) == 2.0


def test_gradient_at_a_breakpoint_is_the_full_cell_slope_in_clip_mode():
    # Regression: jnp.clip has gradient 0.5 at its bounds, which halved the
    # rate-damping derivatives at p = q = r = 0 in the linearized models.
    g = jnp.array([-1.0, 0.0, 1.0])
    table = Table((g,), jnp.array([[2.0], [0.0], [-2.0]]))
    for mode in ("linear", "clip"):
        assert float(jax.grad(lambda x, m=mode: interp(table, x, extrap=m)[0])(0.0)) == -2.0


def test_slice_table_picks_breakpoint():
    rng = np.random.default_rng(1)
    table, grids, data = _random_table(rng, (4, 5))
    s = slice_table(table, axis=1, value=float(grids[1][2]))
    np.testing.assert_array_equal(s.data, data[:, 2])
