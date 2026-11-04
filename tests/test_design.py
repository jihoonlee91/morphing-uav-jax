import numpy as np

from morphjax import design


def _closed_loop_eigs(setup, gains, model):
    g = setup.grid
    out = []
    for idx in np.ndindex(*g.residual.shape):
        a, b = model(g.A[idx], g.B[idx])
        out.append(np.linalg.eigvals(a + b @ gains[idx]))
    return np.array(out)


def test_all_63_points_trimmed(setup):
    assert setup.grid.residual.shape == (7, 3, 3)
    assert setup.grid.residual.max() < 1e-8


def test_lqr_designs_stabilize_every_point(setup):
    g = setup.grid
    lon = _closed_loop_eigs(setup, design.longitudinal_gains(g), design.longitudinal_model)
    assert lon.real.max() < 0
    for gains in (design.lateral_gains_gs(g), design.lateral_gains_fixed(g)):
        assert _closed_loop_eigs(setup, gains, design.lateral_model).real.max() < 0


def test_lpv_certificate(setup):
    lpv = setup.lpv
    assert lpv.lyapunov_margin < 0  # common Lyapunov matrix verified at all points
    assert np.linalg.eigvalsh(lpv.X).min() > 0
    eig = _closed_loop_eigs(setup, lpv.gains, design.lateral_model)
    assert eig.real.max() < -0.1 + 1e-6  # decay-rate constraint
    assert np.abs(eig).max() < 30.0 + 1e-3  # disk constraint


def test_delayed_spectral_radius_scalar_case():
    # x' = u, u = -k x applied one sample late: z^2 - z + k dt = 0, stable iff k dt < 1.
    a, b, dt = np.zeros((1, 1)), np.ones((1, 1)), 0.02
    assert design.delayed_spectral_radius(a, b, np.array([[-0.9 / dt]]), dt, 1) < 1
    assert design.delayed_spectral_radius(a, b, np.array([[-1.1 / dt]]), dt, 1) > 1
    # Without delay the same gain is a deadbeat-ish stable loop.
    assert design.delayed_spectral_radius(a, b, np.array([[-1.1 / dt]]), dt, 0) < 1


def test_higher_control_cost_buys_delay_margin(setup):
    g = setup.grid
    cheap = design.lateral_gains_gs(g, design.LatWeights(effort=1.0))
    costly = design.lateral_gains_gs(g, design.LatWeights(effort=0.0625))
    assert not design.lateral_delay_margin_ok(g, cheap, 0.02, 3)
    assert design.lateral_delay_margin_ok(g, costly, 0.02, 3)


def test_fixed_gain_is_the_nominal_design(setup):
    g = setup.grid
    fixed, gs = design.lateral_gains_fixed(g), design.lateral_gains_gs(g)
    idx = design.nominal_index(g)
    np.testing.assert_allclose(fixed[0, 0, 0], gs[idx])
