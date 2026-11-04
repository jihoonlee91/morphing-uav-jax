import numpy as np

from morphjax.dynamics import Cmd, W, wind_derivative
from morphjax.trim import linearize, synthetic_to_true, trim, true_to_synthetic


def test_trim_converges_across_envelope(plant):
    for v in (60.0, 90.0, 120.0):
        for e in (-25.0, -12.5, 0.0):
            tp = trim(plant, v, 0.0, e)
            assert bool(tp.converged), (v, e, float(tp.residual))
            d = np.asarray(wind_derivative(plant, tp.xw, tp.cmd))
            assert np.abs(d[[W.V, W.ALPHA, W.BETA, W.P, W.Q, W.R, W.H]]).max() < 1e-8


def test_trim_trends(plant):
    slow, fast = trim(plant, 60.0, 0.0, -12.5), trim(plant, 120.0, 0.0, -12.5)
    assert float(slow.xw[W.ALPHA]) > float(fast.xw[W.ALPHA])  # slower -> higher alpha
    long_, short = trim(plant, 90.0, 0.0, 0.0), trim(plant, 90.0, 0.0, -25.0)
    assert float(short.xw[W.ALPHA]) > float(long_.xw[W.ALPHA])  # shorter span -> more alpha
    assert 2.0 < np.rad2deg(float(long_.xw[W.ALPHA])) < 6.0


def test_linearization_matches_finite_differences(plant):
    # The tables are piecewise linear and trim sits on breakpoints (beta = 0,
    # p = q = r = 0), where the Jacobian is the slope of the cell on the
    # positive side. The Kalviste rate blending is also piecewise in (p, r)
    # jointly, so columns are checked one at a time with positive steps.
    tp = trim(plant, 90.0, 5000.0, -12.5)
    A, B = linearize(plant, tp.xw, tp.cmd)
    f0 = wind_derivative(plant, tp.xw, tp.cmd)
    eps = 1e-7
    for j in range(18):
        f1 = wind_derivative(plant, tp.xw.at[j].add(eps), tp.cmd)
        # atol covers the O(eps) curvature term of a one-sided difference.
        np.testing.assert_allclose((f1 - f0) / eps, A[:, j], rtol=1e-4, atol=1e-4)
    for j in range(6):
        f1 = wind_derivative(plant, tp.xw, tp.cmd.at[j].add(eps))
        np.testing.assert_allclose((f1 - f0) / eps, B[:, j], rtol=1e-4, atol=1e-4)


def test_open_loop_modes(plant):
    tp = trim(plant, 90.0, 0.0, -12.5)
    A, _ = linearize(plant, tp.xw, tp.cmd)
    A = np.asarray(A)
    lon = [W.V, W.ALPHA, W.Q, W.THETA]
    lat = [W.BETA, W.P, W.R, W.PHI]
    ev_lon = np.linalg.eigvals(A[np.ix_(lon, lon)])
    ev_lat = np.linalg.eigvals(A[np.ix_(lat, lat)])
    sp = ev_lon[np.argmax(np.abs(ev_lon))]
    assert sp.real < 0 and 4 < abs(sp) < 12  # short period
    dr = ev_lat[np.argmax(np.abs(ev_lat.imag))]
    assert dr.real < 0 and -dr.real / abs(dr) < 0.2  # lightly damped Dutch roll


def test_symmetric_morphing_enters_lateral_model_only_through_trim(plant):
    tp = trim(plant, 90.0, 0.0, -12.5)
    _, B = linearize(plant, tp.xw, tp.cmd)
    B = np.asarray(B)
    # Asymmetric morphing drives the roll actuator state, symmetric drives eta_s.
    assert B[W.ETA_A, Cmd.ETA_A] > 0 and B[W.ETA_S, Cmd.ETA_S] > 0
    assert B[W.ETA_A, Cmd.ETA_S] == 0


def test_synthetic_airspeed_roundtrip():
    for h in (0.0, 5000.0, 10000.0):
        for v in (60.0, 90.0, 120.0):
            assert np.isclose(true_to_synthetic(synthetic_to_true(v, h), h), v)
    assert synthetic_to_true(120.0, 0.0) == 120.0
