import numpy as np

from morphjax.verify import check_designs


def test_designs_hold_between_grid_points(setup):
    # Two points strictly between design points (airspeed, altitude and span
    # all off-grid), checked with the gains the runtime schedule interpolates.
    axes = (np.array([85.0, 105.0]), np.array([2500.0]), np.array([-6.25]))
    checks = check_designs(setup, axes)
    assert checks["gs"].off_grid == 2
    assert checks["gs"].stable == 2
    assert checks["lpv"].stable == 2
    assert checks["lpv"].certified == 2  # the common Lyapunov certificate extends here
    assert checks["gs"].certified is None
