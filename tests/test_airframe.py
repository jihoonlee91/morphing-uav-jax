"""Mass model checks against numbers reported in Lee et al., Sensors 2023 (Sec. 2.1.2)."""

import jax.numpy as jnp
import numpy as np

from morphjax.airframe import Airframe, mass_properties

AF = Airframe()


def _mp(el, er, af=AF):
    m, cg, J = mass_properties(jnp.asarray(el), jnp.asarray(er), af)
    return float(m), np.asarray(cg), np.asarray(J)


def test_unmorphed_is_nominal():
    m, cg, J = _mp(0.0, 0.0)
    assert np.isclose(m, AF.mass)
    np.testing.assert_allclose(cg, AF.cg_nominal, atol=1e-12)
    np.testing.assert_allclose(J, AF.inertia_nominal, atol=1e-12)


def test_roll_inertia_drops_23_percent():
    # Paper: "The x-axis moment of inertia can be varied up to 23%".
    _, _, J = _mp(-25.0, -25.0)
    drop = 1.0 - J[0, 0] / AF.Ixx
    assert 0.225 < drop < 0.235


def test_asymmetric_cg_shift_is_1p3_percent_mac():
    # Paper: "The CM can be shifted up to 1.3% MAC off the plane of symmetry".
    _, cg, _ = _mp(-25.0, 0.0)
    shift = (cg[1] - float(AF.cg_nominal[1])) / AF.cbar * 100
    assert 1.25 < shift < 1.35


def test_fore_aft_shift_of_original_code_is_0p23_percent_mac():
    # Paper: "the CM moving back and forth up to 0.23% MAC" — reproduced by the
    # original MATLAB form of Eq. (24), which uses the dihedral angle in x.
    af = Airframe(tip_x_uses_sweep=False)
    _, cg, _ = _mp(-25.0, -25.0, af)
    shift = (cg[0] - float(af.cg_nominal[0])) / af.cbar * 100
    assert 0.20 < shift < 0.26


def test_inertia_is_symmetric_positive_definite_everywhere():
    for el in np.linspace(-25, 0, 6):
        for er in np.linspace(-25, 0, 6):
            _, _, J = _mp(el, er)
            np.testing.assert_allclose(J, J.T, atol=1e-12)
            assert np.linalg.eigvalsh(J).min() > 0


def test_mirror_symmetry_of_the_morphing_increment():
    # Swapping the wings mirrors the *change* in mass properties. (The NASA
    # airframe itself is slightly asymmetric: CG y-offset and nonzero Ixy.)
    _, cg_a, J_a = _mp(-20.0, -5.0)
    _, cg_b, J_b = _mp(-5.0, -20.0)
    y0, J0 = float(AF.cg_nominal[1]), np.asarray(AF.inertia_nominal)
    assert np.isclose(cg_a[1] - y0, -(cg_b[1] - y0))
    flip = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
    np.testing.assert_allclose(J_a - J0, flip * (J_b - J0), atol=1e-3)
