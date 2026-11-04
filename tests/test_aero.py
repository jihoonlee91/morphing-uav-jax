import jax.numpy as jnp
import numpy as np

from morphjax.aero import MIRROR, build_aero_model, coefficients
from morphjax.airframe import DEG, Airframe
from morphjax.data import AERO_SHA256, fetch_aero_database
from morphjax.interp import interp

AF = Airframe()


def _c(model, alpha=4.0, beta=0.0, pqr=(0.0, 0.0, 0.0), surf=(0.0, 0.0, 0.0), el=0.0, er=0.0):
    return np.asarray(
        coefficients(
            model, alpha * DEG, beta * DEG, 150.0, jnp.asarray(pqr), jnp.asarray(surf),
            jnp.asarray(el), jnp.asarray(er), AF.b, AF.cbar,
        )
    )  # fmt: skip


def test_database_is_pinned(tables):
    import hashlib

    assert hashlib.sha256(fetch_aero_database().read_bytes()).hexdigest() == AERO_SHA256
    assert tables.basic.data.shape == (32, 27, 6)


def test_rudder_mirror_matches_gtm_rule(tables):
    # GTM_DesignSim: a positive rudder is the mirror image of the negative one.
    model = build_aero_model(tables)
    for a, b, d in [(4.0, 3.0, 12.0), (10.0, -6.0, 25.0), (2.0, 0.0, 0.0)]:
        pos = np.asarray(interp(model.rudder, a, b, d))
        neg = MIRROR * np.asarray(interp(tables.rudder, a, -b, -d))
        np.testing.assert_allclose(pos, neg, atol=1e-12)


def test_control_sign_conventions(plant):
    m = plant.aero
    base = _c(m)
    assert _c(m, surf=(10, 0, 0))[4] < base[4]  # elevator TE down -> nose down
    assert _c(m, surf=(0, 10, 0))[3] > base[3]  # aileron > 0 -> roll right
    assert _c(m, surf=(0, 0, 10))[5] < base[5]  # rudder TE left -> nose left


def test_morphing_increments(plant):
    m = plant.aero
    base = _c(m)
    full = _c(m, el=-25.0, er=-25.0)
    half = _c(m, el=-12.5, er=-12.5)
    np.testing.assert_allclose(half - base, 0.5 * (full - base), atol=1e-12)  # Eq. (41) is linear
    assert full[2] > base[2]  # retracting both tips loses lift (CZ = -lift)
    left = _c(m, el=-25.0)
    assert left[3] < base[3]  # left tip in -> left wing drops (paper: -dCl)


def test_morphing_mirror_symmetry(plant):
    # Swapping the wings and flipping the sideslip mirrors the increment, Eq. (42).
    m = plant.aero
    inc_a = _c(m, beta=3.0, el=-20.0, er=-5.0) - _c(m, beta=3.0)
    inc_b = _c(m, beta=-3.0, el=-5.0, er=-20.0) - _c(m, beta=-3.0)
    np.testing.assert_allclose(inc_a, MIRROR * inc_b, atol=1e-12)


def test_rate_damping_opposes_rates(plant):
    m = plant.aero
    base = _c(m)
    assert _c(m, pqr=(0.5, 0, 0))[3] < base[3]
    assert _c(m, pqr=(0, 0.3, 0))[4] < base[4]
    assert _c(m, pqr=(0, 0, 0.3))[5] < base[5]
