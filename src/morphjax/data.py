"""Download, verify, and load the NASA GTM-T2 aerodynamic database.

The table file is not redistributed with this repository. It is fetched from
NASA's official ``nasa/GTM_DesignSim`` repository at a pinned commit and
checked against a SHA-256 digest, so every run uses byte-identical data.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import scipy.io

NASA_COMMIT = "9717143270144aca1f5d38d7c24c0fce678d1589"
AERO_URL = (
    "https://raw.githubusercontent.com/nasa/GTM_DesignSim/"
    f"{NASA_COMMIT}/gtm_design/config/T2_polynomial_aerodatabase.mat"
)
AERO_SHA256 = "7f34845a54bcecdd788f91226b6be6b386612b3cb691a42538854b4c49466f5c"
AERO_FILENAME = "T2_polynomial_aerodatabase.mat"

# Index of the "Left Wingtip (25%) Off" case in dC6_damage. The morphing model
# reads this case as "left wingtip retracted by 25 % of the semispan".
WINGTIP_CASE = 3


def cache_dir() -> Path:
    """Directory for downloaded data (override with ``MORPHJAX_DATA``)."""
    env = os.environ.get("MORPHJAX_DATA")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "morphing-flight-jax"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_aero_database(force: bool = False) -> Path:
    """Return a local path to the verified aero database, downloading it once."""
    target = cache_dir() / AERO_FILENAME
    if target.exists() and not force and _sha256(target) == AERO_SHA256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    req = urllib.request.Request(AERO_URL, headers={"User-Agent": "morphing-flight-jax"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        out.write(resp.read())
    digest = _sha256(tmp)
    if digest != AERO_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Aero database checksum mismatch: got {digest}, want {AERO_SHA256}")
    tmp.replace(target)
    return target


class Table(NamedTuple):
    """A rectilinear lookup table: ``data`` has shape ``(*grid_sizes, n_out)``."""

    grids: tuple[jnp.ndarray, ...]
    data: jnp.ndarray


class AeroTables(NamedTuple):
    """GTM-T2 aero tables used by :mod:`morphjax.aero` (angles in degrees)."""

    basic: Table  # (alpha, beta) -> [CX CY CZ Cl Cm Cn]
    sym: Table  # (alpha, beta) -> [CY Cl Cn] symmetrising increment
    elevator: Table  # (alpha, beta, stab, elev) -> [CX CZ Cm]
    aileron: Table  # (alpha, beta, right aileron) -> 6 coefficients
    rudder: Table  # (alpha, beta, rudder <= 0) -> 6 coefficients
    roll_damp: Table  # (alpha, p_hat) -> [CY Cl Cn]
    pitch_damp: Table  # (alpha, q_hat) -> [CX CZ Cm]
    yaw_damp: Table  # (alpha, r_hat) -> [CY Cl Cn]
    rotary: Table  # (alpha, omega_hat, beta) -> 6 coefficients
    wingtip: Table  # (alpha, beta) -> 6 coefficients, left 25 % wingtip removed
    gear_up: Table  # (alpha,) -> [CX CZ Cm], landing gear retracted


def _grid(x: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(np.asarray(x, dtype=np.float64).ravel())


def load_aero_tables(path: Path | None = None) -> AeroTables:
    """Load the ``.mat`` database into JAX arrays."""
    path = path or fetch_aero_database()
    m = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)

    def arr(x: np.ndarray) -> jnp.ndarray:
        return jnp.asarray(np.asarray(x, dtype=np.float64))

    bas, sym = m["C6_bas"], m["dC3_sym"]
    ele, ail, rud = m["dC3_ele"], m["dC6_ail"], m["dC6_rud"]
    p, q, r, w = m["dC3_p"], m["dC3_q"], m["dC3_r"], m["dC6_w"]
    dmg, lgr = m["dC6_damage"].bas, m["dC3_lgr"]
    return AeroTables(
        basic=Table((_grid(bas.alpha), _grid(bas.beta)), arr(bas.data)),
        sym=Table((_grid(sym.alpha), _grid(sym.beta)), arr(sym.data)),
        elevator=Table(
            (_grid(ele.alpha), _grid(ele.beta), _grid(ele.stab), _grid(ele.elev)), arr(ele.data)
        ),
        aileron=Table((_grid(ail.alpha), _grid(ail.beta), _grid(ail.ail)), arr(ail.data)),
        rudder=Table((_grid(rud.alpha), _grid(rud.beta), _grid(rud.rud)), arr(rud.data)),
        roll_damp=Table((_grid(p.alpha), _grid(p.phat)), arr(p.data)),
        pitch_damp=Table((_grid(q.alpha), _grid(q.qhat)), arr(q.data)),
        yaw_damp=Table((_grid(r.alpha), _grid(r.rhat)), arr(r.data)),
        rotary=Table((_grid(w.alpha), _grid(w.what), _grid(w.beta)), arr(w.data)),
        wingtip=Table(
            (_grid(dmg.alpha), _grid(dmg.beta)), arr(np.asarray(dmg.data)[:, :, WINGTIP_CASE, :])
        ),
        gear_up=Table((_grid(lgr.alpha),), arr(np.asarray(lgr.data)[:, 0, :])),
    )
