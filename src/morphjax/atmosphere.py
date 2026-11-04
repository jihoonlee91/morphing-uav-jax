"""1976 U.S. Standard Atmosphere, troposphere only (h < 36,089 ft)."""

from __future__ import annotations

import jax.numpy as jnp

RHO0 = 0.0023769  # slug/ft^3
T0 = 518.67  # deg R
LAPSE = 0.00356616  # deg R / ft
EXPONENT = 5.2559  # g / (R * lapse)


def temperature_ratio(h: jnp.ndarray) -> jnp.ndarray:
    return 1.0 - LAPSE * h / T0


def pressure_ratio(h: jnp.ndarray) -> jnp.ndarray:
    return temperature_ratio(h) ** EXPONENT


def density(h: jnp.ndarray) -> jnp.ndarray:
    """Air density in slug/ft^3 at geometric altitude ``h`` (ft)."""
    return RHO0 * temperature_ratio(h) ** (EXPONENT - 1.0)
