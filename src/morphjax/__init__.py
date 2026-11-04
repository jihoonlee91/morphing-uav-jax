"""Differentiable model and controllers for an asymmetric span-morphing UAV."""

import jax

# Trim and Jacobian linearization need double precision: the trim residuals are
# driven to ~1e-10 and the LMI synthesis is sensitive to small model errors.
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
