"""Use the model as a library: trim, linearize, design, simulate.

uv run python examples/quickstart.py
"""

import jax
import numpy as np

import morphjax  # noqa: F401  (enables float64)
from morphjax import design, pipeline
from morphjax.control import lateral_law
from morphjax.dynamics import W
from morphjax.scenarios import sample_scenario
from morphjax.sim import rollout
from morphjax.trim import linearize, trim

setup = pipeline.build()  # downloads the NASA tables on first use, builds all designs
plant = setup.plant

# 1. Trim at 90 kt, 3,000 ft, both wingtips half retracted, and linearize.
tp = trim(plant, 90.0, 3000.0, -12.5)
A, B = linearize(plant, tp.xw, tp.cmd)
print(f"trim: alpha {np.rad2deg(float(tp.xw[W.ALPHA])):.2f} deg, residual {float(tp.residual):.1e}")
a_lat, _ = design.lateral_model(np.asarray(A), np.asarray(B))
print("open-loop lateral poles:", np.round(np.linalg.eigvals(a_lat[:4, :4]), 2))

# 2. Fly a random 15 s bank-angle scenario with the gain-scheduled LQR while
#    the symmetric span steps between short and long (rate = inf).
sched = setup.schedules["gs"]
scen = sample_scenario(jax.random.PRNGKey(42), 750, float("inf"))
traj = jax.jit(lambda s: rollout(plant, sched, lambda x, p, i: lateral_law(sched, x, p, i), s))(
    scen
)
err = np.rad2deg(np.asarray(traj.xw)[:, W.PHI] - np.asarray(scen.phi_cmd))
rmse = np.sqrt(np.mean(err**2))
print(f"roll-tracking RMS error {rmse:.2f} deg, lost control: {bool(traj.failed.any())}")

# 3. Everything is differentiable: sensitivity of trim angle of attack to span.
dalpha = jax.grad(lambda e: trim(plant, 90.0, 3000.0, e).xw[W.ALPHA])(-12.5)
print(f"d(alpha_trim)/d(eta_s) = {np.rad2deg(float(dalpha)):.4f} deg per % span")
