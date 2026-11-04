import math

import jax
import jax.numpy as jnp
import numpy as np

from morphjax.airframe import DEG
from morphjax.control import lateral_law
from morphjax.dynamics import N_STATE, W
from morphjax.scenarios import morph_schedule, roll_reference, sample_scenario
from morphjax.sim import DT, departed, rollout


def _lat(sched):
    return lambda xw, pc, ig: lateral_law(sched, xw, pc, ig)


def _quiet(scen):
    return scen._replace(phi_cmd=jnp.zeros_like(scen.phi_cmd))


def test_trimmed_flight_stays_trimmed(setup):
    sched = setup.schedules["gs"]
    scen = sample_scenario(jax.random.PRNGKey(0), 250, 0.0)
    tr = rollout(setup.plant, sched, _lat(sched), _quiet(scen))
    xw = np.asarray(tr.xw)
    assert not bool(tr.failed.any())
    assert np.abs(xw[:, W.H] - float(scen.h)).max() < 1.0
    assert np.abs(np.rad2deg(xw[:, W.PHI])).max() < 0.2


def test_tracks_bank_steps_under_fast_morphing(setup):
    sched = setup.schedules["gs"]
    scen = sample_scenario(jax.random.PRNGKey(3), 500, math.inf)
    tr = rollout(setup.plant, sched, _lat(sched), scen)
    err = np.rad2deg(np.asarray(tr.xw)[:, W.PHI] - np.asarray(scen.phi_cmd))
    assert not bool(tr.failed.any())
    assert np.sqrt(np.mean(err**2)) < 5.0


def test_delay_changes_response_and_runs_are_deterministic(setup):
    sched = setup.schedules["gs"]
    scen = sample_scenario(jax.random.PRNGKey(1), 150, 0.0, noise=1.0)
    f = jax.jit(lambda s: rollout(setup.plant, sched, _lat(sched), s))
    a, b = f(scen), f(scen)
    np.testing.assert_array_equal(a.xw, b.xw)
    c = f(scen._replace(delay=jnp.asarray(3)))
    assert np.abs(np.asarray(c.xw) - np.asarray(a.xw)).max() > 1e-6


def test_departure_detection():
    xw = jnp.zeros(N_STATE).at[W.V].set(150.0)
    assert not bool(departed(xw))
    assert bool(departed(xw.at[W.PHI].set(90 * DEG)))
    assert bool(departed(xw.at[W.ALPHA].set(30 * DEG)))
    assert bool(departed(xw.at[W.V].set(jnp.nan)))


def test_scenario_generators():
    ref = np.asarray(roll_reference(jax.random.PRNGKey(0), 750))
    assert ref[0] == 0.0 and np.abs(ref).max() < 31 * DEG
    for rate in (0.0, 5.0, math.inf):
        s = np.asarray(morph_schedule(jax.random.PRNGKey(1), 750, jnp.asarray(rate)))
        assert s.min() >= -25.0 and s.max() <= 0.0
        if rate == 0.0:
            assert np.ptp(s) == 0.0
        elif math.isfinite(rate):
            assert np.abs(np.diff(s)).max() <= rate * DT + 1e-12
        else:
            assert np.ptp(s) > 10.0
