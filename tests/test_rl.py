import jax
import jax.numpy as jnp
import numpy as np

from morphjax.rl import (
    Batch,
    PPOConfig,
    gae,
    init_params,
    load_params,
    log_prob,
    make_train,
    save_params,
)


def test_gae_matches_reference_loop():
    rng = np.random.default_rng(0)
    T, N, gamma, lam = 12, 3, 0.97, 0.9
    rew, val = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    term = np.zeros((T, N), bool)
    term[5, 1] = True
    last = rng.normal(size=N)
    batch = Batch(None, None, None, jnp.asarray(val), jnp.asarray(rew), None, jnp.asarray(term))
    adv, ret = gae(batch, jnp.asarray(last), gamma, lam)

    ref = np.zeros((T, N))
    for n in range(N):
        a = 0.0
        for t in reversed(range(T)):
            nv = last[n] if t == T - 1 else val[t + 1, n]
            nt = 0.0 if term[t, n] else 1.0
            a = rew[t, n] + gamma * nv * nt - val[t, n] + gamma * lam * nt * a
            ref[t, n] = a
    np.testing.assert_allclose(adv, ref, atol=1e-12)
    np.testing.assert_allclose(ret, ref + val, atol=1e-12)


def test_log_prob_is_gaussian():
    cfg = PPOConfig()
    p = init_params(jax.random.PRNGKey(0), cfg)
    obs, act = jnp.zeros((1, 12)), jnp.zeros((1, 3))
    mu = np.zeros(3)  # output layer bias is zero and obs is zero
    sd = np.exp(cfg.init_log_std)
    expect = np.sum(-0.5 * ((0 - mu) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi))
    np.testing.assert_allclose(log_prob(p, obs, act)[0], expect, atol=1e-6)


def test_params_roundtrip(tmp_path):
    cfg = PPOConfig()
    p = init_params(jax.random.PRNGKey(1), cfg)
    save_params(tmp_path / "p.npz", p)
    q = load_params(tmp_path / "p.npz", cfg)
    for a, b in zip(jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(q), strict=True):
        np.testing.assert_array_equal(a, b)


def test_one_ppo_update_runs(setup):
    cfg = PPOConfig(num_envs=4, steps=40, updates=1, epochs=1, minibatches=2)
    init_state, update = make_train(setup.plant, setup.schedules["gs"], cfg)
    params, opt = init_state(jax.random.PRNGKey(0))
    new, _, stats = update(params, opt, jax.random.PRNGKey(1))
    assert all(np.isfinite(float(v)) for v in stats.values())
    old, upd = jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(new)
    moved = [np.abs(np.asarray(a) - np.asarray(b)).max() for a, b in zip(old, upd, strict=True)]
    assert max(moved) > 0


def test_history_policy_sees_the_pending_command_and_runs_delayed(setup, tmp_path):
    import json

    from morphjax.rl import LEGACY_ACT_SCALE, load_policy, make_policy_lateral, training_scenario
    from morphjax.sim import rollout

    cfg = PPOConfig(steps=30, delay=1, history=1)
    params = init_params(jax.random.PRNGKey(2), cfg)
    assert params["pi"][0]["w"].shape[0] == 15  # 12 + one pending 3-vector
    scen = training_scenario(jax.random.PRNGKey(5), cfg)
    assert int(scen.delay) == 1
    lat = make_policy_lateral(setup.schedules["gs"], params, cfg=cfg)
    traj = rollout(setup.plant, setup.schedules["gs"], lat, scen)
    assert np.isfinite(np.asarray(traj.xw)).all()
    # the saved interface round-trips; a config without act_scale is a legacy policy
    save_params(tmp_path / "h.npz", params)
    (tmp_path / "h.json").write_text(json.dumps({"config": cfg._asdict()}))
    _, back = load_policy(tmp_path / "h.npz")
    assert back.history == 1 and tuple(back.act_scale) == (30.0, 30.0, 12.5)
    save_params(tmp_path / "old.npz", init_params(jax.random.PRNGKey(0), PPOConfig()))
    (tmp_path / "old.json").write_text(json.dumps({"config": {"hidden": 128}}))
    assert tuple(load_policy(tmp_path / "old.npz")[1].act_scale) == LEGACY_ACT_SCALE


def test_cleanrl_style_update_normalizes_and_trains(setup, tmp_path):
    from morphjax.rl import CLEANRL, load_policy

    cfg = PPOConfig(num_envs=4, steps=40, updates=1, **{**CLEANRL, "minibatches": 2, "epochs": 1})
    init_state, update = make_train(setup.plant, setup.schedules["gs"], cfg)
    params, opt = init_state(jax.random.PRNGKey(0))
    new, _, stats = update(params, opt, jax.random.PRNGKey(1))
    assert all(np.isfinite(float(v)) for v in stats.values())
    assert float(new["obs_norm"]["count"]) > 1.0 and float(new["ret_norm"]["count"]) > 1.0
    assert not np.allclose(new["pi"][0]["w"], params["pi"][0]["w"])
    import json

    save_params(tmp_path / "c.npz", new)
    (tmp_path / "c.json").write_text(json.dumps({"config": cfg._asdict()}))
    back, _ = load_policy(tmp_path / "c.npz")
    np.testing.assert_array_equal(back["obs_norm"]["mean"], new["obs_norm"]["mean"])


def test_round3_linear_policy_and_masked_buffer(setup):
    from morphjax.design import LAT_X_SCALE
    from morphjax.rl import OBS_SCALE, OBS_SHIFT, observation, policy_mean, training_scenario

    cfg = PPOConfig(linear=True)
    p = init_params(jax.random.PRNGKey(0), cfg)
    k = np.random.default_rng(0).normal(size=(3, 8))
    p["K"] = jnp.broadcast_to(jnp.asarray(k.reshape(-1)), p["K"].shape)
    dx = np.random.default_rng(1).normal(size=8) * LAT_X_SCALE * 0.1
    raw = np.concatenate([dx, [-12.5, 90.0, 5000.0, dx[3]]])
    obs = (jnp.asarray(raw) - OBS_SHIFT) / OBS_SCALE
    np.testing.assert_allclose(policy_mean(p, obs), k @ (dx / LAT_X_SCALE), rtol=1e-9)
    # delay-only randomization stays within 0..3 samples
    d = jax.vmap(lambda kk: training_scenario(kk, PPOConfig(steps=20, delay_rand=3)).delay)(
        jax.random.split(jax.random.PRNGKey(0), 64)
    )
    assert int(d.min()) == 0 and int(d.max()) == 3
    # masked buffer: with delay 1 only the newest command is observed
    from morphjax.sim import initial_carry

    sc = training_scenario(jax.random.PRNGKey(2), PPOConfig(steps=20))
    c = initial_carry(setup.plant, sc)
    buf = c.buf.at[:, 1].add(3.0)  # aileron 3 deg off trim in every slot
    from morphjax.dynamics import body_to_wind

    o, _ = observation(setup.schedules["gs"], body_to_wind(c.x), 0.0, 0.0, buf, 3,
                       (30.0, 30.0, 12.5), 1, True)  # fmt: skip
    pend = np.asarray(o[12:]).reshape(3, 3)
    assert pend[0, 0] != 0 and np.all(pend[1:] == 0)
