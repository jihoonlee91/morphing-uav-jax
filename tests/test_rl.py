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
