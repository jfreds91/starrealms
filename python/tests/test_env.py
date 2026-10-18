import json

import numpy as np
import pytest

import starrealms as sr
from starrealms.env import LAYOUT, MultiEnv


def test_constants_and_layout():
    assert sr.N_ACTIONS == 622
    assert min(s.start for s in LAYOUT.values()) == 0
    assert LAYOUT["ids"].stop == sr.OBS_DIM
    assert len(sr.cards()) == sr.N_CARDS


def test_vecenv_random_rollout():
    env = sr.VecEnv(16, ["core", "colony_wars", "frontiers"], 3)
    obs, mask, seat, learner = env.reset()
    assert obs.shape == (16, sr.OBS_DIM) and obs.dtype == np.float32
    assert mask.shape == (16, sr.N_ACTIONS) and mask.dtype == bool
    rng = np.random.default_rng(0)
    games = 0
    for _ in range(3000):
        assert mask.any(axis=1).all()
        acts = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
        obs, mask, seat, learner, rewards, done, winner, turns, done_learner = env.step(acts)
        for i in np.flatnonzero(done):
            games += 1
            if winner[i] >= 0:
                assert rewards[i, winner[i]] == 1 and rewards[i, 1 - winner[i]] == -1
        assert (rewards[~done] == 0).all()
    assert games > 0


def test_illegal_action_rejected():
    env = sr.VecEnv(2)
    _, mask, _, _ = env.reset()
    bad = np.array([np.flatnonzero(~m)[0] for m in mask])
    with pytest.raises(ValueError):
        env.step(bad)


def test_bot_opponent_only_returns_learner_turns():
    env = MultiEnv({"heuristic": 8}, seed=1)
    res = env.reset()
    for _ in range(500):
        assert (res.seat == res.learner_seat).all()
        acts = [np.flatnonzero(m)[0] for m in res.mask]
        res = env.step(np.array(acts))


def test_game_replay_roundtrip():
    g = sr.Game(["core"], 42)
    while g.winner is None:
        g.step(g.bot_action("heuristic"))
    r = g.replay()
    g2 = sr.Game.from_replay(r)
    assert g2.winner == g.winner and g2.turn == g.turn
    view = json.loads(g.view(0))
    assert view["outcome"] is not None


def test_arena():
    a, b, d = sr.arena("heuristic", "random", 50)
    assert a + b + d == 50 and a > 40


def test_gae_per_seat():
    from starrealms.ppo import compute_gae

    # one env, seats alternate 0,1,0,1; game ends after step 3 with seat 1 winning
    values = np.zeros((4, 1), np.float32)
    seats = np.array([[0], [1], [0], [1]])
    dones = np.array([[False], [False], [False], [True]])
    rewards = np.zeros((4, 1, 2), np.float32)
    rewards[3, 0] = [-1, 1]
    adv = compute_gae(values, seats, dones, rewards, np.zeros((4, 1), np.float32),
                      np.zeros(1, np.float32), np.zeros(1, np.int64), np.zeros(1, np.float32),
                      gamma=1.0, lam=1.0, shaping=0.0)
    assert adv[:, 0].tolist() == [-1, 1, -1, 1]
