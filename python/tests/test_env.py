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


def test_privileged_matches_true_piles():
    g = sr.Game(["core", "colony_wars"], 7)
    layout = {name: (off, n) for name, off, n in sr.priv_layout()}
    assert sr.PRIV_IDS == sum(n for _, n in layout.values())
    for _ in range(40):
        if g.winner is not None:
            break
        priv = np.asarray(g.privileged())
        assert priv.shape == (sr.PRIV_IDS,) and priv.dtype == np.int16
        seat = int(g.active)
        opp_hand, _ = g.true_pile(1 - seat)
        _, my_deck = g.true_pile(seat)
        _, opp_deck = g.true_pile(1 - seat)
        for key, pile in (("opp_hand", opp_hand), ("my_next", my_deck), ("opp_next", opp_deck)):
            off, n = layout[key]
            got = [int(c) for c in priv[off : off + n] if c]
            assert got == [int(c) + 1 for c in list(pile)[:n]], key
        g.step(g.legal()[0])


def test_vecenv_privileged_roundtrip():
    env = sr.VecEnv(4, ["core"], 1, privileged=True)
    obs, mask, _, _, priv = env.reset()
    assert obs.shape[0] == 4 and priv.shape == (4, sr.PRIV_IDS) and priv.dtype == np.int16
    acts = np.array([np.flatnonzero(m)[0] for m in mask])
    out = env.step(acts)
    assert out[-1].shape == priv.shape
    wrapped = MultiEnv({"self": 2}, seed=1, privileged=True)
    res = wrapped.reset()
    assert res.priv.shape == (2, sr.PRIV_IDS)
    res = wrapped.step(np.array([np.flatnonzero(m)[0] for m in res.mask]))
    assert res.priv.shape == (2, sr.PRIV_IDS) and res.rewards.shape == (2, 2)


def test_named_scalars():
    assert len(sr.SCALAR_NAMES) == LAYOUT["scalars"].stop - LAYOUT["scalars"].start
    env = sr.VecEnv(8, ["core"], 2)
    obs, mask, *_ = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs, mask, *_ = env.step(np.array([rng.choice(np.flatnonzero(m)) for m in mask]))
        diff = obs[:, sr.SCALAR["my_authority"]] - obs[:, sr.SCALAR["opp_authority"]]
        np.testing.assert_allclose(obs[:, sr.SCALAR["authority_diff"]], diff, atol=1e-6)
