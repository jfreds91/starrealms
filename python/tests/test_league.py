import numpy as np
import pytest

torch = pytest.importorskip("torch")
from starrealms.league import League, Ratings  # noqa: E402
from starrealms.model import ModelConfig, Policy  # noqa: E402


def small_policy():
    return Policy(ModelConfig(embed=4, hidden=8, layers=1))


def test_eviction_keeps_anchor_and_returns_evicted_tag():
    league = League(size=3)
    p = small_policy()
    assert [league.add(p, t) for t in (0, 25, 50)] == [None, None, None]
    assert league.add(p, 75) == 25
    assert league.add(p, 100) == 50
    assert sorted(league.members) == [0, 75, 100]
    league.record(50, 1.0)  # results for an evicted member are ignored
    assert 50 not in league.score


def test_pfsp_prefers_opponents_the_learner_loses_to():
    league = League(size=4)
    p = small_policy()
    for t in (0, 1, 2):
        league.add(p, t)
    for _ in range(300):
        league.record(0, 1.0)  # learner always beats 0
        league.record(2, 0.0)  # and always loses to 2
    rng = np.random.default_rng(0)
    draws = [league.sample(rng, p_self=0.0) for _ in range(3000)]
    counts = {t: draws.count(t) for t in (0, 1, 2)}
    assert counts[2] > counts[1] > counts[0]
    assert all(league.sample(rng, p_self=1.0) == -1 for _ in range(20))


def test_batched_league_matches_members():
    """`League.act` runs every member in one vmapped pass; each row must follow its own member."""
    from starrealms import N_ACTIONS, OBS_DIM
    from starrealms.league import Stack

    league = League(size=4)
    for t in (0, 25, 50):
        torch.manual_seed(t)
        league.add(small_policy(), t)
    rng = np.random.default_rng(0)
    obs = torch.as_tensor(rng.random((9, OBS_DIM)), dtype=torch.float32)
    mask = torch.ones((9, N_ACTIONS), dtype=torch.bool)
    tags = np.array([50, 0, 25, 0, 0, 50, 25, 25, 0])
    stack = Stack(league.members)
    k = np.searchsorted(stack.tags, tags)
    pos = np.array([sum(k[:i] == k[i]) for i in range(len(k))])
    o = torch.zeros((3, 4, OBS_DIM))
    o[k, pos] = obs
    got = stack.forward(stack.params, o, torch.ones((3, 4, N_ACTIONS), dtype=torch.bool))[k, pos]
    for i, t in enumerate(tags):
        want = league.members[t](obs[i : i + 1], mask[i : i + 1])[0][0]
        torch.testing.assert_close(got[i], want, rtol=1e-5, atol=1e-5)
    a = league.act(obs, mask, tags)
    assert a.shape == (9,)


def test_elo_matches_win_rate():
    r = Ratings(anchor="b")
    r.games("a", "b", 760, 0, 240)
    # 76% expected score is about +200 Elo
    assert r.rating("a") == pytest.approx(400 * np.log10(0.76 / 0.24), abs=5)
    assert r.rating("b") == 0
    assert [row["name"] for row in r.table()] == ["a", "b"]


def test_bots_calibrate_in_order():
    r = Ratings()
    r.calibrate_bots(["random", "greedy", "heuristic"], ["core"], games=100)
    assert r.rating("heuristic") == 0 > r.rating("greedy") > r.rating("random")


def test_result_order_does_not_matter():
    results = [("a", "b", 60, 5, 35), ("b", "c", 70, 0, 30), ("a", "c", 50, 10, 40)]
    fwd, rev = Ratings(anchor="c"), Ratings(anchor="c")
    for x in results:
        fwd.games(*x)
    for x in reversed(results):
        rev.games(*x)
    for n in "abc":
        assert fwd.rating(n) == pytest.approx(rev.rating(n), abs=1e-6)


def test_checkpoint_chain_follows_heuristic_win_rate():
    """The Phase A failure: each checkpoint is evaluated against the bots, the first snapshot,
    and its predecessor (about even). Ratings must still rise with the heuristic win rate,
    and a checkpoint that beats the heuristic most of the time must outrank it."""
    r = Ratings()
    r.games("heuristic", "greedy", 270, 0, 30)
    r.games("heuristic", "random", 300, 0, 0)
    r.games("greedy", "random", 297, 0, 3)
    win_rates = [0.05, 0.13, 0.25, 0.4, 0.5, 0.6, 0.66, 0.72, 0.75, 0.77]
    for k, p in enumerate(win_rates):
        name = f"u{k:05d}"
        w = round(500 * p)
        r.games(name, "heuristic", w, 0, 500 - w)
        r.games(name, "greedy", min(500, 300 + 25 * k), 0, max(0, 200 - 25 * k))
        if k > 0:
            r.games(name, "u00000", 500, 0, 0)
            r.games(name, f"u{k - 1:05d}", 255, 0, 245)
    ratings = [r.rating(f"u{k:05d}") for k in range(len(win_rates))]
    assert ratings == sorted(ratings), ratings
    assert r.rating("u00009") > r.rating("heuristic") + 100
