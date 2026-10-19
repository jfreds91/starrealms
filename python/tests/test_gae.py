"""`compute_gae` against a slow reference that walks each (env, seat) chain explicitly."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tyro")

from starrealms.ppo import compute_gae  # noqa: E402


def reference_gae(values, seats, dones, rewards, phi, v_last, seat_last, phi_last, gamma, lam, shaping):
    """Per step, find the same seat's next decision in the same game and apply the GAE recursion.

    Successor of a decision, in order of precedence:
    * the same seat's next decision in the same game: bootstrap from it and continue the recursion;
    * the game ended: terminal reward for this seat, zero value and potential, recursion stops;
    * the rollout ended with this seat to move: bootstrap from `v_last` / `phi_last`, recursion stops;
    * the rollout ended with the other seat to move: unknown, so bootstrap from the step's own value.
    """
    T, N = values.shape
    adv = np.zeros((T, N))
    for i in range(N):
        game = np.concatenate([[0], np.cumsum(dones[:-1, i])])
        for s in (0, 1):
            steps = [t for t in range(T) if seats[t, i] == s]
            for k in reversed(range(len(steps))):
                t = steps[k]
                later = [u for u in steps[k + 1 :] if game[u] == game[t]]
                end = next((u for u in range(t, T) if dones[u, i]), None)
                r, cont_adv = 0.0, 0.0
                if later:
                    u = later[0]
                    nv, nphi, cont_adv = values[u, i], phi[u, i], adv[u, i]
                elif end is not None:
                    nv, nphi, r = 0.0, 0.0, rewards[end, i, s]
                elif seat_last[i] == s:
                    nv, nphi = v_last[i], phi_last[i]
                else:
                    nv, nphi = values[t, i], phi[t, i]
                delta = r + shaping * (gamma * nphi - phi[t, i]) + gamma * nv - values[t, i]
                adv[t, i] = delta + gamma * lam * cont_adv
    return adv


def run(values, seats, dones, rewards, phi, v_last, seat_last, phi_last, gamma=0.99, lam=0.95, shaping=0.0):
    args = (values.astype(np.float32), seats.astype(np.int64), dones.astype(bool), rewards.astype(np.float32),
            phi.astype(np.float32), v_last.astype(np.float32), seat_last.astype(np.int64),
            phi_last.astype(np.float32))
    got = compute_gae(*args, gamma, lam, shaping)
    want = reference_gae(*args, gamma, lam, shaping)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    return got


def col(*xs):
    return np.array(xs)[:, None]


def test_game_ends_on_opponent_move():
    # seats 0,1,0,1; seat 1's last action wins. Seat 0's last decision still gets -1.
    rewards = np.zeros((4, 1, 2))
    rewards[3, 0] = [-1, 1]
    adv = run(np.zeros((4, 1)), col(0, 1, 0, 1), col(0, 0, 0, 1), rewards, np.zeros((4, 1)),
              np.zeros(1), np.zeros(1), np.zeros(1), gamma=1.0, lam=1.0)
    assert adv[:, 0].tolist() == [-1, 1, -1, 1]


def test_new_game_starts_mid_rollout():
    # game 1: seat 0 plays twice and wins at t=1. game 2 starts at t=2 and runs past the rollout end
    # with seat 0 to move next, so seat 0's t=2 decision bootstraps from v_last.
    values = col(0.1, 0.2, 0.3, 0.4)
    rewards = np.zeros((4, 1, 2))
    rewards[1, 0] = [1, -1]
    adv = run(values, col(0, 0, 0, 1), col(0, 1, 0, 0), rewards, np.zeros((4, 1)),
              np.array([0.5]), np.array([0]), np.zeros(1), gamma=0.9, lam=0.8)
    assert adv[1, 0] == pytest.approx(1 - 0.2)
    assert adv[0, 0] == pytest.approx((0.9 * 0.2 - 0.1) + 0.9 * 0.8 * (1 - 0.2))
    assert adv[2, 0] == pytest.approx(0.9 * 0.5 - 0.3)
    # seat 1 at t=3: the rollout ends with seat 0 to move, so its successor is unknown
    assert adv[3, 0] == pytest.approx(0.9 * 0.4 - 0.4)


def test_shaping_is_potential_based():
    # With gamma=1 and lam=1, shaping telescopes: a seat's first advantage changes by -phi_0.
    rng = np.random.default_rng(0)
    seats = col(0, 0, 1, 0, 1, 1, 0)
    dones = col(0, 0, 0, 0, 0, 0, 1)
    rewards = np.zeros((7, 1, 2))
    rewards[6, 0] = [1, -1]
    phi = rng.normal(size=(7, 1))
    base = run(np.zeros((7, 1)), seats, dones, rewards, phi, np.zeros(1), np.zeros(1), np.zeros(1),
               gamma=1.0, lam=1.0, shaping=0.0)
    shaped = run(np.zeros((7, 1)), seats, dones, rewards, phi, np.zeros(1), np.zeros(1), np.zeros(1),
                 gamma=1.0, lam=1.0, shaping=0.5)
    assert shaped[0, 0] - base[0, 0] == pytest.approx(-0.5 * phi[0, 0])
    assert shaped[2, 0] - base[2, 0] == pytest.approx(-0.5 * phi[2, 0])


def random_rollout(rng, T, N):
    seats = np.zeros((T, N), np.int64)
    dones = np.zeros((T, N), bool)
    rewards = np.zeros((T, N, 2))
    for i in range(N):
        s = int(rng.integers(2))
        for t in range(T):
            seats[t, i] = s
            if rng.random() < 0.12:
                dones[t, i] = True
                w = rng.integers(-1, 2)  # -1 draw, else winning seat
                rewards[t, i] = [0, 0] if w == -1 else ([1, -1] if w == 0 else [-1, 1])
                s = int(rng.integers(2))
            elif rng.random() < 0.4:  # turns are runs of several decisions
                s = 1 - s
    seat_last = rng.integers(0, 2, N)
    return seats, dones, rewards, seat_last


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("gamma,lam,shaping", [(0.99, 0.95, 0.0), (0.995, 0.9, 0.3), (1.0, 1.0, 1.0)])
def test_matches_reference_on_random_rollouts(seed, gamma, lam, shaping):
    rng = np.random.default_rng(seed)
    T, N = 48, 6
    seats, dones, rewards, seat_last = random_rollout(rng, T, N)
    run(rng.normal(size=(T, N)), seats, dones, rewards, rng.normal(size=(T, N)),
        rng.normal(size=N), seat_last, rng.normal(size=N), gamma, lam, shaping)
