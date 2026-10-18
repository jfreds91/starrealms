import numpy as np
import pytest

import starrealms as sr

torch = pytest.importorskip("torch")

from starrealms._engine import srnn_forward  # noqa: E402
from starrealms.export import export  # noqa: E402
from starrealms.model import ModelConfig, Policy  # noqa: E402


def _observations(n: int) -> np.ndarray:
    env = sr.VecEnv(n, ["core", "colony_wars", "frontiers"], 5)
    obs, mask, _, _ = env.reset()
    rng = np.random.default_rng(1)
    for _ in range(40):
        obs, mask, *_ = env.step(np.array([rng.choice(np.flatnonzero(m)) for m in mask]))
    return obs


def test_rust_forward_matches_torch(tmp_path):
    torch.manual_seed(0)
    policy = Policy(ModelConfig(embed=8, hidden=64, layers=2)).eval()
    path = tmp_path / "m.srnn"
    export(policy, path)
    obs = _observations(6)
    with torch.no_grad():
        h = policy.trunk(policy.features(torch.as_tensor(obs)))
        logits, value = policy.pi(h).numpy(), policy.v(h).squeeze(-1).numpy()
    for i in range(len(obs)):
        rl, rv = srnn_forward(str(path), obs[i])
        np.testing.assert_allclose(rl, logits[i], rtol=1e-4, atol=1e-4)
        assert abs(rv - value[i]) < 1e-4


def test_model_opponent_in_vecenv(tmp_path):
    path = tmp_path / "m.srnn"
    export(Policy(ModelConfig(embed=8, hidden=32, layers=1)), path)
    env = sr.VecEnv(4, ["core"], 0, f"model:{path}@1.0")
    _, mask, seat, learner = env.reset()
    for _ in range(200):
        assert (seat == learner).all()
        _, mask, seat, learner, *_ = env.step(np.array([np.flatnonzero(m)[0] for m in mask]))
