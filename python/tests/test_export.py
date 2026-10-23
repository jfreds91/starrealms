import numpy as np
import pytest

import starrealms as sr

torch = pytest.importorskip("torch")

from starrealms._engine import srnn_forward  # noqa: E402
from starrealms.export import export  # noqa: E402
from starrealms.model import EntityConfig, EntityPolicy, ModelConfig, Policy  # noqa: E402


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


def test_rust_entity_forward_matches_torch(tmp_path):
    """The candle `EntityNet` drops padding tokens instead of masking them, so only legal
    logits (whose targets exist) are compared."""
    torch.manual_seed(0)
    policy = EntityPolicy(EntityConfig(d=32, layers=2, heads=4, ff=64)).eval()
    with torch.no_grad():  # zero-initialized residual projections would hide attention bugs
        for p in policy.parameters():
            p.add_(0.05 * torch.randn_like(p))
    path = tmp_path / "m.safetensors"
    export(policy, path)
    env = sr.VecEnv(6, ["core", "colony_wars", "frontiers"], 5)
    obs, mask, _, _ = env.reset()
    rng = np.random.default_rng(1)
    for _ in range(40):
        with torch.no_grad():
            logits, value = policy(torch.as_tensor(obs), torch.ones(mask.shape, dtype=torch.bool))
        for i in range(len(obs)):
            rl, rv = srnn_forward(str(path), obs[i])
            np.testing.assert_allclose(rl[mask[i]], logits[i].numpy()[mask[i]], rtol=1e-4, atol=1e-4)
            assert abs(rv - value[i].item()) < 1e-4
        obs, mask, *_ = env.step(np.array([rng.choice(np.flatnonzero(m)) for m in mask]))


def test_export_omits_privileged_critic(tmp_path):
    import json
    import struct

    torch.manual_seed(0)
    policy = EntityPolicy(EntityConfig(d=32, layers=1, heads=4, ff=64))
    path = tmp_path / "m.safetensors"
    export(policy, path)
    data = path.read_bytes()
    n = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8 : 8 + n])
    assert "critic_tok" not in header
    assert not any(k.startswith(("priv_type_emb.", "value_priv1.", "value_priv2.")) for k in header)


def test_export_checks_extension(tmp_path):
    with pytest.raises(ValueError):
        export(EntityPolicy(EntityConfig(d=16, layers=1, heads=2, ff=16)), tmp_path / "m.srnn")


@pytest.mark.parametrize("arch", ["mlp", "entity"])
def test_model_opponent_in_vecenv(tmp_path, arch):
    if arch == "mlp":
        policy, path = Policy(ModelConfig(embed=8, hidden=32, layers=1)), tmp_path / "m.srnn"
    else:
        policy, path = EntityPolicy(EntityConfig(d=16, layers=1, heads=2, ff=16)), tmp_path / "m.safetensors"
    export(policy, path)
    env = sr.VecEnv(4, ["core"], 0, f"model:{path}@1.0")
    _, mask, seat, learner = env.reset()
    for _ in range(200):
        assert (seat == learner).all()
        _, mask, seat, learner, *_ = env.step(np.array([np.flatnonzero(m)[0] for m in mask]))
