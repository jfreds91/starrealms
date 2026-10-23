"""End-to-end check that PPO learns: a short run must clearly beat the random bot.

    uv run pytest -m slow
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tyro")

from starrealms.evaluate import evaluate  # noqa: E402
from starrealms.model import EntityConfig, EntityPolicy, Policy, load  # noqa: E402
from starrealms.ppo import Args, main  # noqa: E402


def _tiny(tmp_path, name, **kw):
    return Args(
        run_name=name,
        out=str(tmp_path),
        groups="self:4",
        total_steps=32,
        num_steps=4,
        minibatch=8,
        epochs=1,
        eval_every=1000,
        snapshot_every=1000,
        eval_games=1,
        eval_opponents="random",
        arch="entity",
        d=32,
        heads=4,
        tf_layers=1,
        ff=64,
        device="cpu",
        bf16=False,
        seed=1,
        **kw,
    )


def test_privileged_smoke(tmp_path):
    out = main(_tiny(tmp_path, "priv", privileged=True))
    policy = load(out / "latest.pt", "cpu")
    assert policy.critic_tok.shape[0] == 32


def test_critic_only_leaves_the_policy(tmp_path):
    init = EntityPolicy(EntityConfig(d=32, layers=1, heads=4, ff=64))
    init.save(tmp_path / "init.pt")
    before = {k: v.clone() for k, v in init.state_dict().items() if k.startswith(("query.", "key.", "blocks.", "card_id."))}
    out = main(_tiny(tmp_path, "crit", privileged=True, critic_only=True, init=str(tmp_path / "init.pt")))
    after = load(out / "latest.pt", "cpu")
    for k, v in before.items():
        torch.testing.assert_close(after.state_dict()[k], v)
    assert not torch.equal(after.value_priv2.weight, init.value_priv2.weight)


@pytest.mark.slow
def test_ppo_beats_random(tmp_path):
    args = Args(
        run_name="learns",
        out=str(tmp_path),
        groups="random:128",
        total_steps=3e5,
        eval_every=10_000,
        eval_games=200,
        eval_opponents="random",
        snapshot_every=10_000,
        hidden=256,
    )
    torch.manual_seed(0)
    untrained = Policy().to(args.device).eval()
    before = evaluate(untrained, "random", 400, ["core"], args.device, greedy=False, seed=98)
    assert before.winrate < 0.65, str(before)

    out = main(args)
    policy = Policy.load(out / "latest.pt", args.device).eval()
    after = evaluate(policy, "random", 1000, ["core"], args.device, seed=99)
    print(f"untrained {before}\ntrained   {after}")
    assert after.interval()[0] > 0.9, str(after)
