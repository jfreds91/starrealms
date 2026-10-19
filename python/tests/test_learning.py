"""End-to-end check that PPO learns: a short run must clearly beat the random bot.

    uv run pytest -m slow
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tyro")

from starrealms.evaluate import evaluate  # noqa: E402
from starrealms.model import Policy  # noqa: E402
from starrealms.ppo import Args, main  # noqa: E402


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
