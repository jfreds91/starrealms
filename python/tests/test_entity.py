import numpy as np
import pytest

import starrealms as sr

torch = pytest.importorskip("torch")

from starrealms import tokens as tk  # noqa: E402
from starrealms.env import MultiEnv  # noqa: E402
from starrealms.league import FixedPool, League, Stack  # noqa: E402
from starrealms.model import EntityConfig, EntityPolicy, load  # noqa: E402


def _states(n: int, steps: int = 60, seed: int = 5):
    """Observations and masks from random play on all sets, collected along the way."""
    env = sr.VecEnv(n, ["core", "colony_wars", "frontiers"], seed)
    obs, mask, _, _ = env.reset()
    rng = np.random.default_rng(seed)
    out_o, out_m = [], []
    for _ in range(steps):
        out_o.append(obs)
        out_m.append(mask)
        obs, mask, *_ = env.step(np.array([rng.choice(np.flatnonzero(m)) for m in mask]))
    return np.concatenate(out_o), np.concatenate(out_m)


def small():
    torch.manual_seed(0)
    return EntityPolicy(EntityConfig(d=32, layers=2, heads=4, ff=64)).eval()


def test_every_action_has_a_target():
    covered = np.zeros(sr.N_ACTIONS, int)
    for name, (start, n) in tk.ACTIONS.items():
        covered[start : start + n] += 1
        if name in tk.POINTER_ACTIONS and tk.POINTER_ACTIONS[name] not in tk.CARD_ADDRESSED:
            assert tk.GROUPS[tk.POINTER_ACTIONS[name]] == n, name
        if tk.POINTER_ACTIONS.get(name) in tk.CARD_ADDRESSED:
            assert n == sr.N_CARDS, name
    assert (covered == 1).all()


def test_legal_actions_point_at_present_entities():
    """A legal slot action must target a non-padding token holding a card; a legal card-id
    action must find that card among the hand/discard tokens (barring discard overflow)."""
    obs, mask = _states(16)
    t = tk.tokenize(torch.as_tensor(obs))
    ids = {"own": t.own_ids, "opp": t.opp_ids, "row": t.row_ids}
    checked = 0
    for name, group in tk.POINTER_ACTIONS.items():
        start, n = tk.ACTIONS[name]
        rows, cols = np.nonzero(mask[:, start : start + n])
        for r, j in zip(rows, cols):
            if group in ids:
                assert ids[group][r, j] > 0, (name, j)
            elif group == "hand":
                s = t.hand_slot[r, j]
                assert s >= 0 and t.hand_ids[r, s] == j + 1, (name, j)
            elif group == "discard":
                s = t.discard_slot[r, j]
                assert s < 0 or t.discard_ids[r, s] == j + 1, (name, j)
            checked += 1
    assert checked > 1000


def test_forward_shapes_and_mask():
    obs, mask = _states(4, steps=10)
    m = small()
    logits, v = m(torch.as_tensor(obs), torch.as_tensor(mask))
    assert logits.shape == (len(obs), sr.N_ACTIONS) and v.shape == (len(obs),)
    assert (logits[~torch.as_tensor(mask)] <= -1e8).all()
    a, _, _ = m.act(torch.as_tensor(obs), torch.as_tensor(mask))
    assert mask[np.arange(len(obs)), a.numpy()].all()


def test_compact_matches_full():
    obs, mask = _states(8, steps=30)
    o, k = torch.as_tensor(obs), torch.as_tensor(mask)
    m = small()
    l1, v1 = m(o, k)
    assert sum(tk.tokenize(o, compact=True).sizes.values()) < tk.N_TOKENS
    m.compact = False
    l2, v2 = m(o, k)
    legal = k
    torch.testing.assert_close(l1[legal], l2[legal], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-5)


def test_padding_does_not_leak():
    """Changing features of an empty own slot must not change anything."""
    obs, mask = _states(4, steps=10)
    o = torch.as_tensor(obs)
    m = small()
    assert (tk.tokenize(o).own_ids[:, -1] == 0).all()
    o2 = o.clone()
    o2[:, sr.LAYOUT["own_slots"].stop - 1] = 1.0
    l1, v1 = m(o, torch.as_tensor(mask))
    l2, v2 = m(o2, torch.as_tensor(mask))
    torch.testing.assert_close(l1, l2)
    torch.testing.assert_close(v1, v2)


def test_save_load_roundtrip(tmp_path):
    m = small()
    m.save(tmp_path / "e.pt")
    m2 = load(tmp_path / "e.pt")
    assert isinstance(m2, EntityPolicy)
    obs, mask = _states(2, steps=5)
    o, k = torch.as_tensor(obs), torch.as_tensor(mask)
    torch.testing.assert_close(m(o, k)[0], m2(o, k)[0])


def test_league_stack_runs_entity_policies():
    league = League(size=3)
    for s in (0, 1):
        torch.manual_seed(s)
        league.add(EntityPolicy(EntityConfig(d=32, layers=1, heads=2, ff=32)), s)
    obs, mask = _states(3, steps=4)
    o, k = torch.as_tensor(obs), torch.as_tensor(mask)
    tags = np.arange(len(obs)) % 2
    stack = Stack(league.members)
    got = stack.forward(stack.params, o[None].expand(2, -1, -1), k[None].expand(2, -1, -1))
    for tag in (0, 1):
        torch.testing.assert_close(got[tag], league.members[tag](o, k)[0], rtol=1e-4, atol=1e-4)
    a = league.act(o, k, tags)
    assert mask[np.arange(len(obs)), a.numpy()].all()


def _priv_states(n: int = 4, steps: int = 6, seed: int = 3):
    env = sr.VecEnv(n, ["core", "colony_wars"], seed, privileged=True)
    obs, mask, _, _, priv = env.reset()
    rng = np.random.default_rng(seed)
    os_, ms, ps = [obs], [mask], [priv]
    for _ in range(steps):
        out = env.step(np.array([rng.choice(np.flatnonzero(m)) for m in mask]))
        obs, mask, priv = out[0], out[1], out[-1]
        os_.append(obs)
        ms.append(mask)
        ps.append(priv)
    return np.concatenate(os_), np.concatenate(ms), np.concatenate(ps)


def test_public_outputs_ignore_privileged_input():
    obs, mask, priv = _priv_states()
    m = small()
    # Attention residuals start at zero, so a fresh critic cannot see the extra tokens yet.
    with torch.no_grad():
        for blk in m.blocks:
            blk.proj.weight.add_(0.05 * torch.randn_like(blk.proj.weight))
            blk.ff2.weight.add_(0.05 * torch.randn_like(blk.ff2.weight))
    o, k, p = (torch.as_tensor(x) for x in (obs, mask, priv))
    l1, v1 = m(o, k)
    l2, v2, vp = m(o, k, p)
    torch.testing.assert_close(l1, l2, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-5)
    swapped = p.clone()
    swapped[:, 0], swapped[:, 1] = p[:, 1].clone(), p[:, 0].clone()
    _, _, vp_swap = m(o, k, swapped)
    torch.testing.assert_close(vp, vp_swap, rtol=1e-4, atol=1e-5)
    changed = p.clone()
    changed[:, 0] = torch.where(p[:, 0] == 1, torch.ones_like(p[:, 0]) * 2, torch.ones_like(p[:, 0]))
    _, _, vp_changed = m(o, k, changed)
    assert not torch.allclose(vp, vp_changed, atol=1e-5)


def test_privileged_value_does_not_train_the_policy_query():
    m = small().train()
    obs, mask, priv = _priv_states(n=2, steps=2)
    _, _, vp = m(*map(torch.as_tensor, (obs, mask, priv)))
    vp.sum().backward()
    assert m.value_priv2.weight.grad.abs().sum() > 0
    assert m.critic_tok.grad.abs().sum() > 0
    assert m.query.weight.grad is None


def test_old_checkpoint_loads_without_privileged_weights(tmp_path):
    m = small()
    m.save(tmp_path / "e.pt")
    ck = torch.load(tmp_path / "e.pt", weights_only=False)
    for key in list(ck["state_dict"]):
        if key == "critic_tok" or key.startswith(("priv_type_emb.", "value_priv1.", "value_priv2.")):
            del ck["state_dict"][key]
    torch.save(ck, tmp_path / "old.pt")
    m2 = load(tmp_path / "old.pt")
    obs, mask = _states(2, steps=3)
    o, k = torch.as_tensor(obs), torch.as_tensor(mask)
    torch.testing.assert_close(m(o, k)[0], m2(o, k)[0])
    ck["state_dict"]["nope.weight"] = torch.zeros(1)
    torch.save(ck, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError):
        load(tmp_path / "bad.pt")


def test_fixed_pool_plays_legal_moves(tmp_path):
    small().save(tmp_path / "a.pt")
    env = MultiEnv({f"model:{tmp_path / 'a.pt'}": 2}, seed=1)
    assert len(env.fixed_paths) == 1 and env.fixed.all()
    pool = FixedPool(env.fixed_paths, "cpu")
    res = env.reset()
    ids = np.zeros(res.obs.shape[0], np.int64)
    actions = pool.act(torch.as_tensor(res.obs), torch.as_tensor(res.mask), ids)
    assert res.mask[np.arange(len(actions)), actions.numpy()].all()
