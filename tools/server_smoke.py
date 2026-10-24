"""End-to-end check of a running sr-server over HTTP and WebSockets.

    uv run tools/server_smoke.py [http://127.0.0.1:3030]

Two players create and join a game via the invite flow and play it to the end
with random legal moves. Checks hidden information, stale-action rejection,
spectators, a bot game, and the replay endpoint. Needs `requests` and `websockets`.
"""

import asyncio
import json
import random
import sys

import requests
import websockets

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:3030"
WS = BASE.replace("http", "ws", 1)


def post(path, body=None, token=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.post(f"{BASE}/api{path}", json=body, headers=h, timeout=10)
    assert r.ok, (path, r.status_code, r.text)
    return r.json()


def get(path, token=None, ok=True):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(f"{BASE}/api{path}", headers=h, timeout=10)
    if ok:
        assert r.ok, (path, r.status_code, r.text)
    return r


async def recv_error(ws):
    while True:
        m = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if m["type"] == "error":
            return m


async def recv_state(ws):
    while True:
        m = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if m["type"] == "state":
            return m
        assert m["type"] == "error", m
        return m


async def two_player_game():
    alice = post("/players", {"name": "Alice"})
    bob = post("/players", {"name": "Bob"})
    g = post("/games", {"sets": ["core", "colony_wars", "frontiers"], "opponent": "human", "seat": "first"}, alice["token"])
    gid = g["id"]
    assert g["seat"] == 0

    async with websockets.connect(f"{WS}/api/games/{gid}/ws?token={alice['token']}") as wa, \
            websockets.connect(f"{WS}/api/games/{gid}/ws") as spec:
        s = await recv_state(wa)
        assert s["game"]["status"] == "waiting" and s["seat"] == 0
        await recv_state(spec)
        # Alice gets no moves, and cannot move, before Bob joins.
        assert s["view"]["actions"] == []
        await wa.send(json.dumps({"type": "act", "action": 0, "n": s["n"]}))
        err = await recv_error(wa)
        assert "waiting" in err["message"], err

        spectated = []

        async def watch():
            async for raw in spec:
                spectated.append(json.loads(raw))

        watcher = asyncio.create_task(watch())
        assert post(f"/games/{gid}/join", token=bob["token"])["seat"] == 1
        async with websockets.connect(f"{WS}/api/games/{gid}/ws?token={bob['token']}") as wb:
            await wa.send(json.dumps({"type": "chat", "text": "  hello bob  "}))
            seen = False
            for _ in range(6):
                m = await recv_state(wb)
                if any(c["text"] == "hello bob" and c["name"] == "Alice" and c["seat"] == 0 for c in m.get("chat", [])):
                    seen = True
                    break
            assert seen, "bob should see alice's chat"
            socks = {0: wa, 1: wb}
            latest = {0: None, 1: None}
            rng = random.Random(7)
            moves = 0
            stale_checked = False
            while True:
                for seat in (0, 1):
                    # drain to the newest state for each seat
                    try:
                        while True:
                            m = json.loads(await asyncio.wait_for(socks[seat].recv(), 0.08))
                            if m["type"] == "state":
                                latest[seat] = m
                    except asyncio.TimeoutError:
                        pass
                a, b = latest[0], latest[1]
                if a is None or b is None or a["n"] != b["n"]:
                    continue
                assert a["view"]["opponent"]["hand"] is None and b["view"]["opponent"]["hand"] is None
                assert a["view"]["opponent"]["deck_contents"] is None
                if a["game"]["status"] == "finished":
                    break
                seat = a["view"]["active"]
                mine = latest[seat]
                other = latest[1 - seat]
                assert other["view"]["actions"] == [], "non-active seat must not get actions"
                acts = [x["id"] for x in mine["view"]["actions"]]
                if not stale_checked and moves > 5:
                    await socks[seat].send(json.dumps({"type": "act", "action": acts[0], "n": mine["n"] - 1}))
                    e = await recv_error(socks[seat])
                    assert "moved on" in e["message"], e
                    stale_checked = True
                if moves % 7 == 0 and any(x["group"] == "play" for x in mine["view"]["actions"]):
                    await socks[seat].send(json.dumps({"type": "play_all", "n": mine["n"]}))
                else:
                    await socks[seat].send(json.dumps({"type": "act", "action": rng.choice(acts), "n": mine["n"]}))
                latest = {0: None, 1: None}
                moves += 1
                assert moves < 5000
            await asyncio.sleep(0.5)
            watcher.cancel()
            spec_state = spectated[-1]
            assert all(m["view"]["you"]["hand"] is None and m["view"]["actions"] == [] for m in spectated)
            assert spec_state["game"]["status"] == "finished" and spec_state["seat"] is None and spec_state["view"]["you"]["hand"] is None
            print(f"two-player game {gid}: {moves} moves, winner seat {a['game']['winner']}, turn {a['view']['turn']}")

    r = get(f"/games/{gid}/replay").json()
    assert len(r["replay"]["actions"]) == a["n"]
    games = get("/games", alice["token"]).json()
    assert any(x["id"] == gid and x["status"] == "finished" for x in games)
    return gid


async def bot_game(spec_name):
    p = post("/players", {"name": "Solo"})
    g = post("/games", {"sets": ["core"], "opponent": spec_name, "seat": "second"}, p["token"])
    gid = g["id"]
    assert get(f"/games/{gid}/replay", ok=False).status_code == 403, "replay must be hidden while running"
    async with websockets.connect(f"{WS}/api/games/{gid}/ws?token={p['token']}") as ws:
        rng = random.Random(1)
        while True:
            s = await recv_state(ws)
            if s["type"] == "error":
                continue
            if s["game"]["status"] == "finished":
                assert s.get("chat") == [], "bots do not chat"
                break
            if s["view"]["active"] == s["seat"] and s["view"]["actions"]:
                await ws.send(json.dumps({"type": "act", "action": rng.choice(s["view"]["actions"])["id"], "n": s["n"]}))
    print(f"vs {spec_name} game {gid}: winner seat {s['game']['winner']} after {s['n']} actions")


async def resign_game():
    p = post("/players", {"name": "Quitter"})
    g = post("/games", {"sets": ["frontiers"], "opponent": "heuristic", "seat": "first"}, p["token"])
    async with websockets.connect(f"{WS}/api/games/{g['id']}/ws?token={p['token']}") as ws:
        await recv_state(ws)
        await ws.send(json.dumps({"type": "resign"}))
        while (s := await recv_state(ws))["game"]["status"] != "finished":
            pass
        assert s["game"]["winner"] == 1 and s["game"]["resigned"] == 0
    print("resignation ok")


async def main():
    assert requests.get(f"{BASE}/healthz").text == "ok"
    assert len(get("/catalog").json()) == 137
    assert requests.get(f"{BASE}/assets/cards/scout.webp").status_code == 200
    assert "<div id=\"root\">" in requests.get(f"{BASE}/g/whatever").text, "SPA fallback"
    gid = await two_player_game()
    opps = [o["spec"] for o in get("/opponents").json()]
    for o in opps:
        await bot_game(o)
    await resign_game()
    print("ok", gid)


asyncio.run(main())
