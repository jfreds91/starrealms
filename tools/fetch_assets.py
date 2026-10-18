#!/usr/bin/env python3
"""Download every external asset the project needs so it can run offline.

Sources
-------
* Official card gallery (starrealms.com/card-gallery): card database JSON,
  card images (WebP), and the faction / cost / defense icons.
* Community card spreadsheet JSON export: carries base defense and outpost
  flags, which the official JSON leaves null for most bases.

Outputs
-------
data/upstream/cards.json             official card database (committed)
data/upstream/star-realms-cards.json community spreadsheet export (committed)
assets/cards/<cardId>.webp           one image per in-scope card (gitignored)
assets/icons/<name>.webp             faction / cost / defense icons (gitignored)
assets/manifest.json                 sha256 + source URL per file; reruns skip
                                     anything whose hash already matches

Usage
-----
    python tools/fetch_assets.py            # in-scope sets only (default)
    python tools/fetch_assets.py --all      # every card in the gallery
    python tools/fetch_assets.py --verify   # re-hash files, report drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "upstream"
ASSET_DIR = ROOT / "assets"
MANIFEST = ASSET_DIR / "manifest.json"

GALLERY = "https://www.starrealms.com/card-gallery/"
CARDS_JSON_URL = GALLERY + "data/cards.json"
SPREADSHEET_JSON_URL = (
    "https://gist.githubusercontent.com/ineffably/e39c99b9d1f4fd00b98dfdbbd223b52e"
    "/raw/d6fb848427689fefbea1f618524ab31263271b69/star-realms-cards.json"
)
ICON_PATHS = [
    "images/icons/faction-blob.webp",
    "images/icons/faction-machine-cult.webp",
    "images/icons/faction-star-empire.webp",
    "images/icons/faction-trade-federation.webp",
    "images/icons/icon-cost.webp",
    "images/icons/icon-defense.webp",
]

IN_SCOPE_SETS = ("coreSet", "colonyWars", "frontiers")

USER_AGENT = "starrealms-sim-asset-fetch/1.0 (personal, non-commercial project)"
DELAY_SECONDS = 0.25
RETRIES = 4


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_error = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"failed after {RETRIES} attempts: {url}: {last_error}")


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"files": {}}


def save_manifest(manifest: dict) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def download(url: str, dest: Path, manifest: dict, *, expect: str | None = None) -> str:
    """Download `url` to `dest` unless the manifest says it is already current.

    Returns one of: "skipped", "downloaded", "updated".
    """
    rel = str(dest.relative_to(ROOT))
    entry = manifest["files"].get(rel)
    if entry and dest.exists() and sha256(dest.read_bytes()) == entry["sha256"]:
        return "skipped"

    data = fetch(url)
    if expect == "webp" and data[:4] != b"RIFF":
        raise RuntimeError(f"not a WebP (got HTML?): {url}")
    if expect == "json":
        json.loads(data)  # raise on garbage

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    manifest["files"][rel] = {
        "sha256": sha256(data),
        "bytes": len(data),
        "source": url,
    }
    time.sleep(DELAY_SECONDS)
    return "updated" if entry else "downloaded"


def in_scope_card_ids(db: dict, all_cards: bool) -> list[str]:
    if all_cards:
        return [c["id"] for c in db["cards"]]
    ids: set[str] = set()
    for s in db["sets"]:
        if s["id"] in IN_SCOPE_SETS:
            ids |= {x["cardId"] for x in s["cards"]}
    return sorted(ids)


def verify(manifest: dict) -> int:
    bad = 0
    for rel, entry in sorted(manifest["files"].items()):
        p = ROOT / rel
        if not p.exists():
            print(f"MISSING  {rel}")
            bad += 1
        elif sha256(p.read_bytes()) != entry["sha256"]:
            print(f"CHANGED  {rel}")
            bad += 1
    print(f"{len(manifest['files']) - bad} ok, {bad} problems")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="download images for every card, not just the three in-scope sets")
    ap.add_argument("--verify", action="store_true", help="check existing files against the manifest and exit")
    args = ap.parse_args()

    manifest = load_manifest()
    if args.verify:
        return verify(manifest)

    counts = {"skipped": 0, "downloaded": 0, "updated": 0}

    def note(result: str, label: str) -> None:
        counts[result] += 1
        if result != "skipped":
            print(f"{result:<10} {label}")

    # 1. Card databases
    note(download(CARDS_JSON_URL, DATA_DIR / "cards.json", manifest, expect="json"), "cards.json")
    note(download(SPREADSHEET_JSON_URL, DATA_DIR / "star-realms-cards.json", manifest, expect="json"), "star-realms-cards.json")
    save_manifest(manifest)

    db = json.loads((DATA_DIR / "cards.json").read_text())
    cards = {c["id"]: c for c in db["cards"]}

    # 2. Card images
    ids = in_scope_card_ids(db, args.all)
    failures: list[str] = []
    for i, cid in enumerate(ids, 1):
        card = cards[cid]
        images = [("", card["image"])]
        if card.get("backImage"):
            images.append(("-back", card["backImage"]))
        for suffix, rel in images:
            url = GALLERY + rel
            dest = ASSET_DIR / "cards" / f"{cid}{suffix}.webp"
            try:
                note(download(url, dest, manifest, expect="webp"), f"[{i}/{len(ids)}] {card['name']}{suffix}")
            except RuntimeError as e:
                failures.append(f"{cid}{suffix}: {e}")
                print(f"FAILED     {card['name']}{suffix}: {e}", file=sys.stderr)
        if i % 20 == 0:
            save_manifest(manifest)

    # 3. Icons
    for rel in ICON_PATHS:
        dest = ASSET_DIR / "icons" / Path(rel).name
        try:
            note(download(GALLERY + rel, dest, manifest, expect="webp"), Path(rel).name)
        except RuntimeError as e:
            failures.append(f"{rel}: {e}")
            print(f"FAILED     {rel}: {e}", file=sys.stderr)

    save_manifest(manifest)

    print(
        f"\n{len(ids)} cards; {counts['downloaded']} downloaded, "
        f"{counts['updated']} updated, {counts['skipped']} already current, {len(failures)} failed"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
