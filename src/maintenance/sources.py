"""Daily upstream revision checks, separate from the age of historical observations."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

from src.maintenance.readiness import policy
from src.maintenance.storage import now_iso, timestamp

RELEASES = {
    "player_stats": ("nflverse/nflverse-data", "stats_player"),
    "team_stats": ("nflverse/nflverse-data", "stats_team"),
    "pbp": ("nflverse/nflverse-data", "pbp"),
    "rosters": ("nflverse/nflverse-data", "rosters"),
    "weekly_rosters": ("nflverse/nflverse-data", "weekly_rosters"),
    "snap_counts": ("nflverse/nflverse-data", "snap_counts"),
    "depth_charts": ("nflverse/nflverse-data", "depth_charts"),
    "injuries": ("nflverse/nflverse-data", "injuries"),
    "qbr": ("nflverse/nflverse-data", "espn_data"),
    "contracts": ("nflverse/nflverse-data", "contracts"),
    "players": ("nflverse/nflverse-data", "players"),
    "opportunity": ("ffverse/ffopportunity", "latest-data"),
}
DOCUMENTS = {
    "schedules": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "player_ids": "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv",
}


def fetch(url: str):
    request = Request(url, headers={"Accept": "application/vnd.github+json"})
    with urlopen(request, timeout=30) as response:
        body = response.read(16 * 1024 * 1024 + 1)
        if len(body) > 16 * 1024 * 1024:
            raise ValueError("Source index exceeds 16 MiB")
        return body, dict(response.headers.items())


def check_sources(previous: dict | None = None, *, fetcher=fetch) -> dict:
    previous = (previous or {}).get("sources", {})

    def check(item):
        name, url = item
        try:
            body, headers = fetcher(url)
            if name in RELEASES:
                payload = json.loads(body)
                assets = payload["assets"]
                if not isinstance(assets, list) or not assets:
                    raise ValueError("Source release has no assets")
                for asset in assets:
                    if (
                        not isinstance(asset.get("name"), str)
                        or type(asset.get("size")) is not int
                        or asset["size"] < 0
                    ):
                        raise ValueError("Invalid source asset schema")
                    timestamp(asset["updated_at"])
                # File revisions catch stat corrections even when season/week did not advance.
                revisions = sorted(
                    (a["name"], a["size"], a["updated_at"], a.get("digest")) for a in assets
                )
                value = {
                    "revision": hashlib.sha256(json.dumps(revisions).encode()).hexdigest(),
                    "latest_upstream_update": max(a["updated_at"] for a in assets),
                    "assets": [{"name": a[0], "updated_at": a[2]} for a in revisions],
                    "upstream_schema": {"status": "valid", "scope": "release inventory"},
                }
            else:
                required = (
                    {"season", "week", "home_team", "away_team"}
                    if name == "schedules"
                    else {"gsis_id", "pfr_id"}
                )
                columns = set(
                    csv.DictReader(io.StringIO(body.decode("utf-8-sig"))).fieldnames or []
                )
                if not required <= columns:
                    raise ValueError(f"Source CSV schema missing {sorted(required - columns)}")
                value = {
                    "revision": hashlib.sha256(body).hexdigest(),
                    "etag": headers.get("ETag"),
                    "upstream_schema": {
                        "status": "valid",
                        "scope": "source CSV",
                        "columns": sorted(columns),
                    },
                }
            value.update(
                url=url,
                status="unchanged"
                if previous.get(name, {}).get("revision") == value["revision"]
                else "changed",
            )
        except Exception as error:
            value = {"url": url, "status": "fetch_failed", "error": str(error)[:300]}
        value["policy"] = policy(name)
        return name, value

    urls = {
        name: f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        for name, (repo, tag) in RELEASES.items()
    } | DOCUMENTS
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = dict(pool.map(check, urls.items()))
    return {
        "checked_at": now_iso(),
        "status": "complete"
        if all(s["status"] != "fetch_failed" for s in sources.values())
        else "partial",
        "sources": sources,
        "coverage_note": "Revision checks do not establish game coverage; inference records live coverage and the weekly producer verifies historical inputs.",
    }
