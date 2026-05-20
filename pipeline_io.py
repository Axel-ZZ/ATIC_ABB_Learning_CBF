"""
pipeline_io.py — shared path / config / index plumbing for the CBF pipeline.

Three stages, each writes to `runs/<stage>/<tag>/`:
    1. trajectories  (expert demos)
    2. datasets      (three-set split: expert_safe / unsafe / safe)
    3. cbfs/<method> (learned CBFs: nn or rff)

Each stage downstream of the first references its parent only by tag.

Conventions
-----------
- `tag = "<human-key-params>__<6-char-hash>"`  — human prefix is informative,
  hash detects when the underlying config actually changed.
- Every run directory contains `config.json` (full params + git sha) and
  `meta.json` (run-time summary).
- Every stage has a top-level `index.csv` for `pd.read_csv` scanning.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"

STAGE_TRAJ = "trajectories"
STAGE_DATA = "datasets"
STAGE_CBF = "cbfs"


# ── tag construction ──────────────────────────────────────────────────────

def _hash_config(config: Mapping[str, Any], n: int = 6) -> str:
    payload = json.dumps(config, sort_keys=True, default=repr).encode()
    return hashlib.sha1(payload).hexdigest()[:n]


def make_tag(human_prefix: str, config: Mapping[str, Any]) -> str:
    """`<prefix>__<hash>` — prefix names what you tuned, hash pins the rest."""
    safe = human_prefix.strip().replace("/", "-").replace(" ", "_")
    return f"{safe}__{_hash_config(config)}"


# ── path resolution ───────────────────────────────────────────────────────

def stage_root(stage: str, method: Optional[str] = None) -> Path:
    if stage == STAGE_CBF:
        if method is None:
            raise ValueError("stage='cbfs' requires method in {'nn','rff'}")
        return RUNS_DIR / STAGE_CBF / method
    return RUNS_DIR / stage


def run_dir(stage: str, tag: str, method: Optional[str] = None,
            create: bool = True) -> Path:
    d = stage_root(stage, method) / tag
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_existing(stage: str, tag: str,
                     method: Optional[str] = None) -> Path:
    d = stage_root(stage, method) / tag
    if not d.exists():
        raise FileNotFoundError(f"No run dir at {d}")
    return d


# ── config + meta ─────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def save_config(d: Path, config: Mapping[str, Any]) -> None:
    payload = {**config,
               "_git_sha": _git_sha(),
               "_saved_at": datetime.now().isoformat(timespec="seconds")}
    (d / "config.json").write_text(json.dumps(payload, indent=2, default=repr))


def load_config(d: Path) -> Dict[str, Any]:
    return json.loads((d / "config.json").read_text())


def save_meta(d: Path, meta: Mapping[str, Any]) -> None:
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=repr))


# ── flat index for rapid scanning ─────────────────────────────────────────

def append_index_row(stage: str, row: Mapping[str, Any],
                     method: Optional[str] = None) -> None:
    """
    Append/replace one row in `runs/<stage>/index.csv`
    (or `runs/cbfs/index.csv`, which combines nn + rff).
    Deduplicates by `tag`.
    """
    idx_path = (RUNS_DIR / STAGE_CBF / "index.csv"
                if stage == STAGE_CBF
                else stage_root(stage) / "index.csv")
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    fieldnames: list[str] = []
    if idx_path.exists():
        with idx_path.open() as f:
            r = csv.DictReader(f)
            fieldnames = list(r.fieldnames or [])
            rows = list(r)

    new_row = {"_ts": datetime.now().isoformat(timespec="seconds"),
               **dict(row)}
    for k in new_row:
        if k not in fieldnames:
            fieldnames.append(k)

    tag = new_row.get("tag")
    if tag is not None:
        rows = [r for r in rows if r.get("tag") != tag]
    rows.append(new_row)

    with idx_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
