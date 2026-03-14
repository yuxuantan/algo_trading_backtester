#!/usr/bin/env python3
"""Clean up run outputs.

Usage examples:
  python scripts/clean_runs.py          # remove all run output under runs/
  python scripts/clean_runs.py --keep 3  # keep last 3 run dirs per category

"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"
STRATEGIES_ROOT = RUNS_ROOT / "strategies"


def _discover_runs_by_workflow() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if not STRATEGIES_ROOT.exists():
        return out

    for manifest in STRATEGIES_ROOT.rglob("manifest.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        workflow = str(payload.get("workflow", "")).strip().lower()
        if not workflow:
            continue
        out.setdefault(workflow, []).append(manifest.parent)

    for workflow, run_dirs in out.items():
        run_dirs.sort(key=lambda p: p.stat().st_mtime)
        out[workflow] = run_dirs
    return out


def _prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()


def clean_runs(keep_last: int | None = None, dry_run: bool = True) -> None:
    """Delete run output directories.

    By default this function does not delete anything (dry-run mode). Set ``dry_run=False``
    to actually remove directories.

    Args:
        keep_last: how many latest directories to keep per subfolder (if None, do not delete anything).
        dry_run: if True, only print what would be removed.
    """

    if not STRATEGIES_ROOT.exists():
        print(f"No runs root found at {RUNS_ROOT}")
        return

    runs_by_workflow = _discover_runs_by_workflow()
    for workflow, dirs in runs_by_workflow.items():
        if keep_last is None:
            # nothing to delete unless explicitly requested
            continue

        if keep_last > 0:
            to_delete = dirs[:-keep_last]
        else:
            to_delete = dirs

        for d in to_delete:
            print(f"Removing {d}")
            if not dry_run:
                shutil.rmtree(d, ignore_errors=True)

    if not dry_run:
        _prune_empty_dirs(STRATEGIES_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean old run output directories.")
    parser.add_argument(
        "--keep",
        type=int,
        default=None,
        help="Keep the last N run directories for each run type. If omitted, no deletion occurs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually delete files. Without this flag, the script only prints what would be removed.",
    )
    args = parser.parse_args()

    clean_runs(keep_last=args.keep, dry_run=not args.force)
    if args.keep is None:
        print("No run directories were deleted; pass --keep N and --force to remove run data.")


if __name__ == "__main__":
    main()
