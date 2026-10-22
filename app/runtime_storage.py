"""Безопасная очистка временных, debug- и cache-файлов runtime."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _files_under(paths: Iterable[Path], root: Path) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if not _is_within(path, root) or not path.exists():
            continue
        if path.is_file():
            result.append(path)
        else:
            result.extend(item for item in path.rglob("*") if item.is_file())
    return list(dict.fromkeys(result))


def cleanup_runtime(
    runtime_root: Path,
    *,
    include_temp: bool,
    include_debug: bool,
    include_cache: bool,
    older_than_days: int,
    dry_run: bool,
) -> dict[str, int]:
    root = runtime_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets: list[Path] = []
    if include_temp:
        targets.append(root / "temp")
    if include_cache:
        targets.append(root / "cache")
    if include_debug:
        targets.append(root / "debug")
        reports = root / "reports"
        if reports.exists():
            debug_names = {
                "debug_errors",
                "pages",
                "crops",
                "buyer_regions",
                "seal",
                "seal_candidates",
                "seal_profiles",
                "signature",
                "tables",
                "ocr",
            }
            targets.extend(
                path
                for path in reports.rglob("*")
                if path.is_dir() and path.name in debug_names
            )
    cutoff = time.time() - max(0, older_than_days) * 86400
    files = [
        path
        for path in _files_under(targets, root)
        if older_than_days <= 0 or path.stat().st_mtime < cutoff
    ]
    freed = sum(path.stat().st_size for path in files)
    if not dry_run:
        for path in files:
            path.unlink(missing_ok=True)
        for directory in sorted(
            {parent for path in files for parent in path.parents if _is_within(parent, root)},
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if directory != root:
                try:
                    directory.rmdir()
                except OSError:
                    pass
    return {"removed_files": len(files), "freed_bytes": freed}
