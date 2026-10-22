"""Create a portable deployment tar.gz with deterministic Linux permissions."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def _metadata(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    if info.isdir():
        info.mode = 0o755
    elif info.name.endswith(".sh"):
        info.mode = 0o755
    else:
        info.mode = 0o644
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output == source or source in output.parents:
        parser.error("output must be outside the deployment directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=6) as archive:
        archive.add(source, arcname="deployment", filter=_metadata)


if __name__ == "__main__":
    main()
