"""Совместимая точка входа для команды check."""

from __future__ import annotations

import sys

from app.cli import main


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        args = ["check", *args]
    raise SystemExit(main(args))

