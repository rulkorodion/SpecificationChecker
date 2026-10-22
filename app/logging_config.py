"""Консольный лог и ограниченный по размеру runtime-файл."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    max_size_mb: int = 10,
    backup_count: int = 5,
) -> None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_dir / "specification-checker.log",
                maxBytes=max(1, int(max_size_mb)) * 1024 * 1024,
                backupCount=max(1, int(backup_count)),
                encoding="utf-8",
            )
        )
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
