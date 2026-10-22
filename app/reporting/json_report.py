"""Предсказуемая сериализация VerificationReport."""

from __future__ import annotations

from pathlib import Path

from app.schemas import VerificationReport


def save_json_report(report: VerificationReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "report.json"
    target.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return target

