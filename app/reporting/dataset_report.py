"""Сводка пакетной оценки."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_dataset_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=lambda value: getattr(value, "value", str(value)),
        ),
        encoding="utf-8",
    )
    lines = [
        "СВОДКА ПАКЕТНОЙ ПРОВЕРКИ",
        "=" * 60,
        f"Найдено документов: {summary['documents_found']}",
        f"Обработано: {summary['processed']}",
        f"Технических ошибок: {summary['technical_errors']}",
        f"APPROVE: {summary['decision_counts'].get('APPROVE', 0)}",
        f"RETURN: {summary['decision_counts'].get('RETURN', 0)}",
        f"MANUAL_REVIEW: {summary['decision_counts'].get('MANUAL_REVIEW', 0)}",
        f"Общая точность: {summary.get('overall_field_accuracy', 0.0) * 100:.2f}%",
        "",
        "Документы:",
    ]
    lines.extend(
        f"- {item['document']}: {item.get('decision', 'ERROR')}"
        + (f" ({item['error']})" if item.get("error") else "")
        for item in summary["results"]
    )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
