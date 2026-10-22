from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "tools" / "rag_ocr_once.py"
PLAN_PATH = Path(__file__).parents[1] / "tools" / "rag_ocr_once_plan.json"
SPEC = importlib.util.spec_from_file_location("rag_ocr_once", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manifest_plan_contains_exactly_119_unique_pages() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    tasks: list[dict[str, object]] = []
    for document in plan["documents"]:
        pages = MODULE._expand_pages(document["pages"])
        tasks.extend(
            {
                "document_id": document["document_id"],
                "version_id": "test-version",
                "page_number": page,
            }
            for page in pages
        )

    MODULE._validate_unique_tasks(tasks)
    assert len(plan["documents"]) == 29
    assert len(tasks) == plan["expected_pages"] == 119


def test_safe_child_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes configured root"):
        MODULE._safe_child(tmp_path, "../outside.pdf")


def test_completed_page_result_has_deterministic_storage_path(tmp_path: Path) -> None:
    result = MODULE._result_path(
        tmp_path,
        {"document_id": "doc-001", "version_id": "v1", "page_number": 7},
    )
    assert result == tmp_path / "pages" / "doc-001" / "v1" / "page_0007.json"


def test_plain_text_fallback_deduplicates_only_adjacent_overlap() -> None:
    assert MODULE._deduplicate_adjacent_lines(
        ["Первая строка\nОбщая строка", "Общая строка\nВторая строка"]
    ) == "Первая строка\nОбщая строка\nВторая строка"


def test_retry_partials_command_is_available() -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            "retry-partials",
            "--manifest",
            "manifest.jsonl",
            "--input-root",
            "input",
            "--output-root",
            "output",
            "--cache-root",
            "cache",
            "--shard-index",
            "1",
        ]
    )
    assert args.handler is MODULE.retry_partials
    assert args.shard_count == 2
