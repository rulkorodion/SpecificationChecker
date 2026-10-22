"""CLI: check, check-mock и evaluate-dataset."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.checks.rule_engine import RuleEngine
from app.config import load_config
from app.enums import FinalDecision
from app.evaluation.dataset_evaluation import evaluate_dataset
from app.exceptions import InputValidationError, SpecificationCheckerError
from app.logging_config import configure_logging
from app.mock_loader import load_mock_extracted
from app.document.loader import DocumentLoader
from app.ocr.base import OCREngine
from app.ocr.cache import OCRCache
from app.ocr.batch import run_ocr_batch
from app.ocr.factory import build_ocr_engine
from app.ocr.monkey_compare import compare_saved_monkey
from app.ocr.monkey_settings import resolve_monkeyocr_settings
from app.reporting.console import RULE
from app.runtime_storage import cleanup_runtime
from app.schemas import ReferenceData
from app.service import SpecificationCheckerService

EXIT_CODES = {
    FinalDecision.APPROVE: 0,
    FinalDecision.RETURN: 10,
    FinalDecision.MANUAL_REVIEW: 20,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Локальная проверка спецификаций")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Проверить документ")
    check.add_argument("--document", type=Path, required=True)
    check.add_argument("--reference", type=Path, required=True)
    check.add_argument(
        "--ground-truth",
        type=Path,
        help="Необязательная разработческая оценка; для production-проверки не нужна",
    )
    check.add_argument("--config", type=Path)
    check.add_argument("--output-dir", type=Path)
    check.add_argument("--save-debug", action="store_true")
    check.add_argument(
        "--debug-level", choices=["none", "errors", "full"], default="none"
    )
    check.add_argument("--reuse-ocr-cache", action="store_true")
    check.add_argument("--force-ocr", action="store_true")
    check.add_argument("--ocr-cache-dir", type=Path)
    check.add_argument("--ocr-language", default=None)
    check.add_argument(
        "--ocr-backend",
        choices=["monkeyocr-v2"],
        default=None,
    )
    check.add_argument("--device", choices=["auto", "cpu", "gpu:0"], default=None)
    check.add_argument("--log-level", default="INFO")

    mock = sub.add_parser("check-mock", help="Проверить Rule Engine без OCR")
    mock.add_argument("--extracted", type=Path, required=True)
    mock.add_argument("--reference", type=Path, required=True)
    mock.add_argument("--config", type=Path)
    mock.add_argument("--output-dir", type=Path, default=Path("output/mock"))
    mock.add_argument("--log-level", default="INFO")

    dataset = sub.add_parser("evaluate-dataset", help="Пакетный прогон")
    dataset.add_argument("--documents-dir", type=Path, required=True)
    dataset.add_argument("--reference-dir", type=Path, required=True)
    dataset.add_argument("--ground-truth-dir", type=Path, required=True)
    dataset.add_argument("--output-dir", type=Path, required=True)
    dataset.add_argument("--config", type=Path)
    dataset.add_argument("--save-debug", action="store_true")
    dataset.add_argument(
        "--debug-level", choices=["none", "errors", "full"], default="none"
    )
    dataset.add_argument("--reuse-ocr-cache", action="store_true")
    dataset.add_argument("--force-ocr", action="store_true")
    dataset.add_argument("--ocr-cache-dir", type=Path)
    dataset.add_argument(
        "--include-ids",
        help="Только указанные ID, например 25-29,31",
    )
    dataset.add_argument("--ocr-language", default=None)
    dataset.add_argument(
        "--ocr-backend",
        choices=["monkeyocr-v2"],
        default=None,
    )
    dataset.add_argument("--device", choices=["auto", "cpu", "gpu:0"], default=None)
    dataset.add_argument("--log-level", default="INFO")

    cache_import = sub.add_parser(
        "import-ocr-cache", help="Импортировать сохранённые OCR JSON без OCR-модели"
    )
    cache_import.add_argument("--document", type=Path, required=True)
    cache_import.add_argument("--saved-output", type=Path, required=True)
    cache_import.add_argument("--config", type=Path)
    cache_import.add_argument("--ocr-cache-dir", type=Path)
    cache_import.add_argument("--log-level", default="INFO")

    cleanup = sub.add_parser("cleanup", help="Безопасно очистить runtime-файлы")
    cleanup.add_argument("--runtime-dir", type=Path, default=Path("runtime"))
    cleanup.add_argument("--older-than-days", type=int, default=0)
    cleanup.add_argument("--temp", action="store_true")
    cleanup.add_argument("--debug", action="store_true")
    cleanup.add_argument("--ocr-cache", action="store_true")
    cleanup.add_argument("--all", action="store_true")
    cleanup.add_argument("--dry-run", action="store_true")
    cleanup.add_argument("--log-level", default="INFO")

    ocr_batch = sub.add_parser(
        "ocr-batch",
        help="Последовательно выполнить только OCR и заполнить компактный кэш",
    )
    ocr_batch.add_argument("--documents", type=Path, nargs="+", required=True)
    ocr_batch.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/reports/ocr_batch/summary.json"),
    )
    ocr_batch.add_argument("--config", type=Path)
    ocr_batch.add_argument("--reuse-ocr-cache", action="store_true")
    ocr_batch.add_argument("--force-ocr", action="store_true")
    ocr_batch.add_argument(
        "--include-visual-checks",
        action="store_true",
        help="Проверить подпись и печать по изображениям, не повторяя OCR",
    )
    ocr_batch.add_argument("--ocr-cache-dir", type=Path)
    ocr_batch.add_argument("--ocr-language", default=None)
    ocr_batch.add_argument(
        "--ocr-backend",
        choices=["monkeyocr-v2"],
        default=None,
    )
    ocr_batch.add_argument(
        "--device", choices=["auto", "cpu", "gpu:0"], default=None
    )
    ocr_batch.add_argument("--log-level", default="INFO")

    compare = sub.add_parser(
        "compare-monkey",
        help="Сравнить сохранённый локальный MonkeyOCR с результатом web без запуска модели",
    )
    compare.add_argument("--document", type=Path, required=True)
    compare.add_argument("--web-result", type=Path, required=True)
    compare.add_argument("--local-result", type=Path)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--log-level", default="INFO")
    return parser


def _build_ocr_engine(
    args, config: dict, *, prepare: bool = True
) -> OCREngine:
    config = resolve_monkeyocr_settings(config)
    if getattr(args, "device", None):
        config["ocr"]["device"] = args.device
    if getattr(args, "ocr_backend", None):
        config["ocr"]["backend"] = args.ocr_backend.replace("-", "_")
    monkey = config["ocr"].setdefault("monkeyocr_v2", {})
    language = args.ocr_language or config["ocr"]["language"]
    engine = build_ocr_engine(config, language)
    if not prepare:
        print(
            "OCR ENGINE: initialization deferred until OCR cache miss "
            f"(backend={config['ocr']['backend']}, "
            f"requested_device={config['ocr']['device']})"
        )
        return engine
    prepare_method = getattr(engine, "prepare", None)
    info = prepare_method() if callable(prepare_method) else {}
    _print_ocr_engine_info(info, config)
    return engine


def _print_ocr_engine_info(info: dict, config: dict) -> None:
    print(
        "OCR ENGINE: "
        f"backend={info.get('backend', config['ocr']['backend'])}, "
        f"runtime={info.get('runtime', 'native')}, "
        f"requested={info.get('requested_device', config['ocr']['device'])}, "
        f"actual={info.get('actual_device', 'not-reported')}, "
        f"GPU={info.get('gpu_model') or 'n/a'}, "
        f"VRAM={info.get('gpu_memory_mib') or 'n/a'} MiB, "
        f"model={info.get('model_id') or 'n/a'}"
    )
    if info.get("fallback_reason"):
        print(f"OCR FALLBACK: {info['fallback_reason']}")


def _debug_level(args) -> str:
    return "full" if getattr(args, "save_debug", False) else args.debug_level


def _parse_id_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    result: set[str] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            try:
                start, finish = int(left), int(right)
            except ValueError as exc:
                raise InputValidationError(f"Некорректный диапазон ID: {item}") from exc
            if start > finish or finish - start > 10000:
                raise InputValidationError(f"Некорректный диапазон ID: {item}")
            result.update(str(number) for number in range(start, finish + 1))
        else:
            if not item.isdigit():
                raise InputValidationError(f"Некорректный ID: {item}")
            result.add(str(int(item)))
    return result or None


def _ocr_cache(args, config: dict) -> OCRCache | None:
    cache_config = config.get("ocr_cache", {})
    if not bool(cache_config.get("enabled", True)):
        return None
    root = args.ocr_cache_dir or Path(
        cache_config.get("directory", "runtime/cache/ocr")
    )
    return OCRCache(
        root,
        max_size_mb=int(cache_config.get("max_size_mb", 1024)),
        retention_days=int(cache_config.get("retention_days", 30)),
        cleanup_on_start=bool(cache_config.get("cleanup_on_start", True)),
    )


def _check(args) -> int:
    service_options = {
        "config_path": args.config,
        "device": args.device,
        "ocr_backend": args.ocr_backend,
        "ocr_language": args.ocr_language,
        "ocr_cache_dir": args.ocr_cache_dir,
        "reuse_ocr_cache": args.reuse_ocr_cache,
        "force_ocr": args.force_ocr,
        "debug_level": _debug_level(args),
        "save_debug": args.save_debug,
        "emit_console": True,
        "ground_truth_path": args.ground_truth,
        "output_dir": args.output_dir,
    }
    service = SpecificationCheckerService(service_options)
    if args.reuse_ocr_cache and not args.force_ocr:
        print(
            "OCR ENGINE: initialization deferred until OCR cache miss "
            f"(backend={service.config['ocr']['backend']}, "
            f"requested_device={service.config['ocr']['device']})"
        )
    else:
        _print_ocr_engine_info(service.prepare(), service.config)
    result = service.check_specification(
        args.document,
        args.reference,
        service_options,
    )
    return EXIT_CODES[FinalDecision(result["decision"])]


def _check_mock(args) -> int:
    config = load_config(args.config)
    # Mock extraction завершается до чтения reference.
    extracted = load_mock_extracted(args.extracted)
    reference = ReferenceData.model_validate_json(args.reference.read_text(encoding="utf-8"))
    checks, decision, reasons = RuleEngine(config).evaluate(reference, extracted)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "mock",
        "extracted_path": str(args.extracted),
        "reference_path": str(args.reference),
        "checks": [item.model_dump(mode="json") for item in checks],
        "final_decision": decision.value,
        "decision_reasons": reasons,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [RULE, "ПРОВЕРКА MOCK EXTRACTION", RULE]
    lines.extend(f"[{item.status.value}] {item.title}: {item.message}" for item in checks)
    lines.extend([RULE, f"ИТОГОВОЕ РЕШЕНИЕ: {decision.value}", RULE])
    text = "\n".join(lines) + "\n"
    (args.output_dir / "console_report.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return EXIT_CODES[decision]


def _dataset(args) -> int:
    config = load_config(args.config)
    cache = _ocr_cache(args, config)
    engine = _build_ocr_engine(
        args,
        config,
        prepare=not (args.reuse_ocr_cache and not args.force_ocr),
    )
    summary = evaluate_dataset(
        args.documents_dir,
        args.reference_dir,
        args.ground_truth_dir,
        args.output_dir,
        config,
        lambda: engine,
        _debug_level(args),
        cache,
        args.reuse_ocr_cache,
        args.force_ocr,
        _parse_id_filter(args.include_ids),
    )
    return 0 if summary["technical_errors"] == 0 else 20


def _import_ocr_cache(args) -> int:
    config = load_config(args.config)
    cache = _ocr_cache(args, config)
    if cache is None:
        raise InputValidationError("OCR-кэш отключён в конфигурации")
    loaded = DocumentLoader(config["document"]["supported_extensions"]).load(
        args.document
    )
    try:
        result = cache.import_saved_run(loaded, config, args.saved_output)
    except ValueError as exc:
        raise InputValidationError(str(exc)) from exc
    print(
        f"OCR CACHE IMPORTED: key={result['key']}, "
        f"size={result['size_bytes'] / 1024:.1f} KiB, path={result['path']}"
    )
    return 0


def _cleanup(args) -> int:
    choose_all = args.all or not (args.temp or args.debug or args.ocr_cache)
    result = cleanup_runtime(
        args.runtime_dir,
        include_temp=choose_all or args.temp,
        include_debug=choose_all or args.debug,
        include_cache=choose_all or args.ocr_cache,
        older_than_days=args.older_than_days,
        dry_run=args.dry_run,
    )
    mode = "DRY-RUN" if args.dry_run else "DONE"
    print(
        f"CLEANUP {mode}: files={result['removed_files']}, "
        f"space={result['freed_bytes'] / (1024 * 1024):.2f} MiB"
    )
    return 0


def _run_ocr_batch(args) -> int:
    config = load_config(args.config)
    cache = _ocr_cache(args, config)
    if cache is None:
        raise InputValidationError("OCR-кэш отключён в конфигурации")
    engine = _build_ocr_engine(
        args,
        config,
        prepare=not (args.reuse_ocr_cache and not args.force_ocr),
    )
    result = run_ocr_batch(
        args.documents,
        args.output,
        config,
        engine,
        cache,
        reuse_cache=args.reuse_ocr_cache,
        force_ocr=args.force_ocr,
        include_visual_checks=args.include_visual_checks,
    )
    print(
        "OCR BATCH: "
        f"documents={result['document_count']}, "
        f"elapsed={result['elapsed_seconds']:.1f}s, "
        f"output={args.output}"
    )
    for item in result["documents"]:
        state = "HIT" if item["cache"].get("hit") else "CREATED"
        print(
            f"- {Path(item['document']).name}: pages={item['page_count']}, "
            f"blocks={item['blocks']}, mean_confidence={item['mean_confidence']:.1%}, "
            f"cache={state}, elapsed={item['elapsed_seconds']:.1f}s"
        )
    return 0


def _compare_monkey(args) -> int:
    local_result = args.local_result or (
        Path("runtime/reports") / args.document.stem / "monkey" / "ocr_document.json"
    )
    if not local_result.is_file():
        raise InputValidationError(
            f"Сохранённый локальный MonkeyOCR не найден: {local_result}. "
            "Команда сравнения не запускает модель; укажите --local-result."
        )
    result = compare_saved_monkey(local_result, args.web_result)
    output = args.output or local_result.parent / "web_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "MONKEY COMPARISON: "
        f"CER={result['cer']:.2%}, WER={result['wer']:.2%}, "
        f"local_layout={result['local_layout_elements']}, "
        f"web_layout={result['web_layout_elements']}, output={output}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_for_logging = load_config(getattr(args, "config", None))
    logging_config = config_for_logging.get("logging", {})
    log_dir = Path(
        config_for_logging.get("runtime", {}).get("logs_dir", "runtime/logs")
    )
    configure_logging(
        args.log_level,
        log_dir,
        int(logging_config.get("max_size_mb", 10)),
        int(logging_config.get("backup_count", 5)),
    )
    try:
        if args.command == "check":
            return _check(args)
        if args.command == "check-mock":
            return _check_mock(args)
        if args.command == "evaluate-dataset":
            return _dataset(args)
        if args.command == "import-ocr-cache":
            return _import_ocr_cache(args)
        if args.command == "cleanup":
            return _cleanup(args)
        if args.command == "ocr-batch":
            return _run_ocr_batch(args)
        if args.command == "compare-monkey":
            return _compare_monkey(args)
        raise InputValidationError(f"Неизвестная команда: {args.command}")
    except InputValidationError as exc:
        print(f"Ошибка входных данных: {exc}", file=sys.stderr)
        return 30
    except SpecificationCheckerError as exc:
        print(f"Техническая ошибка: {exc}", file=sys.stderr)
        return 40
    except Exception as exc:
        print(f"Непредвиденная техническая ошибка: {exc}", file=sys.stderr)
        return 40


if __name__ == "__main__":
    raise SystemExit(main())
