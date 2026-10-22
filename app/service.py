"""Public, process-reusable programmatic interface for specification checks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from threading import Lock, RLock
from typing import Any

from app.config import load_config, merge_config
from app.ocr.monkey_settings import resolve_monkeyocr_settings
from app.enums import CheckStatus, FinalDecision
from app.exceptions import InputValidationError
from app.ocr.base import OCREngine
from app.ocr.cache import OCRCache
from app.ocr.factory import build_ocr_engine
from app.pipeline import run_check
from app.schemas import ReferenceData, VerificationReport
from app.vision.seal_detector import OpenCVSealDetector, SealDetector
from app.vision.signature_detector import OpenCVSignatureDetector, SignatureDetector

ReferenceInput = Path | str | Mapping[str, Any] | ReferenceData

_RUN_OPTION_KEYS = {
    "output_dir",
    "debug_level",
    "save_debug",
    "reuse_ocr_cache",
    "force_ocr",
    "emit_console",
    "job_id",
    "ground_truth_path",
}
_SERVICE_OPTION_KEYS = {
    "config_path",
    "config_overrides",
    "device",
    "ocr_backend",
    "ocr_language",
    "ocr_cache_dir",
}


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


def _build_ocr_cache(config: dict[str, Any]) -> OCRCache | None:
    cache_config = config.get("ocr_cache", {})
    if not bool(cache_config.get("enabled", True)):
        return None
    return OCRCache(
        Path(cache_config.get("directory", "runtime/cache/ocr")),
        max_size_mb=int(cache_config.get("max_size_mb", 1024)),
        retention_days=int(cache_config.get("retention_days", 30)),
        cleanup_on_start=bool(cache_config.get("cleanup_on_start", True)),
    )


def _one_c_comment(decision: FinalDecision, reasons: list[str]) -> str:
    prefix = {
        FinalDecision.APPROVE: "Спецификация согласована.",
        FinalDecision.RETURN: "Спецификация возвращена.",
        FinalDecision.MANUAL_REVIEW: "Требуется ручная проверка спецификации.",
    }[decision]
    details = "; ".join(reason.strip() for reason in reasons if reason.strip())
    return f"{prefix} {details}".strip()


def _overall_confidence(report: VerificationReport) -> float:
    required = [
        check.confidence
        for check in report.checks
        if check.is_required and check.status is not CheckStatus.SKIPPED
    ]
    return round(min(required), 6) if required else 0.0


def serialize_service_result(
    report: VerificationReport,
    output_dir: Path,
) -> dict[str, Any]:
    """Convert the internal report into a stable JSON-ready integration payload."""

    field_details = report.extracted_fields.model_dump(mode="json")
    extracted_values = {
        name: details.get("normalized_value")
        for name, details in field_details.items()
        if isinstance(details, dict) and "normalized_value" in details
    }
    field_confidence = {
        name: details.get("confidence", 0.0)
        for name, details in field_details.items()
        if isinstance(details, dict) and "confidence" in details
    }
    payload = {
        "decision": report.final_decision.value,
        "extracted_values": extracted_values,
        "field_details": field_details,
        "checks": [check.model_dump(mode="json") for check in report.checks],
        "reasons": list(report.decision_reasons),
        "comment_1c": _one_c_comment(
            report.final_decision, report.decision_reasons
        ),
        "confidence": {
            "overall": _overall_confidence(report),
            "fields": field_confidence,
        },
        "metadata": {
            "specification_id": report.specification_id,
            "document_path": report.document_path,
            "reference_source": report.reference_path,
            "document_hash_sha256": report.document_hash_sha256,
            "page_count": report.page_count,
            "started_at": report.started_at.isoformat(),
            "finished_at": report.finished_at.isoformat(),
            "processing_time_seconds": report.processing_time_seconds,
            "output_dir": str(output_dir),
            "warnings": list(report.warnings),
            "technical_errors": list(report.technical_errors),
            "software_versions": dict(report.software_versions),
            "performance": deepcopy(report.performance),
            "document_quality": [
                item.model_dump(mode="json") for item in report.document_quality
            ],
        },
    }
    # Fail immediately during development if a future field stops being JSON-safe.
    json.dumps(payload, ensure_ascii=False)
    return payload


class SpecificationCheckerService:
    """Own one OCR engine/cache and reuse them for sequential document checks."""

    def __init__(
        self,
        runtime_config: Mapping[str, Any] | None = None,
        *,
        ocr_engine: OCREngine | None = None,
        seal_detector: SealDetector | None = None,
        signature_detector: SignatureDetector | None = None,
    ) -> None:
        options = dict(runtime_config or {})
        config_path = _path_or_none(options.get("config_path"))
        config = load_config(config_path)
        overrides = options.get("config_overrides", {})
        if overrides:
            if not isinstance(overrides, Mapping):
                raise InputValidationError("config_overrides должен быть объектом")
            config = merge_config(config, dict(overrides))

        if options.get("device") is not None:
            config["ocr"]["device"] = str(options["device"])
        if options.get("ocr_backend") is not None:
            config["ocr"]["backend"] = str(options["ocr_backend"]).replace(
                "-", "_"
            )
        if options.get("ocr_language") is not None:
            config["ocr"]["language"] = str(options["ocr_language"])
        monkey = config["ocr"].setdefault("monkeyocr_v2", {})
        if options.get("ocr_cache_dir") is not None:
            config.setdefault("ocr_cache", {})["directory"] = str(
                options["ocr_cache_dir"]
            )

        config = resolve_monkeyocr_settings(config)

        self.config = config
        self.ocr_engine = ocr_engine or build_ocr_engine(
            config, str(config["ocr"].get("language", "ru"))
        )
        self.ocr_cache = _build_ocr_cache(config)
        self.seal_detector = seal_detector or OpenCVSealDetector(
            float(config["buyer_visual_checks"]["min_seal_confidence"])
        )
        self.signature_detector = signature_detector or OpenCVSignatureDetector(
            float(config["buyer_visual_checks"]["min_signature_confidence"])
        )
        self._run_lock = RLock()
        self._run_defaults = {
            "debug_level": str(options.get("debug_level", "none")),
            "save_debug": bool(options.get("save_debug", False)),
            "reuse_ocr_cache": bool(options.get("reuse_ocr_cache", True)),
            "force_ocr": bool(options.get("force_ocr", False)),
            "emit_console": bool(options.get("emit_console", False)),
            "job_id": options.get("job_id"),
            "ground_truth_path": options.get("ground_truth_path"),
            "output_dir": options.get("output_dir"),
        }

    def prepare(self) -> dict[str, Any]:
        """Load the model once; later checks reuse the same in-memory instance."""

        with self._run_lock:
            method = getattr(self.ocr_engine, "prepare", None)
            return method() if callable(method) else {}

    def check_specification(
        self,
        document_path: str | Path,
        reference: ReferenceInput,
        runtime_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = dict(self._run_defaults)
        supplied = dict(runtime_config or {})
        options.update({key: value for key, value in supplied.items() if key in _RUN_OPTION_KEYS})
        document = Path(document_path)
        output = _path_or_none(options.get("output_dir"))
        if output is None:
            output = Path(
                self.config.get("runtime", {}).get(
                    "reports_dir", "runtime/reports"
                )
            ) / document.stem
        debug_level = "full" if bool(options["save_debug"]) else str(
            options["debug_level"]
        )
        ground_truth = _path_or_none(options.get("ground_truth_path"))

        # MonkeyOCR keeps mutable per-document state and GPU buffers. A single
        # service intentionally processes documents sequentially.
        with self._run_lock:
            report = run_check(
                document_path=document,
                reference_path=reference,
                ground_truth_path=ground_truth,
                output_dir=output,
                config=self.config,
                ocr_engine=self.ocr_engine,
                save_debug=bool(options["save_debug"]),
                seal_detector=self.seal_detector,
                signature_detector=self.signature_detector,
                debug_level=debug_level,
                ocr_cache=self.ocr_cache,
                reuse_ocr_cache=bool(options["reuse_ocr_cache"]),
                force_ocr=bool(options["force_ocr"]),
                emit_console=bool(options["emit_console"]),
                job_id=str(options["job_id"]) if options.get("job_id") else None,
            )
        return serialize_service_result(report, output)


_DEFAULT_SERVICE: SpecificationCheckerService | None = None
_DEFAULT_SERVICE_SIGNATURE: str | None = None
_DEFAULT_SERVICE_LOCK = Lock()


def check_specification(
    document_path: str | Path,
    reference: ReferenceInput,
    runtime_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check one document through a lazily created process-wide service."""

    global _DEFAULT_SERVICE, _DEFAULT_SERVICE_SIGNATURE
    supplied = dict(runtime_config or {})
    service_options = {
        key: value for key, value in supplied.items() if key in _SERVICE_OPTION_KEYS
    }
    signature = json.dumps(service_options, sort_keys=True, default=str)
    if _DEFAULT_SERVICE is None:
        with _DEFAULT_SERVICE_LOCK:
            if _DEFAULT_SERVICE is None:
                _DEFAULT_SERVICE = SpecificationCheckerService(service_options)
                _DEFAULT_SERVICE_SIGNATURE = signature
    elif service_options and signature != _DEFAULT_SERVICE_SIGNATURE:
        raise InputValidationError(
            "Процессный check_specification уже инициализирован с другой "
            "конфигурацией; создайте отдельный SpecificationCheckerService"
        )
    return _DEFAULT_SERVICE.check_specification(
        document_path, reference, runtime_config
    )
