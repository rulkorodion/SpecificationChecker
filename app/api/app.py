"""Versioned FastAPI application that submits checks to a Redis worker."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import shutil
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.api.schemas import (
    Base64Document,
    CheckJobResponse,
    CheckRequestJSON,
    CheckRequestForm,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
)
from app.api.rejections import archive_rejected_request, cleanup_rejected_requests
from app.api.storage import APISettings, JobWorkspace
from app.config import load_config
from app.persistence.database import Database, DatabaseConfigurationError
from app.persistence.models import JobStatus
from app.persistence.repository import CheckJobRecord
from app.persistence.service import JobPersistenceError, JobPersistenceService
from app.queueing import JobQueue, QueueSettings, QueueUnavailableError, RedisJobQueue
from app.schemas import ReferenceData

LOGGER = logging.getLogger(__name__)
API_VERSION = "v1"
ALLOWED_SIGNATURES = {
    ".pdf": (b"%PDF-",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
}
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}
SIGNED_SPECIFICATION_PATTERN = re.compile(
    r"^подписанная[\s_]+спецификация[\s_]+к[\s_]+заказу",
    re.IGNORECASE,
)
ORDER_NUMBER_PATTERN = re.compile(r"(?:^|[^a-zа-я0-9])(?:n|№)\s*0*(\d+)", re.IGNORECASE)
IMAGE_PAGE_SUFFIX_PATTERN = re.compile(
    r"(?:[\s_-]+(?:стр(?:аница)?|page)?[\s_-]*\d+(?:[\s_-]+из[\s_-]+\d+)?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DecodedDocument:
    source_index: int
    source: Base64Document
    content: bytes
    suffix: str
    media_type: str
    sha256: str


class RequestProblem(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_detail(code: str, message: str, job_id: str | None) -> dict[str, Any]:
    return {"code": code, "message": message, "job_id": job_id}


def _config_path_from_environment() -> Path | None:
    value = os.getenv("SPECIFICATION_CHECKER_CONFIG", "").strip()
    return Path(value) if value else None


def _default_job_service_factory() -> JobPersistenceService:
    return JobPersistenceService(Database.from_environment())


def _default_queue_factory() -> JobQueue:
    return RedisJobQueue(QueueSettings.from_environment())


def get_job_service(request: Request) -> JobPersistenceService:
    service = getattr(request.app.state, "job_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=_error_detail(
                "database_unavailable", "Постоянное хранилище заданий недоступно.", None
            ),
        )
    return service


def get_job_queue(request: Request) -> JobQueue:
    queue = getattr(request.app.state, "job_queue", None)
    if queue is None:
        raise HTTPException(
            status_code=503,
            detail=_error_detail("redis_unavailable", "Очередь Redis недоступна.", None),
        )
    return queue


def _parse_reference(raw: str, settings: APISettings) -> dict[str, Any]:
    if len(raw.encode("utf-8")) > settings.max_reference_size_bytes:
        raise RequestProblem(413, "reference_too_large", "Reference JSON превышает допустимый размер.")
    try:
        value = json.loads(raw, parse_float=Decimal)
    except json.JSONDecodeError as exc:
        raise RequestProblem(
            422,
            "invalid_reference_json",
            f"Reference содержит некорректный JSON: строка {exc.lineno}, столбец {exc.colno}.",
        ) from exc
    return _validate_reference(value, settings)


def _validate_reference(value: Any, settings: APISettings) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RequestProblem(422, "invalid_reference_json", "Reference должен быть JSON-объектом.")
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    if len(serialized) > settings.max_reference_size_bytes:
        raise RequestProblem(413, "reference_too_large", "Reference JSON превышает допустимый размер.")
    return dict(value)


def _normalize_reference(value: dict[str, Any]) -> dict[str, Any]:
    try:
        reference = ReferenceData.model_validate(value)
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            first = errors[0]
            field = ".".join(str(part) for part in first.get("loc", ()))
            message = str(first.get("msg", "Некорректное значение"))
            if field:
                message = f"{field}: {message}"
        else:
            message = "Эталонные данные не соответствуют ожидаемой структуре."
        raise RequestProblem(422, "invalid_reference", message) from exc
    return reference.model_dump(mode="json")


def _validate_filename(filename: str | None) -> str:
    if not filename:
        raise RequestProblem(422, "missing_filename", "У файла отсутствует имя.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SIGNATURES:
        raise RequestProblem(
            415,
            "unsupported_document_format",
            f"Поддерживаются только форматы: {', '.join(sorted(ALLOWED_SIGNATURES))}.",
        )
    return suffix


async def _save_document(
    upload: UploadFile, destination: Path, suffix: str, settings: APISettings
) -> tuple[int, str]:
    total = 0
    header = b""
    digest = hashlib.sha256()
    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(settings.upload_chunk_size_bytes):
                total += len(chunk)
                if total > settings.max_upload_size_bytes:
                    raise RequestProblem(
                        413,
                        "document_too_large",
                        f"Документ превышает лимит {settings.max_upload_size_bytes // (1024 * 1024)} МБ.",
                    )
                if len(header) < 16:
                    header += chunk[: 16 - len(header)]
                digest.update(chunk)
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if total == 0:
        destination.unlink(missing_ok=True)
        raise RequestProblem(422, "empty_document", "Загруженный документ пуст.")
    if not any(header.startswith(signature) for signature in ALLOWED_SIGNATURES[suffix]):
        destination.unlink(missing_ok=True)
        raise RequestProblem(
            415,
            "document_signature_mismatch",
            "Содержимое файла не соответствует его расширению.",
        )
    return total, digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_base64_document(
    encoded: str,
    *,
    filename: str,
    content_type: str | None,
    remaining_bytes: int,
) -> tuple[bytes, str, str]:
    payload = encoded.strip()
    if payload.lower().startswith("data:"):
        try:
            header, payload = payload.split(",", 1)
        except ValueError as exc:
            raise RequestProblem(422, "invalid_document_base64", "Некорректный data URI документа.") from exc
        if ";base64" not in header.lower():
            raise RequestProblem(422, "invalid_document_base64", "Data URI документа должен содержать Base64.")

    compact = "".join(payload.split())
    if not compact:
        raise RequestProblem(422, "empty_document", "Документ в Base64 пуст.")
    estimated_size = (len(compact) * 3) // 4
    if estimated_size > remaining_bytes + 2:
        raise RequestProblem(413, "document_too_large", "Суммарный размер документов превышает допустимый лимит.")
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestProblem(422, "invalid_document_base64", f"Документ {filename!r} содержит некорректный Base64.") from exc
    if not decoded:
        raise RequestProblem(422, "empty_document", f"Документ {filename!r} пуст.")
    if len(decoded) > remaining_bytes:
        raise RequestProblem(413, "document_too_large", "Суммарный размер документов превышает допустимый лимит.")
    detected_suffix = next(
        (
            suffix
            for suffix, signatures in ALLOWED_SIGNATURES.items()
            if any(decoded.startswith(signature) for signature in signatures)
        ),
        None,
    )
    if detected_suffix is None:
        raise RequestProblem(
            415,
            "unsupported_document_format",
            f"Не удалось определить поддерживаемый формат документа {filename!r} по его содержимому.",
        )
    # filename/content_type are metadata supplied by 1C. The file signature is
    # authoritative because real requests may omit extensions or send malformed
    # MIME values such as application/png and application/.jpg.
    canonical_suffix = ".jpg" if detected_suffix == ".jpeg" else detected_suffix
    return decoded, canonical_suffix, MEDIA_TYPES[canonical_suffix]


def _attachment_basename(filename: str) -> str:
    return re.split(r"[\\/]", filename.strip())[-1]


def _is_signed_specification(filename: str) -> bool:
    return bool(SIGNED_SPECIFICATION_PATTERN.match(_attachment_basename(filename)))


def _normalized_digits(value: str) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits.lstrip("0") or "0"


def _filename_order_number(filename: str) -> str | None:
    normalized = _attachment_basename(filename).replace("ё", "е")
    match = ORDER_NUMBER_PATTERN.search(normalized)
    return _normalized_digits(match.group(1)) if match else None


def _filename_matches_application(filename: str, application_number: str) -> bool:
    target = _normalized_digits(application_number)
    order_number = _filename_order_number(filename)
    if order_number is not None:
        return order_number == target
    numeric_tokens = re.findall(r"\d+", _attachment_basename(filename))
    return target in {_normalized_digits(token) for token in numeric_tokens}


def _image_group_key(filename: str) -> str:
    basename = _attachment_basename(filename).casefold()
    stem = re.sub(r"\.(?:jpe?g|png)$", "", basename, flags=re.IGNORECASE)
    stem = IMAGE_PAGE_SUFFIX_PATTERN.sub("", stem)
    return re.sub(r"[\s_]+", " ", stem).strip()


def _decode_candidate_documents(
    candidates: list[tuple[int, Base64Document]],
    settings: APISettings,
) -> tuple[list[DecodedDocument], list[RequestProblem]]:
    if len(candidates) > settings.max_documents_per_job:
        raise RequestProblem(
            413,
            "too_many_documents",
            f"В одном задании разрешено не более {settings.max_documents_per_job} документов спецификации.",
        )
    remaining = settings.max_upload_size_bytes
    decoded_documents: list[DecodedDocument] = []
    unsupported: list[RequestProblem] = []
    for source_index, item in candidates:
        try:
            content, suffix, media_type = _decode_base64_document(
                item.content_base64,
                filename=item.filename,
                content_type=item.content_type,
                remaining_bytes=remaining,
            )
        except RequestProblem as exc:
            if exc.status_code == 413:
                raise
            unsupported.append(exc)
            continue
        remaining -= len(content)
        decoded_documents.append(
            DecodedDocument(
                source_index=source_index,
                source=item,
                content=content,
                suffix=suffix,
                media_type=media_type,
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    return decoded_documents, unsupported


def _select_json_documents(
    payload: CheckRequestJSON,
    settings: APISettings,
    *,
    application_number: str,
) -> list[DecodedDocument]:
    indexed = list(enumerate(payload.documents, start=1))
    signed = [item for item in indexed if _is_signed_specification(item[1].filename)]
    if not signed:
        raise RequestProblem(
            415,
            "signed_specification_not_found",
            "Среди вложений не найдена подписанная спецификация к заказу.",
        )

    decoded, unsupported = _decode_candidate_documents(signed, settings)
    if not decoded:
        if unsupported:
            raise unsupported[0]
        raise RequestProblem(
            415,
            "unsupported_document_format",
            "Подписанная спецификация имеет неподдерживаемый формат.",
        )

    pdfs = [item for item in decoded if item.suffix == ".pdf"]
    if pdfs:
        matching = [
            item
            for item in pdfs
            if _filename_matches_application(item.source.filename, application_number)
        ]
        choices = matching or pdfs
        if len(choices) == 1:
            return choices
        if len({item.sha256 for item in choices}) == 1:
            return [choices[0]]
        raise RequestProblem(
            422,
            "ambiguous_signed_specification",
            "Найдено несколько разных подписанных PDF; однозначный выбор невозможен.",
        )

    images = [item for item in decoded if item.suffix in {".jpg", ".png"}]
    matching_images = [
        item
        for item in images
        if _filename_matches_application(item.source.filename, application_number)
    ]
    if matching_images:
        return matching_images
    identifiers = {
        identifier
        for item in images
        if (identifier := _filename_order_number(item.source.filename)) is not None
    }
    if len(identifiers) > 1:
        raise RequestProblem(
            422,
            "ambiguous_signed_specification",
            "Найдены изображения нескольких разных подписанных спецификаций.",
        )
    if not identifiers and len({_image_group_key(item.source.filename) for item in images}) > 1:
        raise RequestProblem(
            422,
            "ambiguous_signed_specification",
            "Найдены изображения нескольких разных подписанных спецификаций.",
        )
    return images


def _merge_source_documents(sources: list[Path], destination: Path) -> None:
    try:
        import fitz

        merged = fitz.open()
        try:
            for source in sources:
                opened = fitz.open(source)
                try:
                    if source.suffix.lower() == ".pdf":
                        merged.insert_pdf(opened)
                    else:
                        converted = fitz.open("pdf", opened.convert_to_pdf())
                        try:
                            merged.insert_pdf(converted)
                        finally:
                            converted.close()
                finally:
                    opened.close()
            if merged.page_count == 0:
                raise RequestProblem(422, "empty_document", "Переданные документы не содержат страниц.")
            temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
            merged.save(temporary)
            temporary.replace(destination)
        finally:
            merged.close()
    except RequestProblem:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise RequestProblem(422, "invalid_document", "Не удалось объединить переданные документы.") from exc


def _save_base64_documents(
    documents: list[DecodedDocument],
    workspace: JobWorkspace,
) -> tuple[Path, str, str]:
    sources: list[Path] = []
    metadata: list[dict[str, Any]] = []
    source_dir = workspace.path / "source_documents"
    if len(documents) > 1:
        source_dir.mkdir()

    for selected_index, item in enumerate(documents, start=1):
        destination = (
            workspace.path / f"document{item.suffix}"
            if len(documents) == 1
            else source_dir / f"{selected_index:03d}{item.suffix}"
        )
        destination.write_bytes(item.content)
        sources.append(destination)
        metadata.append(
            {
                "index": selected_index,
                "source_index": item.source_index,
                "filename": item.source.filename,
                "declared_content_type": item.source.content_type,
                "content_type": item.media_type,
                "size_bytes": len(item.content),
                "sha256": item.sha256,
            }
        )

    workspace.write_json("source_documents.json", {"documents": metadata})
    if len(sources) == 1:
        canonical = sources[0]
    else:
        canonical = workspace.path / "document.pdf"
        _merge_source_documents(sources, canonical)
    return canonical.resolve(), _sha256_file(canonical), metadata[0]["filename"] if len(metadata) == 1 else f"{len(metadata)} selected documents"


async def _read_submission(
    request: Request, settings: APISettings
) -> tuple[str, CheckRequestForm | CheckRequestJSON, UploadFile | None]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    try:
        if content_type == "application/json":
            raw = json.loads(await request.body(), parse_float=Decimal)
            request.state.parsed_json_body = raw
            return "json", CheckRequestJSON.model_validate(raw), None
        if content_type == "multipart/form-data":
            form_data = await request.form()
            document = form_data.get("document")
            if not isinstance(document, StarletteUploadFile):
                raise RequestProblem(422, "missing_document", "Поле document должно содержать файл.")
            reference = form_data.get("reference")
            if not isinstance(reference, str):
                raise RequestProblem(422, "missing_reference", "Поле reference должно содержать JSON.")
            external_id = form_data.get("external_id")
            return (
                "multipart",
                CheckRequestForm.model_validate(
                    {"reference": reference, "external_id": external_id}
                ),
                document,
            )
    except RequestProblem:
        raise
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        message = "Тело запроса не соответствует ожидаемой структуре."
        if isinstance(exc, ValidationError) and exc.errors():
            message = str(exc.errors()[0].get("msg", message))
        raise RequestProblem(422, "invalid_request", message) from exc
    raise RequestProblem(
        415,
        "unsupported_content_type",
        "Поддерживаются application/json и multipart/form-data.",
    )


def _elapsed(started_at: datetime, finished_at: datetime) -> float:
    return max(0.0, (finished_at - started_at).total_seconds())


def _job_response(record: CheckJobRecord) -> CheckJobResponse:
    error = None
    if record.error_code or record.error_message:
        error = ErrorDetail(
            code=record.error_code or "check_failed",
            message=record.error_message or "Проверка завершилась ошибкой.",
            job_id=record.job_id,
        )
    return CheckJobResponse(
        job_id=record.job_id,
        external_id=record.external_id,
        status=record.status,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        processing_time_seconds=record.processing_time_seconds,
        result=record.result,
        error=error,
    )


ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Job was not found."},
    413: {"model": ErrorResponse, "description": "Document or reference is too large."},
    415: {"model": ErrorResponse, "description": "Unsupported or disguised file format."},
    422: {"model": ErrorResponse, "description": "Invalid request or reference."},
    503: {"model": ErrorResponse, "description": "PostgreSQL or Redis is unavailable."},
}

CHECK_REQUEST_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["documents", "reference"],
                    "properties": {
                        "external_id": {"type": ["string", "null"], "maxLength": 128},
                        "documents": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["filename", "content_base64"],
                                "properties": {
                                    "filename": {"type": "string"},
                                    "content_type": {"type": ["string", "null"]},
                                    "content_base64": {"type": "string", "format": "byte"},
                                },
                            },
                        },
                        "reference": {"type": "object", "additionalProperties": True},
                    },
                }
            },
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["document", "reference"],
                    "properties": {
                        "document": {"type": "string", "format": "binary"},
                        "reference": {"type": "string"},
                        "external_id": {"type": ["string", "null"], "maxLength": 128},
                    },
                }
            },
        },
    }
}


def create_app(
    *,
    service_factory: Callable[[], Any] | None = None,
    job_service_factory: Callable[[], JobPersistenceService] | None = None,
    queue_factory: Callable[[], JobQueue] | None = None,
    api_settings: APISettings | None = None,
    config_path: Path | None = None,
) -> FastAPI:
    """Application factory. ``service_factory`` is retained but never invoked."""

    del service_factory  # Checker intentionally lives only in the worker process.
    resolved_config_path = config_path or _config_path_from_environment()
    settings = api_settings or APISettings.from_config(load_config(resolved_config_path))
    persistence_factory = job_service_factory or _default_job_service_factory
    resolved_queue_factory = queue_factory or _default_queue_factory

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.job_service = None
        application.state.job_queue = None
        application.state.database_initialization_error = None
        application.state.redis_initialization_error = None
        try:
            application.state.job_service = persistence_factory()
        except DatabaseConfigurationError as exc:
            application.state.database_initialization_error = str(exc)
        except Exception:
            application.state.database_initialization_error = "Не удалось инициализировать PostgreSQL"
            LOGGER.exception("Database initialization failed")
        try:
            application.state.job_queue = resolved_queue_factory()
        except Exception as exc:
            application.state.redis_initialization_error = str(exc) or "Не удалось инициализировать Redis"
            LOGGER.exception("Redis queue initialization failed")
        if settings.rejected_requests_enabled:
            try:
                await run_in_threadpool(cleanup_rejected_requests, settings)
            except OSError:
                LOGGER.warning("Could not clean rejected-request quarantine", exc_info=True)
        try:
            yield
        finally:
            if application.state.job_queue is not None:
                application.state.job_queue.close()
            if application.state.job_service is not None:
                application.state.job_service.close()
            application.state.job_queue = None
            application.state.job_service = None

    application = FastAPI(
        title="Specification Checker API",
        description="Asynchronous API for 1C integration through PostgreSQL and Redis.",
        version="1.5.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json",
    )
    router = APIRouter(prefix=f"/api/{API_VERSION}")

    @application.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        summary="PostgreSQL, Redis and worker status",
    )
    async def health(request: Request):
        job_service = getattr(request.app.state, "job_service", None)
        queue = getattr(request.app.state, "job_queue", None)
        database_available = False
        redis_available = False
        worker_id = None
        worker_ttl = None
        worker_registrations: list[tuple[str, int]] = []
        database_error = getattr(request.app.state, "database_initialization_error", None)
        redis_error = getattr(request.app.state, "redis_initialization_error", None)
        if job_service is not None:
            try:
                await run_in_threadpool(job_service.health_check)
                database_available, database_error = True, None
            except JobPersistenceError:
                database_error = "PostgreSQL недоступен"
        if queue is not None:
            try:
                await run_in_threadpool(queue.ping)
                worker_registrations = await run_in_threadpool(
                    queue.worker_registrations
                )
                if worker_registrations:
                    worker_id, worker_ttl = worker_registrations[0]
                redis_available, redis_error = True, None
            except QueueUnavailableError:
                redis_error = "Redis недоступен"
        worker_registered = bool(worker_id and worker_ttl and worker_ttl > 0)
        healthy = database_available and redis_available and worker_registered
        payload = HealthResponse(
            status="ok" if healthy else "unhealthy",
            service="specification-checker",
            api_version=API_VERSION,
            checker_initialized=False,
            ocr_backend=None,
            database_available=database_available,
            database_error=database_error,
            redis_available=redis_available,
            redis_error=redis_error,
            worker_registered=worker_registered,
            worker_id=worker_id,
            worker_registration_ttl_seconds=worker_ttl,
            worker_count=len(worker_registrations),
            worker_ids=[item[0] for item in worker_registrations],
        )
        return JSONResponse(
            status_code=200 if healthy else 503,
            content=payload.model_dump(mode="json"),
        )

    @router.get(
        "/checks/{job_id}",
        response_model=CheckJobResponse,
        responses=ERROR_RESPONSES,
        summary="Read a persisted check job",
    )
    async def get_check(
        job_id: UUID,
        job_service: Annotated[JobPersistenceService, Depends(get_job_service)],
    ) -> CheckJobResponse:
        try:
            record = await run_in_threadpool(job_service.get, str(job_id))
        except JobPersistenceError as exc:
            raise HTTPException(
                status_code=503,
                detail=_error_detail("database_unavailable", "Не удалось прочитать задание из PostgreSQL.", str(job_id)),
            ) from exc
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=_error_detail("job_not_found", "Задание с таким job_id не найдено.", str(job_id)),
            )
        return _job_response(record)

    @router.post(
        "/checks",
        response_model=CheckJobResponse,
        status_code=202,
        responses=ERROR_RESPONSES,
        summary="Submit one specification check",
        description=(
            "Accepts a JSON/Base64 attachment bundle or the legacy multipart upload. For JSON "
            "bundles, only the signed order specification is selected; unrelated attachments "
            "are ignored. The selected input is stored and job_id is enqueued without running "
            "OCR in FastAPI."
        ),
        openapi_extra=CHECK_REQUEST_OPENAPI,
    )
    async def create_check(
        request: Request,
        job_service: Annotated[JobPersistenceService, Depends(get_job_service)],
        queue: Annotated[JobQueue, Depends(get_job_queue)],
    ) -> CheckJobResponse:
        job_id = str(uuid4())
        created_at = datetime.now(timezone.utc)
        workspace = JobWorkspace(settings.jobs_dir, job_id)
        record_created = False
        transport: str | None = None
        submission: CheckRequestForm | CheckRequestJSON | None = None
        upload: UploadFile | None = None
        try:
            transport, submission, upload = await _read_submission(request, settings)
            if transport == "multipart":
                assert isinstance(submission, CheckRequestForm)
                assert upload is not None
                suffix = _validate_filename(upload.filename)
                reference = _normalize_reference(
                    _parse_reference(submission.reference, settings)
                )
                document_path = (workspace.path / f"document{suffix}").resolve()
                _, document_sha256 = await _save_document(
                    upload, document_path, suffix, settings
                )
                external_id = submission.external_id
                original_filename = upload.filename
            else:
                assert isinstance(submission, CheckRequestJSON)
                reference = _normalize_reference(
                    _validate_reference(submission.reference, settings)
                )
                selected_documents = await run_in_threadpool(
                    _select_json_documents,
                    submission,
                    settings,
                    application_number=reference["application_number"],
                )
                document_path, document_sha256, original_filename = await run_in_threadpool(
                    _save_base64_documents,
                    selected_documents,
                    workspace,
                )
                external_id = submission.external_id
            workspace.write_json("reference.json", reference)
            record = await run_in_threadpool(
                job_service.create_processing,
                job_id=job_id,
                external_id=external_id,
                created_at=created_at,
                started_at=created_at,
                document_path=str(document_path),
                document_sha256=document_sha256,
            )
            record_created = True
            try:
                workspace.update_manifest(
                    status=JobStatus.PROCESSING.value,
                    created_at=created_at,
                    external_id=external_id,
                    original_filename=original_filename,
                )
            except OSError:
                LOGGER.warning("Could not write manifest for job %s", job_id, exc_info=True)
            try:
                await run_in_threadpool(queue.enqueue, job_id)
            except Exception as exc:
                completed_at = datetime.now(timezone.utc)
                details = {
                    **_error_detail("queue_enqueue_failed", "Не удалось поставить задание в Redis.", job_id),
                    "exception_type": type(exc).__name__,
                    "technical_message": str(exc),
                }
                failed = await run_in_threadpool(
                    job_service.fail,
                    job_id,
                    error_code="queue_enqueue_failed",
                    error_message="Не удалось поставить задание в Redis.",
                    error_details=details,
                    completed_at=completed_at,
                    processing_time_seconds=_elapsed(created_at, completed_at),
                )
                try:
                    workspace.update_manifest(
                        status=JobStatus.FAILED.value,
                        created_at=created_at,
                        external_id=external_id,
                        original_filename=original_filename,
                        error=details,
                    )
                except OSError:
                    LOGGER.warning("Could not update failed manifest for job %s", job_id, exc_info=True)
                raise HTTPException(status_code=503, detail=_error_detail(failed.error_code or "queue_enqueue_failed", failed.error_message or "Redis недоступен.", job_id)) from exc
            return _job_response(record)
        except RequestProblem as exc:
            archived = False
            if settings.rejected_requests_enabled:
                try:
                    quarantine_path = await archive_rejected_request(
                        settings=settings,
                        request=request,
                        workspace_path=workspace.path,
                        request_id=job_id,
                        status_code=exc.status_code,
                        error_code=exc.code,
                        error_message=exc.message,
                        submission=submission,
                        upload=upload,
                    )
                    archived = True
                    LOGGER.warning(
                        "Rejected API request %s archived at %s: HTTP %s %s",
                        job_id,
                        quarantine_path,
                        exc.status_code,
                        exc.code,
                    )
                except Exception:
                    LOGGER.exception(
                        "Could not archive rejected API request %s", job_id
                    )
            if not archived:
                shutil.rmtree(workspace.path, ignore_errors=True)
            raise HTTPException(status_code=exc.status_code, detail=_error_detail(exc.code, exc.message, None)) from exc
        except JobPersistenceError as exc:
            if not record_created:
                shutil.rmtree(workspace.path, ignore_errors=True)
            raise HTTPException(
                status_code=503,
                detail=_error_detail("database_unavailable", "Не удалось сохранить состояние задания в PostgreSQL.", job_id),
            ) from exc
        finally:
            if upload is not None:
                await upload.close()

    application.include_router(router)
    return application


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "app.api.app:app",
        host=os.getenv("SPECIFICATION_CHECKER_HOST", "127.0.0.1"),
        port=int(os.getenv("SPECIFICATION_CHECKER_PORT", "8000")),
    )
