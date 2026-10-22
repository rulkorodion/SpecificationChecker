"""Short-lived, bounded quarantine for HTTP requests rejected before queuing."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.storage import APISettings

LOGGER = logging.getLogger(__name__)
QUARANTINE_DIRNAME = "_rejected"
ALLOWED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _folder_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def cleanup_rejected_requests(
    settings: APISettings,
    *,
    preserve_request_id: str | None = None,
) -> dict[str, int]:
    """Remove expired/oldest quarantine entries and enforce hard count/size caps."""

    root = settings.jobs_dir / QUARANTINE_DIRNAME
    if not root.is_dir():
        return {"removed": 0, "freed_bytes": 0}

    removed = 0
    freed = 0
    cutoff = datetime.now(timezone.utc) - timedelta(
        hours=settings.rejected_request_retention_hours
    )
    entries: list[tuple[Path, float, int]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            modified = path.stat().st_mtime
            size = _folder_size(path)
        except OSError:
            continue
        if datetime.fromtimestamp(modified, timezone.utc) < cutoff and path.name != preserve_request_id:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
            freed += size
        else:
            entries.append((path, modified, size))

    entries.sort(key=lambda item: item[1])
    total_size = sum(item[2] for item in entries)
    while entries and (
        len(entries) > settings.rejected_request_max_entries
        or total_size > settings.rejected_request_max_total_size_bytes
    ):
        candidate_index = next(
            (
                index
                for index, item in enumerate(entries)
                if item[0].name != preserve_request_id
            ),
            None,
        )
        if candidate_index is None:
            break
        path, _, size = entries.pop(candidate_index)
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
        freed += size
        total_size -= size
    return {"removed": removed, "freed_bytes": freed}


def _decode_base64_best_effort(value: Any, remaining_bytes: int) -> bytes:
    if not isinstance(value, str):
        raise ValueError("content_base64 is not a string")
    payload = value.strip()
    if payload.lower().startswith("data:"):
        if "," not in payload:
            raise ValueError("invalid data URI")
        payload = payload.split(",", 1)[1]
    compact = "".join(payload.split())
    if not compact:
        raise ValueError("empty Base64")
    if (len(compact) * 3) // 4 > remaining_bytes + 2:
        raise ValueError("decoded document exceeds quarantine per-request limit")
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid Base64") from exc
    if not decoded:
        raise ValueError("empty decoded document")
    if len(decoded) > remaining_bytes:
        raise ValueError("decoded document exceeds quarantine per-request limit")
    return decoded


def _suffix_for(filename: Any, content: bytes) -> str:
    if isinstance(filename, str):
        suffix = Path(filename.replace("\\", "/")).suffix.lower()
        if suffix in ALLOWED_SUFFIXES:
            return suffix
    signatures = (
        (b"%PDF-", ".pdf"),
        (b"\xff\xd8\xff", ".jpg"),
        (b"\x89PNG\r\n\x1a\n", ".png"),
    )
    return next((suffix for signature, suffix in signatures if content.startswith(signature)), ".bin")


def _existing_document_files(path: Path) -> list[Path]:
    return [
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in ALLOWED_SUFFIXES
    ]


def _archive_sync(
    *,
    settings: APISettings,
    workspace_path: Path,
    request_id: str,
    rejected_at: datetime,
    status_code: int,
    error_code: str,
    error_message: str,
    source_ip: str | None,
    transport_content_type: str | None,
    parsed_payload: Any,
    submission_payload: Any,
    upload_filename: str | None,
    upload_content_type: str | None,
    upload_bytes: bytes | None,
    raw_body: bytes | None,
) -> Path:
    root = settings.jobs_dir / QUARANTINE_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    cleanup_rejected_requests(settings)

    temporary = root / f".{request_id}.tmp"
    destination = root / request_id
    if temporary.exists() or destination.exists():
        raise FileExistsError(f"Quarantine request already exists: {request_id}")
    if workspace_path.exists():
        workspace_path.replace(temporary)
    else:
        temporary.mkdir()
    try:
        temporary.chmod(0o700)
    except OSError:
        pass

    payload = submission_payload if isinstance(submission_payload, dict) else parsed_payload
    external_id = payload.get("external_id") if isinstance(payload, dict) else None
    reference = payload.get("reference") if isinstance(payload, dict) else None
    if isinstance(reference, str):
        try:
            reference = json.loads(reference)
        except (json.JSONDecodeError, TypeError):
            pass
    reference_saved = False
    if reference is not None and not (temporary / "reference.json").exists():
        _write_json(temporary / "reference.json", reference)
        reference_saved = True
    elif (temporary / "reference.json").exists():
        reference_saved = True

    document_metadata: list[dict[str, Any]] = []
    existing_documents = _existing_document_files(temporary)
    for item in existing_documents:
        document_metadata.append(
            {
                "saved_as": item.relative_to(temporary).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
                "recovered_from": "request_workspace",
            }
        )

    documents = payload.get("documents") if isinstance(payload, dict) else None
    needs_raw_body = False
    if not existing_documents and isinstance(documents, list):
        rejected_documents = temporary / "documents"
        rejected_documents.mkdir(exist_ok=True)
        remaining = settings.max_upload_size_bytes
        for index, item in enumerate(documents, start=1):
            metadata: dict[str, Any] = {
                "index": index,
                "filename": item.get("filename") if isinstance(item, dict) else None,
                "declared_content_type": item.get("content_type") if isinstance(item, dict) else None,
            }
            try:
                if not isinstance(item, dict):
                    raise ValueError("document entry is not an object")
                content = _decode_base64_best_effort(
                    item.get("content_base64"), remaining
                )
                remaining -= len(content)
                suffix = _suffix_for(item.get("filename"), content)
                saved = rejected_documents / f"document-{index:03d}{suffix}"
                saved.write_bytes(content)
                metadata.update(
                    {
                        "saved_as": saved.relative_to(temporary).as_posix(),
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "recovered_from": "json_base64",
                    }
                )
            except (OSError, ValueError) as exc:
                metadata["recovery_error"] = str(exc)
                needs_raw_body = True
            document_metadata.append(metadata)
    elif not existing_documents and upload_bytes is not None:
        suffix = _suffix_for(upload_filename, upload_bytes)
        saved = temporary / f"document{suffix}"
        saved.write_bytes(upload_bytes)
        document_metadata.append(
            {
                "filename": upload_filename,
                "declared_content_type": upload_content_type,
                "saved_as": saved.relative_to(temporary).as_posix(),
                "size_bytes": len(upload_bytes),
                "sha256": hashlib.sha256(upload_bytes).hexdigest(),
                "recovered_from": "multipart_upload",
            }
        )
    elif not existing_documents:
        needs_raw_body = True

    raw_body_saved = False
    raw_body_limit = (
        (settings.max_upload_size_bytes * 4) // 3
        + settings.max_reference_size_bytes
        + 1024 * 1024
    )
    if needs_raw_body and raw_body is not None and len(raw_body) <= raw_body_limit:
        raw_name = (
            "request.json"
            if (transport_content_type or "").split(";", 1)[0].strip().lower()
            == "application/json"
            else "request-body.bin"
        )
        (temporary / raw_name).write_bytes(raw_body)
        raw_body_saved = True

    metadata = {
        "schema_version": "1.0",
        "request_id": request_id,
        "rejected_at": rejected_at.isoformat(),
        "http_status": status_code,
        "error": {"code": error_code, "message": error_message},
        "source_ip": source_ip,
        "request_content_type": transport_content_type,
        "external_id": external_id,
        "reference_saved": reference_saved,
        "documents": document_metadata,
        "raw_body_saved": raw_body_saved,
        "retention_hours": settings.rejected_request_retention_hours,
    }
    _write_json(temporary / "rejection.json", metadata)
    for item in temporary.rglob("*"):
        try:
            item.chmod(0o700 if item.is_dir() else 0o600)
        except OSError:
            pass
    temporary.replace(destination)
    cleanup_rejected_requests(settings, preserve_request_id=request_id)
    return destination


async def archive_rejected_request(
    *,
    settings: APISettings,
    request: Request,
    workspace_path: Path,
    request_id: str,
    status_code: int,
    error_code: str,
    error_message: str,
    submission: Any = None,
    upload: UploadFile | None = None,
) -> Path:
    """Preserve recoverable request data without creating a job or queue entry."""

    parsed_payload = getattr(request.state, "parsed_json_body", None)
    submission_payload = (
        submission.model_dump(mode="python")
        if hasattr(submission, "model_dump")
        else None
    )
    upload_bytes: bytes | None = None
    upload_filename: str | None = None
    upload_content_type: str | None = None
    if upload is not None:
        upload_filename = upload.filename
        upload_content_type = upload.content_type
        try:
            await upload.seek(0)
            candidate = await upload.read(settings.max_upload_size_bytes + 1)
            if len(candidate) <= settings.max_upload_size_bytes:
                upload_bytes = candidate
        except Exception:
            LOGGER.warning("Could not reread rejected multipart upload", exc_info=True)

    raw_body: bytes | None = None
    if parsed_payload is not None:
        try:
            raw_body = await request.body()
        except Exception:
            raw_body = None
    elif upload is None:
        content_length = request.headers.get("content-length")
        raw_body_limit = (
            (settings.max_upload_size_bytes * 4) // 3
            + settings.max_reference_size_bytes
            + 1024 * 1024
        )
        if not content_length or (
            content_length.isdigit() and int(content_length) <= raw_body_limit
        ):
            try:
                raw_body = await request.body()
            except Exception:
                raw_body = None

    return await run_in_threadpool(
        _archive_sync,
        settings=settings,
        workspace_path=workspace_path,
        request_id=request_id,
        rejected_at=datetime.now(timezone.utc),
        status_code=status_code,
        error_code=error_code,
        error_message=error_message,
        source_ip=request.client.host if request.client else None,
        transport_content_type=request.headers.get("content-type"),
        parsed_payload=parsed_payload,
        submission_payload=submission_payload,
        upload_filename=upload_filename,
        upload_content_type=upload_content_type,
        upload_bytes=upload_bytes,
        raw_body=raw_body,
    )
