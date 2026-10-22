from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.storage import APISettings
from app.job_storage import JobStorage
from app.persistence import Base, Database, JobPersistenceService, JobStatus
from app.queueing import QueueSettings, QueueUnavailableError


def valid_reference(**overrides):
    payload = {
        "specification_id": "42",
        "application_number": "208341",
        "total_quantity": "0.123",
        "total_with_vat": "8902.34",
        "transportation_costs_included": False,
        "delivery_deadline": "2026-08-30",
        "delivery_type": "PICKUP",
        "payment_type": "PREPAYMENT",
        "deferment_days": None,
    }
    payload.update(overrides)
    return payload


def image_bytes(image_format: str) -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (2, 2), color=(255, 255, 255))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


class FakeQueue:
    def __init__(self) -> None:
        self.settings = QueueSettings(redis_url="redis://test")
        self.pending: list[str] = []
        self.processing: list[str] = []
        self.enqueue_error = False
        self.ping_error = False
        self.worker_id: str | None = "worker-test"

    def ping(self):
        if self.ping_error:
            raise QueueUnavailableError("offline")

    def enqueue(self, job_id):
        if self.enqueue_error:
            raise QueueUnavailableError("offline")
        self.pending.append(job_id)

    def dequeue(self, worker_id, timeout_seconds=None):
        if not self.pending:
            return None
        value = self.pending.pop(0)
        self.processing.append(value)
        return value

    def acknowledge(self, job_id, worker_id):
        if job_id in self.processing:
            self.processing.remove(job_id)

    def requeue(self, job_id, worker_id):
        self.acknowledge(job_id, worker_id)
        self.pending.append(job_id)

    def recover_inflight(self, worker_id, include_current=False):
        count = len(self.processing)
        self.pending.extend(self.processing)
        self.processing.clear()
        return count

    def acquire_worker(self, worker_id):
        self.worker_id = worker_id
        return True

    def refresh_worker(self, worker_id):
        return self.worker_id == worker_id

    def release_worker(self, worker_id):
        if self.worker_id == worker_id:
            self.worker_id = None

    def worker_registration(self):
        return (self.worker_id, 60) if self.worker_id else (None, None)

    def worker_registrations(self):
        return [(self.worker_id, 60)] if self.worker_id else []

    def close(self):
        pass


@pytest.fixture
def api_client(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'jobs.sqlite3'}")
    Base.metadata.create_all(database.engine)
    job_service = JobPersistenceService(database)
    queue = FakeQueue()
    forbidden_checker_calls: list[bool] = []
    settings = APISettings(
        jobs_dir=tmp_path / "jobs",
        max_upload_size_bytes=1024,
        max_reference_size_bytes=1024,
        upload_chunk_size_bytes=16,
        rejected_requests_enabled=False,
    )
    application = create_app(
        service_factory=lambda: forbidden_checker_calls.append(True),
        job_service_factory=lambda: job_service,
        queue_factory=lambda: queue,
        api_settings=settings,
    )
    with TestClient(application) as client:
        yield client, queue, forbidden_checker_calls, settings, job_service


@pytest.fixture
def quarantine_api_client(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'jobs.sqlite3'}")
    Base.metadata.create_all(database.engine)
    job_service = JobPersistenceService(database)
    queue = FakeQueue()
    settings = APISettings(
        jobs_dir=tmp_path / "jobs",
        max_upload_size_bytes=4096,
        max_reference_size_bytes=1024,
        upload_chunk_size_bytes=16,
        rejected_requests_enabled=True,
        rejected_request_retention_hours=24,
        rejected_request_max_entries=10,
        rejected_request_max_total_size_bytes=1024 * 1024,
    )
    application = create_app(
        job_service_factory=lambda: job_service,
        queue_factory=lambda: queue,
        api_settings=settings,
    )
    with TestClient(application) as client:
        yield client, queue, settings


@pytest.mark.integration
def test_post_creates_processing_enqueues_and_get_is_consistent(api_client):
    client, queue, checker_calls, settings, job_service = api_client
    pdf = b"%PDF-1.4\n"
    response = client.post(
        "/api/v1/checks",
        files={"document": ("specification.pdf", pdf, "application/pdf")},
        data={"reference": json.dumps(valid_reference()), "external_id": "1C-42"},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "PROCESSING"
    assert payload["result"] is None
    assert queue.pending == [payload["job_id"]]
    assert checker_calls == []
    record = job_service.get(payload["job_id"])
    assert record is not None and record.status is JobStatus.PROCESSING
    assert record.document_sha256 == hashlib.sha256(pdf).hexdigest()
    assert len(record.document_sha256) == 64
    assert record.created_at.utcoffset().total_seconds() == 0
    assert Path(record.document_path).is_file()
    assert json.loads((settings.jobs_dir / payload["job_id"] / "reference.json").read_text("utf-8")) == valid_reference()
    loaded = client.get(f"/api/v1/checks/{payload['job_id']}")
    assert loaded.status_code == 200
    assert loaded.json() == payload


@pytest.mark.integration
def test_json_base64_post_creates_the_same_processing_job(api_client):
    client, queue, checker_calls, settings, job_service = api_client
    pdf = b"%PDF-1.4\njson-transport"
    response = client.post(
        "/api/v1/checks",
        json={
            "external_id": "1C-json-42",
            "documents": [
                {
                    "filename": "Подписанная_спецификация_к_заказу_N208341.pdf",
                    "content_type": "application/pdf",
                    "content_base64": base64.b64encode(pdf).decode("ascii"),
                }
            ],
            "reference": valid_reference(),
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["external_id"] == "1C-json-42"
    assert payload["status"] == "PROCESSING"
    assert queue.pending == [payload["job_id"]]
    assert checker_calls == []
    record = job_service.get(payload["job_id"])
    document = settings.jobs_dir / payload["job_id"] / "document.pdf"
    assert record is not None and Path(record.document_path) == document.resolve()
    assert document.read_bytes() == pdf
    assert record.document_sha256 == hashlib.sha256(pdf).hexdigest()
    source_manifest = json.loads(
        (settings.jobs_dir / payload["job_id"] / "source_documents.json").read_text("utf-8")
    )
    assert source_manifest["documents"][0]["filename"] == "Подписанная_спецификация_к_заказу_N208341.pdf"


@pytest.mark.integration
@pytest.mark.parametrize(
    "unrelated_documents",
    [
        [
            {
                "filename": "Неподписанная СпецификацияN00000254068",
                "content_type": "application/pdf",
                "content_base64": base64.b64encode(b"%PDF-1.4\nunsigned").decode("ascii"),
            }
        ],
        [
            {
                "filename": r"U:\\1C\\служебный документ.docx",
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "content_base64": "not-base64-and-must-be-ignored",
            },
            {
                "filename": "Неподписанная СпецификацияN00000254068",
                "content_base64": base64.b64encode(b"%PDF-1.4\nunsigned").decode("ascii"),
            },
        ],
    ],
    ids=["unsigned-pdf-without-extension", "docx-and-unsigned-pdf"],
)
def test_json_bundle_ignores_unrelated_attachments(api_client, unrelated_documents):
    client, queue, _, settings, job_service = api_client
    signed_pdf = b"%PDF-1.4\nsigned-specification"
    signed_filename = r"U:\\1C\\Подписанная_спецификация_к_заказу_N208341.pdf"
    response = client.post(
        "/api/v1/checks",
        json={
            "external_id": "1C-real-bundle",
            "documents": unrelated_documents
            + [
                {
                    "filename": signed_filename,
                    "content_type": "application/octet-stream",
                    "content_base64": base64.b64encode(signed_pdf).decode("ascii"),
                }
            ],
            "reference": valid_reference(),
        },
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert queue.pending == [job_id]
    record = job_service.get(job_id)
    assert record is not None
    assert Path(record.document_path).read_bytes() == signed_pdf
    source_manifest = json.loads(
        (settings.jobs_dir / job_id / "source_documents.json").read_text("utf-8")
    )
    assert len(source_manifest["documents"]) == 1
    assert source_manifest["documents"][0]["filename"] == signed_filename
    assert source_manifest["documents"][0]["source_index"] == len(unrelated_documents) + 1
    assert source_manifest["documents"][0]["content_type"] == "application/pdf"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("image_format", "filename", "declared_content_type", "expected_suffix"),
    [
        (
            "JPEG",
            r"U:\\1C\\ПОДПИСАННАЯ СПЕЦИФИКАЦИЯ К ЗАКАЗУ N208341.JPG",
            None,
            ".jpg",
        ),
        (
            "PNG",
            "Подписанная_спецификация_к_заказу_N208341.png",
            "application/pdf",
            ".png",
        ),
    ],
)
def test_json_bundle_accepts_signed_images_by_actual_content(
    api_client, image_format, filename, declared_content_type, expected_suffix
):
    client, queue, _, settings, job_service = api_client
    content = image_bytes(image_format)
    document = {
        "filename": filename,
        "content_base64": base64.b64encode(content).decode("ascii"),
    }
    if declared_content_type is not None:
        document["content_type"] = declared_content_type
    response = client.post(
        "/api/v1/checks",
        json={
            "documents": [
                {
                    "filename": "Данные заказа.docx",
                    "content_base64": base64.b64encode(b"PK\x03\x04noise").decode("ascii"),
                },
                document,
            ],
            "reference": valid_reference(),
        },
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert queue.pending == [job_id]
    record = job_service.get(job_id)
    assert record is not None
    assert Path(record.document_path).suffix == expected_suffix
    assert Path(record.document_path).read_bytes() == content
    source_manifest = json.loads(
        (settings.jobs_dir / job_id / "source_documents.json").read_text("utf-8")
    )
    assert source_manifest["documents"][0]["source_index"] == 2


@pytest.mark.integration
def test_reference_application_number_selects_the_matching_signed_pdf(api_client):
    client, queue, _, settings, _ = api_client
    selected = b"%PDF-1.4\nselected"
    response = client.post(
        "/api/v1/checks",
        json={
            "documents": [
                {
                    "filename": "Подписанная_спецификация_к_заказу_N111111.pdf",
                    "content_base64": base64.b64encode(b"%PDF-1.4\nother").decode("ascii"),
                },
                {
                    "filename": "Подписанная спецификация к заказу N208341.pdf",
                    "content_base64": base64.b64encode(selected).decode("ascii"),
                },
            ],
            "reference": valid_reference(),
        },
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert queue.pending == [job_id]
    assert (settings.jobs_dir / job_id / "document.pdf").read_bytes() == selected
    source_manifest = json.loads(
        (settings.jobs_dir / job_id / "source_documents.json").read_text("utf-8")
    )
    assert source_manifest["documents"][0]["source_index"] == 2


@pytest.mark.integration
def test_json_reference_accepts_decimal_numbers_and_stores_canonical_strings(api_client):
    client, queue, _, settings, _ = api_client
    pdf = b"%PDF-1.4\nnumber-reference"
    response = client.post(
        "/api/v1/checks",
        json={
            "external_id": "1C-number-reference",
            "documents": [
                {
                    "filename": "Подписанная_спецификация_к_заказу_N208341.pdf",
                    "content_base64": base64.b64encode(pdf).decode("ascii"),
                }
            ],
            "reference": valid_reference(
                total_quantity=21.109,
                total_with_vat=1467075.5,
            ),
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert queue.pending == [payload["job_id"]]
    stored = json.loads(
        (settings.jobs_dir / payload["job_id"] / "reference.json").read_text("utf-8")
    )
    assert stored["total_quantity"] == "21.109"
    assert stored["total_with_vat"] == "1467075.5"


@pytest.mark.integration
def test_signed_image_pages_are_merged_into_one_pdf_job(api_client):
    fitz = pytest.importorskip("fitz")
    client, queue, _, settings, job_service = api_client
    first = image_bytes("PNG")
    second = image_bytes("JPEG")
    assert len(first) + len(second) <= settings.max_upload_size_bytes
    response = client.post(
        "/api/v1/checks",
        json={
            "external_id": "1C-multiple",
            "documents": [
                {
                    "filename": r"U:\\1C\\Подписанная спецификация к заказу N208341 стр 1.png",
                    "content_type": "application/octet-stream",
                    "content_base64": base64.b64encode(first).decode("ascii"),
                },
                {
                    "filename": "подписанная_спецификация_к_заказу_N208341_стр_2.jpg",
                    "content_base64": base64.b64encode(second).decode("ascii"),
                },
            ],
            "reference": valid_reference(),
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert queue.pending == [payload["job_id"]]
    record = job_service.get(payload["job_id"])
    canonical = settings.jobs_dir / payload["job_id"] / "document.pdf"
    assert record is not None and Path(record.document_path) == canonical.resolve()
    assert record.document_sha256 == hashlib.sha256(canonical.read_bytes()).hexdigest()
    merged = fitz.open(canonical)
    try:
        assert merged.page_count == 2
    finally:
        merged.close()
    source_manifest = json.loads(
        (settings.jobs_dir / payload["job_id"] / "source_documents.json").read_text("utf-8")
    )
    assert [item["filename"] for item in source_manifest["documents"]] == [
        r"U:\\1C\\Подписанная спецификация к заказу N208341 стр 1.png",
        "подписанная_спецификация_к_заказу_N208341_стр_2.jpg",
    ]
    assert [item["source_index"] for item in source_manifest["documents"]] == [1, 2]
    assert [item["content_type"] for item in source_manifest["documents"]] == [
        "image/png",
        "image/jpeg",
    ]
    worker_input = JobStorage(settings.jobs_dir).load_input(record)
    assert worker_input.document_path == canonical.resolve()
    assert worker_input.reference == valid_reference()


@pytest.mark.integration
def test_json_document_array_limits_are_enforced(api_client):
    client, queue, _, settings, _ = api_client
    encoded = base64.b64encode(b"%PDF-1.4").decode("ascii")
    too_many = client.post(
        "/api/v1/checks",
        json={
            "documents": [
                {
                    "filename": f"Подписанная_спецификация_к_заказу_N208341_{index}.pdf",
                    "content_base64": encoded,
                }
                for index in range(settings.max_documents_per_job + 1)
            ],
            "reference": valid_reference(),
        },
    )
    assert too_many.status_code == 413
    assert too_many.json()["detail"]["code"] == "too_many_documents"

    oversized_part = b"%PDF-" + b"x" * (settings.max_upload_size_bytes // 2)
    total_too_large = client.post(
        "/api/v1/checks",
        json={
            "documents": [
                {
                    "filename": f"Подписанная_спецификация_к_заказу_N208341_{index}.pdf",
                    "content_base64": base64.b64encode(oversized_part).decode("ascii"),
                }
                for index in (1, 2)
            ],
            "reference": valid_reference(),
        },
    )
    assert total_too_large.status_code == 413
    assert total_too_large.json()["detail"]["code"] == "document_too_large"
    assert queue.pending == []
    assert not settings.jobs_dir.exists() or list(settings.jobs_dir.iterdir()) == []


@pytest.mark.integration
@pytest.mark.parametrize(
    ("document", "status", "code"),
    [
        (
            {
                "filename": "Подписанная_спецификация_к_заказу_N208341.pdf",
                "content_base64": "not-base64!",
            },
            422,
            "invalid_document_base64",
        ),
        (
            {
                "filename": "Подписанная_спецификация_к_заказу_N208341.pdf",
                "content_base64": base64.b64encode(b"not-pdf").decode("ascii"),
            },
            415,
            "unsupported_document_format",
        ),
    ],
)
def test_invalid_json_documents_are_rejected_before_database(api_client, document, status, code):
    client, queue, _, settings, _ = api_client
    response = client.post(
        "/api/v1/checks",
        json={"documents": [document], "reference": valid_reference()},
    )
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert queue.pending == []
    assert not settings.jobs_dir.exists() or list(settings.jobs_dir.iterdir()) == []


@pytest.mark.integration
def test_health_reports_database_redis_and_worker_separately(api_client):
    client, queue, _, _, _ = api_client
    healthy = client.get("/health")
    assert healthy.status_code == 200
    body = healthy.json()
    assert body["database_available"] is True
    assert body["redis_available"] is True
    assert body["worker_registered"] is True
    assert body["worker_count"] == 1
    assert body["worker_ids"] == ["worker-test"]
    assert body["checker_initialized"] is False

    queue.worker_id = None
    no_worker = client.get("/health")
    assert no_worker.status_code == 503
    assert no_worker.json()["redis_available"] is True
    assert no_worker.json()["worker_registered"] is False

    queue.ping_error = True
    no_redis = client.get("/health")
    assert no_redis.status_code == 503
    assert no_redis.json()["database_available"] is True
    assert no_redis.json()["redis_available"] is False


@pytest.mark.integration
def test_redis_enqueue_error_marks_job_failed(api_client):
    client, queue, _, _, job_service = api_client
    queue.enqueue_error = True
    response = client.post(
        "/api/v1/checks",
        files={"document": ("spec.pdf", b"%PDF-1.4", "application/pdf")},
        data={"reference": json.dumps(valid_reference())},
    )
    assert response.status_code == 503
    job_id = response.json()["detail"]["job_id"]
    record = job_service.get(job_id)
    assert record is not None and record.status is JobStatus.FAILED
    assert record.error_code == "queue_enqueue_failed"
    assert record.error_details["exception_type"] == "QueueUnavailableError"
    assert record.completed_at.utcoffset().total_seconds() == 0


@pytest.mark.integration
@pytest.mark.parametrize(
    ("filename", "content", "status", "code"),
    [
        ("spec.exe", b"MZ", 415, "unsupported_document_format"),
        ("spec.pdf", b"not-pdf", 415, "document_signature_mismatch"),
        ("spec.pdf", b"%PDF-" + b"x" * 1024, 413, "document_too_large"),
    ],
)
def test_invalid_upload_is_rejected_before_database_and_queue(api_client, filename, content, status, code):
    client, queue, _, settings, _ = api_client
    response = client.post(
        "/api/v1/checks",
        files={"document": (filename, content, "application/octet-stream")},
        data={"reference": json.dumps(valid_reference())},
    )
    assert response.status_code == status
    assert response.json()["detail"] == {"code": code, "message": response.json()["detail"]["message"], "job_id": None}
    assert queue.pending == []
    assert not settings.jobs_dir.exists() or list(settings.jobs_dir.iterdir()) == []


@pytest.mark.integration
def test_invalid_reference_and_missing_job(api_client):
    client, queue, _, _, _ = api_client
    invalid = client.post(
        "/api/v1/checks",
        files={"document": ("spec.pdf", b"%PDF-1.4", "application/pdf")},
        data={"reference": "{bad}"},
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["job_id"] is None
    assert queue.pending == []
    missing = client.get("/api/v1/checks/00000000-0000-0000-0000-000000000000")
    assert missing.status_code == 404


@pytest.mark.integration
def test_semantically_invalid_reference_is_rejected_before_queue(api_client):
    client, queue, _, settings, _ = api_client
    response = client.post(
        "/api/v1/checks",
        json={
            "documents": [
                {
                    "filename": "spec.pdf",
                    "content_base64": base64.b64encode(b"%PDF-1.4").decode("ascii"),
                }
            ],
            "reference": valid_reference(total_quantity="not-a-number"),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_reference"
    assert "total_quantity" in response.json()["detail"]["message"]
    assert queue.pending == []
    assert not settings.jobs_dir.exists() or list(settings.jobs_dir.iterdir()) == []


@pytest.mark.integration
def test_bundle_without_signed_specification_returns_415_and_is_quarantined(
    quarantine_api_client,
):
    client, queue, settings = quarantine_api_client
    first = b"%PDF-1.4\nunsigned"
    second = b"PK\x03\x04docx"
    reference = valid_reference()
    response = client.post(
        "/api/v1/checks",
        json={
            "external_id": "1C-no-signed-specification",
            "documents": [
                {
                    "filename": "Неподписанная СпецификацияN00000254068",
                    "content_base64": base64.b64encode(first).decode("ascii"),
                },
                {
                    "filename": "Данные заказа.docx",
                    "content_base64": base64.b64encode(second).decode("ascii"),
                },
            ],
            "reference": reference,
        },
    )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "signed_specification_not_found"
    assert queue.pending == []
    quarantine_entries = list((settings.jobs_dir / "_rejected").iterdir())
    assert len(quarantine_entries) == 1
    quarantine = quarantine_entries[0]
    rejection = json.loads((quarantine / "rejection.json").read_text("utf-8"))
    assert rejection["http_status"] == 415
    assert rejection["error"]["code"] == "signed_specification_not_found"
    assert rejection["external_id"] == "1C-no-signed-specification"
    assert rejection["reference_saved"] is True
    assert len(rejection["documents"]) == 2
    assert (quarantine / "documents" / "document-001.pdf").read_bytes() == first
    assert (quarantine / "documents" / "document-002.bin").read_bytes() == second
    assert json.loads((quarantine / "reference.json").read_text("utf-8")) == reference


@pytest.mark.integration
def test_multiple_ambiguous_signed_pdfs_return_422_and_are_quarantined(
    quarantine_api_client,
):
    client, queue, settings = quarantine_api_client
    first = b"%PDF-1.4\nfirst"
    second = b"%PDF-1.4\nsecond"
    response = client.post(
        "/api/v1/checks",
        json={
            "external_id": "1C-ambiguous-signed-specification",
            "documents": [
                {
                    "filename": "Подписанная_спецификация_к_заказу_N111111.pdf",
                    "content_base64": base64.b64encode(first).decode("ascii"),
                },
                {
                    "filename": "Подписанная спецификация к заказу N222222.pdf",
                    "content_base64": base64.b64encode(second).decode("ascii"),
                },
            ],
            "reference": valid_reference(application_number="333333"),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "ambiguous_signed_specification"
    assert queue.pending == []
    quarantine_entries = list((settings.jobs_dir / "_rejected").iterdir())
    assert len(quarantine_entries) == 1
    quarantine = quarantine_entries[0]
    rejection = json.loads((quarantine / "rejection.json").read_text("utf-8"))
    assert rejection["http_status"] == 422
    assert rejection["error"]["code"] == "ambiguous_signed_specification"
    assert len(rejection["documents"]) == 2
    assert (quarantine / "documents" / "document-001.pdf").read_bytes() == first
    assert (quarantine / "documents" / "document-002.pdf").read_bytes() == second
    assert not any(path.name.endswith(".tmp") for path in quarantine.rglob("*"))


@pytest.mark.integration
def test_semantically_invalid_json_reference_is_still_quarantined_early(
    quarantine_api_client,
):
    client, queue, settings = quarantine_api_client
    pdf = b"%PDF-1.4\nrejected-reference"
    reference = valid_reference(total_quantity="not-a-number")
    response = client.post(
        "/api/v1/checks",
        json={
            "external_id": "1C-invalid-reference",
            "documents": [
                {
                    "filename": "Подписанная_спецификация_к_заказу_N208341.pdf",
                    "content_base64": base64.b64encode(pdf).decode("ascii"),
                }
            ],
            "reference": reference,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_reference"
    assert queue.pending == []
    quarantine_entries = list((settings.jobs_dir / "_rejected").iterdir())
    assert len(quarantine_entries) == 1
    quarantine = quarantine_entries[0]
    rejection = json.loads((quarantine / "rejection.json").read_text("utf-8"))
    assert rejection["error"]["code"] == "invalid_reference"
    assert (quarantine / "documents" / "document-001.pdf").read_bytes() == pdf


@pytest.mark.integration
def test_health_and_post_fail_closed_without_database(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    queue = FakeQueue()
    application = create_app(
        queue_factory=lambda: queue,
        api_settings=APISettings(jobs_dir=tmp_path / "jobs", max_upload_size_bytes=1024, max_reference_size_bytes=1024, upload_chunk_size_bytes=16, rejected_requests_enabled=False),
    )
    with TestClient(application) as client:
        health = client.get("/health")
        response = client.post(
            "/api/v1/checks",
            files={"document": ("spec.pdf", b"%PDF-1.4", "application/pdf")},
            data={"reference": json.dumps(valid_reference())},
        )
    assert health.status_code == 503
    assert health.json()["database_available"] is False
    assert health.json()["redis_available"] is True
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "database_unavailable"


@pytest.mark.integration
def test_openapi_exposes_asynchronous_swagger_contract(api_client):
    client, _, _, _, _ = api_client
    document = client.get("/openapi.json").json()
    operation = document["paths"]["/api/v1/checks"]["post"]
    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert "application/json" in operation["requestBody"]["content"]
    assert operation["responses"]["202"]["content"]["application/json"]["schema"]["$ref"].endswith("CheckJobResponse")
    assert "/api/v1/checks/{job_id}" in document["paths"]
