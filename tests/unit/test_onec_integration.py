from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.integrations import (
    MockOneCAdapter,
    OneCIntegrationService,
    OneCIntegrationSettings,
)
from app.integrations.service import DeliveryStatus
from app.job_storage import JobStorage
from app.persistence import Base, Database, JobPersistenceService, JobStatus
from app.queueing import QueueSettings
from app.worker import SpecificationWorker


REFERENCE = {
    "specification_id": "mock-1",
    "application_number": "42",
    "total_quantity": "12.177",
    "total_with_vat": "775944.27",
    "transportation_costs_included": True,
    "delivery_deadline": "2026-08-07",
    "delivery_type": "PICKUP",
    "payment_type": "PREPAYMENT",
    "deferment_days": None,
}


def check_result():
    return {
        "decision": "APPROVE",
        "extracted_values": {"application_number": "42"},
        "field_details": {},
        "checks": [
            {
                "code": "APPLICATION_NUMBER",
                "title": "Номер приложения",
                "status": "PASS",
                "expected": "42",
                "actual": "42",
                "confidence": 0.99,
                "message": "Совпадает",
                "page_number": 1,
                "bounding_box": None,
                "evidence": [],
                "is_required": True,
            }
        ],
        "reasons": ["Все обязательные проверки пройдены"],
        "comment_1c": "Спецификация согласована.",
        "confidence": {"overall": 0.99, "fields": {}},
        "metadata": {},
    }


class CountingMockAdapter(MockOneCAdapter):
    def __init__(self, settings, *, fail=False):
        super().__init__(settings)
        self.fail = fail
        self.send_calls = 0

    def send_result(self, payload):
        self.send_calls += 1
        if self.fail:
            raise ConnectionError("1C unavailable")
        return super().send_result(payload)


class Queue:
    def __init__(self):
        self.settings = QueueSettings(redis_url="redis://test", block_timeout_seconds=1)
        self.pending = []
        self.processing = []
    def ping(self): pass
    def enqueue(self, item): self.pending.append(item)
    def dequeue(self, worker_id, timeout_seconds=None):
        if not self.pending: return None
        item = self.pending.pop(0); self.processing.append(item); return item
    def acknowledge(self, item, worker_id):
        if item in self.processing: self.processing.remove(item)
    def requeue(self, item, worker_id): self.acknowledge(item, worker_id); self.pending.append(item)
    def recover_inflight(self, worker_id, include_current=False): return 0
    def acquire_worker(self, worker_id): return True
    def refresh_worker(self, worker_id): return True
    def release_worker(self, worker_id): pass
    def worker_registration(self): return "worker", 60
    def worker_registrations(self): return [("worker", 60)]
    def close(self): pass


class Checker:
    def __init__(self): self.calls = 0
    def check_specification(self, document, reference, runtime_config=None):
        self.calls += 1
        return check_result()


def make_persistence(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'jobs.sqlite'}")
    Base.metadata.create_all(database.engine)
    return JobPersistenceService(database)


def make_completed_record(tmp_path, service, storage, result_payload=None):
    job_id = str(uuid4())
    folder = storage.job_dir(job_id); folder.mkdir(parents=True)
    document = folder / "document.pdf"; document.write_bytes(b"%PDF-1.4")
    (folder / "reference.json").write_text(json.dumps(REFERENCE), "utf-8")
    now = datetime.now(timezone.utc)
    service.create_processing(
        job_id=job_id, external_id="1C-42", created_at=now, started_at=now,
        document_path=str(document), document_sha256=hashlib.sha256(document.read_bytes()).hexdigest(),
    )
    return service.complete(
        job_id, result=result_payload or check_result(), completed_at=now,
        processing_time_seconds=0.25,
    )


def test_mock_adapter_normalizes_reference_and_document(tmp_path):
    settings = OneCIntegrationSettings(mock_outbox_dir=tmp_path / "outbox")
    adapter = MockOneCAdapter(settings)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps({**REFERENCE, "check_date": "2026-08-04"}), "utf-8")
    document_path = tmp_path / "spec.pdf"; document_path.write_bytes(b"%PDF-1.4")

    reference = adapter.reference_to_internal(reference_path)
    document = adapter.get_document(document_path)

    assert reference.application_number == "42"
    assert str(reference.total_quantity) == "12.177"
    assert document.path == document_path.resolve()
    assert document.media_type == "application/pdf"
    assert document.sha256 == hashlib.sha256(b"%PDF-1.4").hexdigest()


def test_canonical_result_and_successful_mock_send(tmp_path):
    service = make_persistence(tmp_path); storage = JobStorage(tmp_path / "jobs")
    record = make_completed_record(tmp_path, service, storage)
    settings = OneCIntegrationSettings(enabled=True, mock_outbox_dir=tmp_path / "outbox")
    adapter = CountingMockAdapter(settings)
    integration = OneCIntegrationService(settings=settings, adapter=adapter, persistence=service)

    canonical = integration.canonical_result(record)
    state = integration.deliver(record)

    assert canonical.external_id == "1C-42"
    assert canonical.decision.value == "APPROVE"
    assert canonical.comment_1c == (
        f"ПРОВЕРКА СПЕЦИФИКАЦИИ №42\n"
        f"ID проверки: {record.job_id}\n\n"
        "1. Номер приложения: совпадает\n\n"
        "ИТОГОВОЕ РЕШЕНИЕ: СОГЛАСОВАНО (APPROVE)\n\n"
        "Причины:\n"
        "1. Все обязательные проверки пройдены\n"
    )
    assert canonical.reasons == ["Все обязательные проверки пройдены"]
    assert canonical.checks[0].code == "APPLICATION_NUMBER"
    assert canonical.extracted_values == {"application_number": "42"}
    assert canonical.overall_confidence == 0.99
    assert state.status is DeliveryStatus.SENT and state.attempts == 1
    payload = json.loads((settings.mock_outbox_dir / f"{record.job_id}.json").read_text("utf-8"))
    assert payload["job_id"] == record.job_id
    assert payload["status"] == "COMPLETED"
    assert payload["decision"] == "APPROVE"
    assert payload["comment_1c"] == canonical.comment_1c
    assert adapter.send_calls == 1


def test_onec_comment_uses_application_number_from_reference_json(tmp_path):
    service = make_persistence(tmp_path); storage = JobStorage(tmp_path / "jobs")
    record = make_completed_record(tmp_path, service, storage)
    reference_path = Path(record.document_path).with_name("reference.json")
    reference_path.write_text(
        json.dumps({**REFERENCE, "application_number": "REF-007"}),
        encoding="utf-8",
    )
    settings = OneCIntegrationSettings(enabled=True, mock_outbox_dir=tmp_path / "outbox")
    integration = OneCIntegrationService(
        settings=settings,
        adapter=MockOneCAdapter(settings),
        persistence=service,
    )

    state = integration.deliver(record)

    assert state is not None
    assert state.payload["decision"] == "APPROVE"
    assert state.payload["comment_1c"].startswith(
        f"ПРОВЕРКА СПЕЦИФИКАЦИИ №REF-007\nID проверки: {record.job_id}\n\n"
    )
    assert "1. Номер приложения: совпадает" in state.payload["comment_1c"]
    assert "ИТОГОВОЕ РЕШЕНИЕ: СОГЛАСОВАНО (APPROVE)" in state.payload["comment_1c"]
    assert "=" not in state.payload["comment_1c"]


def test_onec_comment_localizes_return_and_humanizes_reasons(tmp_path):
    result = check_result()
    result["decision"] = "RETURN"
    result["checks"][0].update(
        status="MISMATCH",
        actual="41",
        message="Значение отличается",
    )
    result["checks"].append(
        {
            **result["checks"][0],
            "code": "BUYER_SIGNATURE",
            "title": "Подпись покупателя",
            "status": "UNCERTAIN",
            "expected": True,
            "actual": "UNCERTAIN",
            "message": "Поле не извлечено или уверенность ниже порога",
        }
    )
    result["reasons"] = [
        "Номер приложения: Значение отличается",
        "Подпись покупателя: Поле не извлечено или уверенность ниже порога",
    ]
    service = make_persistence(tmp_path)
    storage = JobStorage(tmp_path / "jobs")
    record = make_completed_record(tmp_path, service, storage, result)
    settings = OneCIntegrationSettings(enabled=True, mock_outbox_dir=tmp_path / "outbox")
    integration = OneCIntegrationService(
        settings=settings,
        adapter=MockOneCAdapter(settings),
        persistence=service,
    )

    canonical = integration.canonical_result(record)

    assert canonical.decision.value == "RETURN"
    assert "ИТОГОВОЕ РЕШЕНИЕ: ВОЗВРАТ (RETURN)" in canonical.comment_1c
    assert "1. Номер приложения: значение отличается" in canonical.comment_1c
    assert (
        "2. Подпись покупателя: не удалось определить с достаточной уверенностью"
        in canonical.comment_1c
    )
    assert "уверенность ниже порога" not in canonical.comment_1c


def test_onec_comment_localizes_manual_review(tmp_path):
    result = check_result()
    result["decision"] = "MANUAL_REVIEW"
    result["checks"][0].update(
        status="UNCERTAIN",
        actual=None,
        message="Поле не извлечено или уверенность ниже порога",
    )
    result["reasons"] = [
        "Номер приложения: Поле не извлечено или уверенность ниже порога"
    ]
    service = make_persistence(tmp_path)
    storage = JobStorage(tmp_path / "jobs")
    record = make_completed_record(tmp_path, service, storage, result)
    settings = OneCIntegrationSettings(enabled=True, mock_outbox_dir=tmp_path / "outbox")
    integration = OneCIntegrationService(
        settings=settings,
        adapter=MockOneCAdapter(settings),
        persistence=service,
    )

    canonical = integration.canonical_result(record)

    assert canonical.decision.value == "MANUAL_REVIEW"
    assert (
        "ИТОГОВОЕ РЕШЕНИЕ: ТРЕБУЕТ РУЧНОЙ ПРОВЕРКИ (MANUAL_REVIEW)"
        in canonical.comment_1c
    )


def test_delivery_error_stays_pending_and_does_not_change_job(tmp_path):
    service = make_persistence(tmp_path); storage = JobStorage(tmp_path / "jobs")
    record = make_completed_record(tmp_path, service, storage)
    settings = OneCIntegrationSettings(enabled=True, mock_outbox_dir=tmp_path / "outbox")
    adapter = CountingMockAdapter(settings, fail=True)
    integration = OneCIntegrationService(settings=settings, adapter=adapter, persistence=service)

    state = integration.deliver(record)
    persisted = service.get(record.job_id)

    assert state.status is DeliveryStatus.PENDING
    assert state.last_error["exception_type"] == "ConnectionError"
    assert persisted.status is JobStatus.COMPLETED
    assert persisted.decision == "APPROVE"
    assert persisted.result == check_result()


def test_worker_retry_sends_without_second_checker_call(tmp_path):
    service = make_persistence(tmp_path); storage = JobStorage(tmp_path / "jobs")
    queue = Queue(); checker = Checker()
    settings = OneCIntegrationSettings(enabled=True, max_delivery_attempts=3, mock_outbox_dir=tmp_path / "outbox")
    adapter = CountingMockAdapter(settings, fail=True)
    integration = OneCIntegrationService(settings=settings, adapter=adapter, persistence=service)
    worker = SpecificationWorker(
        queue=queue, job_service=service, checker=checker, storage=storage,
        integration_service=integration, worker_id="test-worker",
    )
    job_id = str(uuid4()); folder = storage.job_dir(job_id); folder.mkdir(parents=True)
    document = folder / "document.pdf"; document.write_bytes(b"%PDF-1.4")
    (folder / "reference.json").write_text(json.dumps(REFERENCE), "utf-8")
    now = datetime.now(timezone.utc)
    service.create_processing(
        job_id=job_id, external_id="1C-retry", created_at=now, started_at=now,
        document_path=str(document), document_sha256=hashlib.sha256(document.read_bytes()).hexdigest(),
    )
    queue.enqueue(job_id)

    assert worker.process_one(timeout_seconds=1) == "completed"
    assert checker.calls == 1
    assert service.get(job_id).status is JobStatus.COMPLETED
    assert service.get(job_id).onec_delivery_status is DeliveryStatus.PENDING

    adapter.fail = False
    queue.enqueue(job_id)
    assert worker.process_one(timeout_seconds=1) == "already_final"
    assert checker.calls == 1
    assert adapter.send_calls == 2
    assert service.get(job_id).onec_delivery_status is DeliveryStatus.SENT


def test_disabled_integration_does_not_create_delivery_state(tmp_path):
    service = make_persistence(tmp_path); storage = JobStorage(tmp_path / "jobs")
    record = make_completed_record(tmp_path, service, storage)
    settings = OneCIntegrationSettings(enabled=False, mock_outbox_dir=tmp_path / "outbox")
    integration = OneCIntegrationService(settings=settings, adapter=MockOneCAdapter(settings), persistence=service)
    assert integration.deliver(record) is None
    assert service.get(record.job_id).onec_delivery_status is None


def test_onec_settings_are_overridable_by_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ONEC_INTEGRATION_ENABLED", "true")
    monkeypatch.setenv("ONEC_ADAPTER", "mock")
    monkeypatch.setenv("ONEC_API_URL", "https://onec.invalid/callback")
    monkeypatch.setenv("ONEC_AUTH_TOKEN", "secret-value")
    monkeypatch.setenv("ONEC_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("ONEC_MAX_DELIVERY_ATTEMPTS", "4")
    monkeypatch.setenv("ONEC_MOCK_OUTBOX_DIR", str(tmp_path / "env-outbox"))

    settings = OneCIntegrationSettings.from_environment(
        {"onec_integration": {"enabled": False, "timeout_seconds": 99}}
    )

    assert settings.enabled is True
    assert settings.api_url == "https://onec.invalid/callback"
    assert settings.auth_token.get_secret_value() == "secret-value"
    assert settings.timeout_seconds == 17
    assert settings.max_delivery_attempts == 4
    assert settings.mock_outbox_dir == tmp_path / "env-outbox"


def test_onec_basic_auth_settings_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("ONEC_AUTH_USERNAME", "test-user")
    monkeypatch.setenv("ONEC_AUTH_PASSWORD", "test-password")

    settings = OneCIntegrationSettings.from_environment()

    assert settings.auth_username == "test-user"
    assert settings.auth_password is not None
    assert settings.auth_password.get_secret_value() == "test-password"
