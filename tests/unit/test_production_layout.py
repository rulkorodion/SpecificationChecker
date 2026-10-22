from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import yaml

from app.api.storage import APISettings
from app.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_persistence_import_does_not_load_ocr_dependencies():
    code = (
        "import sys; import app.persistence.database; "
        "assert 'app.service' not in sys.modules; "
        "assert 'numpy' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
    )


def test_testdata_has_explicit_structure_and_no_legacy_sample_paths():
    root = ROOT / "testdata" / "specifications"
    assert {item.name for item in root.iterdir() if item.is_dir()} == {
        "documents", "reference", "ground_truth", "extracted_mock"
    }
    assert not (ROOT / "samples").exists()
    for path in [ROOT / "app", ROOT / "tests"]:
        for source in path.rglob("*.py"):
            if source.resolve() == Path(__file__).resolve():
                continue
            assert "samples/" not in source.read_text("utf-8")


def test_production_and_development_configs_are_isolated(monkeypatch):
    monkeypatch.delenv("API_DOCS_ENABLED", raising=False)
    production = load_config(ROOT / "config" / "production.yaml")
    development = load_config(ROOT / "config" / "development.yaml")
    assert APISettings.from_config(production).docs_enabled is False
    assert APISettings.from_config(development).docs_enabled is True
    assert production["ocr"]["monkeyocr_v2"]["cache_dir"] == "/models/huggingface"
    assert production["onec_integration"]["enabled"] is False
    assert development["onec_integration"]["adapter"] == "mock"
    assert development["onec_integration"]["enabled"] is True


def test_compose_contains_required_services_volumes_and_two_gpu_workers():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text("utf-8"))
    assert set(compose["services"]) == {
        "api", "worker", "worker-gpu-1", "postgres", "redis", "migrate"
    }
    assert {"jobs_data", "postgres_data", "redis_data"} <= set(compose["volumes"])
    api_volumes = compose["services"]["api"]["volumes"]
    worker_volumes = compose["services"]["worker"]["volumes"]
    assert "jobs_data:/runtime/jobs" in api_volumes
    assert "jobs_data:/runtime/jobs" in worker_volumes
    assert "postgres_data:/var/lib/postgresql" in compose["services"]["postgres"]["volumes"]
    for index in (0, 1):
        worker = compose["services"]["worker" if index == 0 else "worker-gpu-1"]
        assert worker["deploy"]["replicas"] == 1
        gpu = worker["deploy"]["resources"]["reservations"]["devices"][0]
        assert gpu == {
            "driver": "nvidia",
            "device_ids": [str(index)],
            "capabilities": ["gpu"],
        }
        assert worker["environment"]["WORKER_ID"] == f"worker-gpu-{index}"
        assert worker["environment"]["NVIDIA_VISIBLE_DEVICES"] == str(index)
    model_mount = next(item for item in worker_volumes if isinstance(item, dict))
    assert model_mount["target"] == "/models/huggingface"
    assert model_mount["read_only"] is True
    for service in ("api", "worker", "worker-gpu-1", "postgres", "redis"):
        assert compose["services"][service]["restart"] == "unless-stopped"
        assert "healthcheck" in compose["services"][service]


def test_docker_context_excludes_tests_data_runtime_models_and_secrets():
    patterns = set((ROOT / ".dockerignore").read_text("utf-8").splitlines())
    assert {"tests", "testdata", "samples", "runtime", "models", ".git", ".env"} <= patterns
    assert (ROOT / "docker" / "Dockerfile.api").is_file()
    assert (ROOT / "docker" / "Dockerfile.worker").is_file()


class MigrationOperations:
    def __init__(self): self.added = []; self.constraints = []; self.dropped = []
    def add_column(self, table, column): self.added.append((table, column.name, column.type))
    def f(self, name): return name
    def create_check_constraint(self, name, table, condition): self.constraints.append((name, table, condition))
    def drop_constraint(self, name, table, type_=None): self.dropped.append(("constraint", name))
    def drop_column(self, table, column): self.dropped.append(("column", column))


def test_onec_delivery_migration_is_separate_and_reversible():
    path = ROOT / "alembic" / "versions" / "20260804_0002_add_onec_delivery_state.py"
    spec = importlib.util.spec_from_file_location("stage6_migration", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    operations = MigrationOperations(); module.op = operations
    assert module.revision == "20260804_0002"
    assert module.down_revision == "20260804_0001"
    module.upgrade()
    assert {name for _, name, _ in operations.added} == {
        "onec_delivery_status", "onec_delivery_attempts", "onec_payload",
        "onec_receipt", "onec_last_error", "onec_delivery_updated_at",
    }
    assert operations.constraints[0][0] == "ck_specification_checks_onec_delivery_status"
    module.downgrade()
    assert ("column", "onec_delivery_status") in operations.dropped
