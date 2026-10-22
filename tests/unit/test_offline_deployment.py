from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = ROOT / "deployment"


def test_offline_deployment_structure_and_smoke_fixture() -> None:
    expected = {
        "compose.production.yml",
        ".env.production.example",
        "VERSION",
        "API_IMAGE_VERSION",
        "WORKER_IMAGE_VERSION",
        "manifest.json",
        "checksums.sha256",
        "README_DEPLOY.md",
        "docker-images",
        "scripts",
        "smoke",
    }
    assert expected <= {item.name for item in DEPLOYMENT.iterdir()}
    assert (DEPLOYMENT / "smoke/document.pdf").read_bytes().startswith(b"%PDF-")
    assert json.loads((DEPLOYMENT / "smoke/reference.json").read_text("utf-8"))
    assert not list(DEPLOYMENT.rglob("*ground_truth*"))


def test_offline_compose_uses_fixed_images_and_no_build_context() -> None:
    compose = yaml.safe_load((DEPLOYMENT / "compose.production.yml").read_text("utf-8"))
    services = compose["services"]
    assert set(services) == {
        "postgres", "redis", "migrate", "api", "worker", "worker-gpu-1"
    }
    assert all("build" not in service for service in services.values())
    assert services["api"]["image"] == "specification-checker-api:0.1.2-stage6.20260806"
    assert services["postgres"]["image"] == "postgres:18.4-alpine"
    assert services["redis"]["image"] == "redis:7.4.10-alpine"
    assert services["migrate"]["image"] == services["api"]["image"]
    for index in (0, 1):
        worker = services["worker" if index == 0 else "worker-gpu-1"]
        assert worker["image"] == "specification-checker-worker:0.1.4-stage6.20260806"
        assert worker["deploy"]["replicas"] == 1
        device = worker["deploy"]["resources"]["reservations"]["devices"][0]
        assert device["device_ids"] == [str(index)]
        assert device["capabilities"] == ["gpu"]
        assert worker["environment"]["WORKER_ID"] == f"worker-gpu-{index}"
    assert "deploy" not in services["api"]


def test_model_is_external_read_only_and_production_is_offline() -> None:
    compose = yaml.safe_load((DEPLOYMENT / "compose.production.yml").read_text("utf-8"))
    for worker_name in ("worker", "worker-gpu-1"):
        worker = compose["services"][worker_name]
        mounts = [item for item in worker["volumes"] if isinstance(item, dict)]
        assert mounts == [
            {
                "type": "bind",
                "source": "${MONKEYOCR_MODEL_PATH:?MONKEYOCR_MODEL_PATH is required}",
                "target": "/models/huggingface",
                "read_only": True,
            }
        ]
        assert worker["environment"]["HF_HUB_OFFLINE"] == "1"
        assert worker["environment"]["TRANSFORMERS_OFFLINE"] == "1"
        assert worker["environment"]["ONEC_INTEGRATION_ENABLED"] == (
            "${ONEC_INTEGRATION_ENABLED:-false}"
        )
        assert worker["environment"]["ONEC_ADAPTER"] == "${ONEC_ADAPTER:-mock}"
        assert worker["environment"]["ONEC_TIMEOUT_SECONDS"] == (
            "${ONEC_TIMEOUT_SECONDS:-15}"
        )
    manifest = json.loads((DEPLOYMENT / "manifest.json").read_text("utf-8"))
    assert manifest["model"]["included"] is False
    assert manifest["model"]["mount_mode"] == "read-only"


def test_server_scripts_are_offline_and_preserve_volumes() -> None:
    scripts = DEPLOYMENT / "scripts"
    expected = {
        "common.sh",
        "check-host.sh",
        "load-images.sh",
        "deploy.sh",
        "status.sh",
        "logs.sh",
        "stop.sh",
        "smoke-test.sh",
    }
    assert expected == {path.name for path in scripts.glob("*.sh")}
    deploy = (scripts / "deploy.sh").read_text("utf-8")
    assert "--no-build" in deploy
    assert "--pull never" in deploy
    assert "compose build" not in deploy
    assert "compose pull" not in deploy
    stop = (scripts / "stop.sh").read_text("utf-8")
    assert "compose down" in stop
    assert "down -v" not in stop
    smoke = (scripts / "smoke-test.sh").read_text("utf-8")
    assert "ground_truth" not in smoke
    assert '"$base_url/api/v1/checks"' in smoke
    assert "HTTP 202" in smoke


def test_deployment_has_no_windows_or_source_paths() -> None:
    for path in DEPLOYMENT.rglob("*"):
        if not path.is_file() or path.suffix in {".pdf", ".tar", ".gz"}:
            continue
        text = path.read_text("utf-8")
        assert "C:\\Users\\Rodion" not in text
        assert "testdata" + "/" not in text
        assert "samples" + "/" not in text
        assert "ground_truth" not in text
        assert "build:" not in text if path.name == "compose.production.yml" else True


def test_env_template_and_builder_cover_required_contract() -> None:
    env_text = (DEPLOYMENT / ".env.production.example").read_text("utf-8")
    for name in (
        "TARGET_PLATFORM",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "DATABASE_URL",
        "REDIS_URL",
        "REDIS_QUEUE_NAME",
        "JOB_STORAGE_ROOT",
        "MONKEYOCR_MODEL_PATH",
        "MONKEYOCR_DTYPE",
        "MONKEYOCR_MAX_PIXELS",
        "MONKEYOCR_MAX_NEW_TOKENS",
        "MONKEYOCR_PDF_RENDER_DPI",
        "API_BIND_ADDRESS",
        "API_PORT",
        "JOB_RETENTION_DAYS",
        "ONEC_INTEGRATION_ENABLED",
        "ONEC_ADAPTER",
        "WORKER_JOB_TIMEOUT_SECONDS",
        "EXPECTED_WORKER_COUNT",
        "JOB_CLEANUP_INTERVAL_SECONDS",
    ):
        assert f"{name}=" in env_text
    builder = (ROOT / "tools/build-deployment-package.ps1").read_text("utf-8")
    for operation in ("buildx build", "pull --platform", "save --output", "load --input"):
        assert operation in builder
    dockerignore = (ROOT / ".dockerignore").read_text("utf-8")
    assert "deployment\n" in dockerignore
    assert "testdata\n" in dockerignore


def test_worker_uses_pytorch_cuda_runtime_without_duplicate_cuda_base() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.worker").read_text("utf-8")
    requirements = (ROOT / "requirements-monkeyocr.txt").read_text("utf-8")
    assert dockerfile.startswith("FROM python:3.12-slim-bookworm\n")
    assert "FROM nvidia/cuda" not in dockerfile
    assert "download.pytorch.org/whl/cu128" in requirements
    assert "torch==2.7.1" in requirements
    assert "torchvision==0.22.1" in requirements
    assert "flash-attn" not in dockerfile


def test_incremental_server_update_rebuilds_only_application_images() -> None:
    builder = (ROOT / "tools" / "build-server-update.ps1").read_text("utf-8")
    assert "Dockerfile.api" in builder
    assert "Dockerfile.worker" in builder
    assert "Invoke-Docker save --output" in builder
    assert "$ApiImage $WorkerImage" in builder
    assert "postgres:18.4-alpine" not in builder
    assert "redis:7.4.10-alpine" not in builder
    update_readme = (DEPLOYMENT / "README_UPDATE.md").read_text("utf-8")
    assert "PostgreSQL" in update_readme and "not included or deleted" in update_readme
    assert "worker-gpu-0" in update_readme and "worker-gpu-1" in update_readme
