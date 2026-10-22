# Specification Checker — offline GPU deployment

Package version: `0.1.4-stage6.20260806`  
Target platform: `linux/amd64`

The GPU worker uses `python:3.12-slim-bookworm` plus the official PyTorch 2.7.1
`cu128` binary dependencies with Blackwell `sm_120` support. It intentionally does not add a second
`nvidia/cuda` runtime layer; the server still requires an NVIDIA driver and
NVIDIA Container Toolkit.

The worker contains only the verified MonkeyOCR inference path: PyTorch SDPA,
deterministic generation, KV cache, and the validated per-input pixel limit.
Both worker containers use the same image; GPU 0 and GPU 1 are selected by
Compose and each process loads one model instance.

The package contains prebuilt API, GPU-worker, PostgreSQL and Redis images. It does not contain application source, test data other than one smoke PDF/reference, secrets, runtime data, database volumes, or MonkeyOCR weights.

## Server prerequisites

- Linux x86_64;
- Docker Engine 24 or newer;
- Docker Compose plugin 2 or newer;
- NVIDIA driver compatible with CUDA 12.8;
- NVIDIA Container Toolkit configured for Docker;
- `curl`, `sha256sum`, `nvidia-smi`;
- at least 25 GiB free disk space for loading and running the images.

The server does not need Python, PyTorch, CUDA SDK, PostgreSQL or Redis packages installed on the host. They are inside the offline Docker images. NVIDIA driver and Container Toolkit cannot be packaged inside an application image and must exist on the server.

## Copy MonkeyOCR separately

Create an external model directory:

```bash
sudo mkdir -p /opt/models
sudo chown <server-user>:<server-user> /opt/models
```

Copy the local Hugging Face cache root to:

```text
/opt/models/MonkeyOCRv2-B-Parsing/
  hub/
    models--zenosai--MonkeyOCRv2-B-Parsing/
      snapshots/
  modules/        # optional; runtime modules use a separate writable cache
```

The model is mounted into the worker as `/models/huggingface:ro`. Production sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `local_files_only: true`; the worker must not download model weights.

## Transfer

SCP/SFTP directory variant:

```bash
scp -r deployment server-user@server:/opt/specification-checker/
```

Single archive variant:

```bash
scp deployment-package-0.1.4-stage6.20260806-linux-amd64.tar.gz* server-user@server:/opt/specification-checker/
ssh server-user@server
cd /opt/specification-checker
sha256sum -c deployment-package-0.1.4-stage6.20260806-linux-amd64.tar.gz.sha256
tar -xzf deployment-package-0.1.4-stage6.20260806-linux-amd64.tar.gz
cd deployment
```

## Configure and deploy

```bash
cp .env.production.example .env.production
chmod 600 .env.production
chmod +x scripts/*.sh
```

Edit `.env.production`:

- replace `POSTGRES_PASSWORD`;
- put the URL-encoded password in `DATABASE_URL`;
- verify `MONKEYOCR_MODEL_PATH`;
- choose the API bind address;
- do not enable 1C until the real adapter is implemented.

The production template uses the same validated inference settings as the
previous local runs. The following safe limits may be changed in
`.env.production` without rebuilding the worker image:

```dotenv
MONKEYOCR_DTYPE=auto
MONKEYOCR_MAX_PIXELS=1003520
MONKEYOCR_MAX_NEW_TOKENS=
MONKEYOCR_PDF_RENDER_DPI=300
EXPECTED_WORKER_COUNT=2
```

Startup logs show the effective dtype/backend and every inference call logs its
application/processor image size, visual-token count and generation limit.

Then run, in this order:

```bash
scripts/check-host.sh
scripts/load-images.sh
scripts/check-host.sh
scripts/deploy.sh
scripts/status.sh
scripts/smoke-test.sh
```

The first host check validates Linux, architecture, Docker, Compose, the host GPU, model layout and disk. Container GPU validation is deferred until the worker image is loaded. `deploy.sh` repeats it as a mandatory check and refuses to start if NVIDIA Container Toolkit cannot expose the GPU.

No script executes `docker compose build`, `docker compose pull`, or `docker compose down -v`.

## Operations

```bash
scripts/status.sh
scripts/logs.sh worker
scripts/logs.sh api
scripts/stop.sh
```

`stop.sh` keeps PostgreSQL, Redis, jobs and runtime volumes. Re-running `deploy.sh` applies Alembic migrations and starts the existing volumes.

Swagger is disabled in production. Health is available at `/health`. The smoke test submits only `smoke/document.pdf` and `smoke/reference.json`; no ground truth is sent through the API.

## Offline images

Expected tags:

```text
specification-checker-api:0.1.1-stage6.20260804
specification-checker-worker:0.1.3-stage6.20260805
postgres:18.4-alpine
redis:7.4.10-alpine
```

`scripts/load-images.sh` validates the image archive SHA-256 before `docker load` and checks all four tags afterwards. It never pulls from a registry.

The API accepts both the legacy `multipart/form-data` upload and the 1C-oriented
`application/json` request. In JSON mode, `documents` is an ordered array of
objects containing `filename`, optional `content_type`, and `content_base64`;
`reference` is a regular nested JSON object. Multiple documents are normalized
to one PDF before the unchanged worker pipeline receives the job. The default
maximum is 10 documents per request and can be changed with
`API_MAX_DOCUMENTS_PER_JOB` in `.env.production`.
