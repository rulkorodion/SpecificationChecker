# Upgrade to two GPU workers

This update replaces only the application images and deployment configuration.
PostgreSQL, Redis, their images, named volumes, the MonkeyOCR model directory,
job files and the existing `.env.production` are not included or deleted.

The update contains:

- `specification-checker-api:0.1.2-stage6.20260806`;
- `specification-checker-worker:0.1.4-stage6.20260806`;
- a production Compose file with service `worker` (`WORKER_ID=worker-gpu-0`) on
  GPU 0 and service `worker-gpu-1` on GPU 1;
- Redis queue code with independent worker registrations and in-flight lists;
- the single verified MonkeyOCR path (SDPA, KV cache and the validated pixel
  limit). The discarded experimental inference path is absent.

Before deployment, preserve the current `.env.production`. Remove obsolete
MonkeyOCR profile/attention/cache variables if they are present and set:

```dotenv
MONKEYOCR_DTYPE=auto
MONKEYOCR_MAX_PIXELS=1003520
MONKEYOCR_MAX_NEW_TOKENS=
MONKEYOCR_PDF_RENDER_DPI=300
WORKER_LOCK_TTL_SECONDS=120
EXPECTED_WORKER_COUNT=2
```

Loading the image archive does not stop the running release. Applying the new
Compose file recreates API and the first worker, starts the second worker and
preserves all named volumes. Do not use `docker compose down -v`.

After deployment, `/health` must contain `"worker_count":2` and the IDs
`worker-gpu-0` and `worker-gpu-1`.
