# Проверка спецификаций — автономное развёртывание с GPU

- Версия пакета: `0.1.4-stage6.20260806`
- Целевая платформа: `linux/amd64`

GPU-обработчик использует `python:3.12-slim-bookworm` и официальные бинарные
зависимости PyTorch 2.7.1 `cu128` с поддержкой Blackwell `sm_120`. Отдельный
слой среды выполнения `nvidia/cuda` намеренно не добавляется; на сервере всё
равно должны быть установлены драйвер NVIDIA и NVIDIA Container Toolkit.

Обработчик содержит только проверенный способ выполнения MonkeyOCR: PyTorch
SDPA, детерминированную генерацию, KV-кэш и проверенное ограничение пикселей для
одного входа. Оба контейнера обработчиков используют один образ; GPU 0 и GPU 1
выбираются через Compose, а каждый процесс загружает один экземпляр модели.

Пакет содержит готовые образы API, GPU-обработчика, PostgreSQL и Redis. В него
не входят исходный код приложения, тестовые данные кроме одного проверочного
PDF и эталонного JSON, секреты, рабочие данные, тома базы данных и веса
MonkeyOCR.

## Требования к серверу

- Linux x86_64;
- Docker Engine 24 или новее;
- плагин Docker Compose 2 или новее;
- драйвер NVIDIA, совместимый с CUDA 12.8;
- NVIDIA Container Toolkit, настроенный для Docker;
- `curl`, `sha256sum`, `nvidia-smi`;
- не менее 25 ГиБ свободного места для загрузки и запуска образов.

На сервер не нужно отдельно устанавливать Python, PyTorch, CUDA SDK, PostgreSQL
или Redis: они находятся внутри автономных Docker-образов. Драйвер NVIDIA и
Container Toolkit нельзя упаковать в образ приложения, поэтому они должны быть
установлены на самом сервере.

## Отдельное копирование MonkeyOCR

Создайте внешний каталог модели:

```bash
sudo mkdir -p /opt/models
sudo chown <server-user>:<server-user> /opt/models
```

Скопируйте корень локального кэша Hugging Face в следующую структуру:

```text
/opt/models/MonkeyOCRv2-B-Parsing/
  hub/
    models--zenosai--MonkeyOCRv2-B-Parsing/
      snapshots/
  modules/        # необязательно; рабочие модули используют отдельный доступный для записи кэш
```

Модель подключается к обработчику как `/models/huggingface:ro`. В production-
режиме заданы `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` и
`local_files_only: true`; обработчик не должен загружать веса модели из сети.

## Передача файлов

Вариант с каталогом через SCP/SFTP:

```bash
scp -r deployment server-user@server:/opt/specification-checker/
```

Вариант с единым архивом:

```bash
scp deployment-package-0.1.4-stage6.20260806-linux-amd64.tar.gz* server-user@server:/opt/specification-checker/
ssh server-user@server
cd /opt/specification-checker
sha256sum -c deployment-package-0.1.4-stage6.20260806-linux-amd64.tar.gz.sha256
tar -xzf deployment-package-0.1.4-stage6.20260806-linux-amd64.tar.gz
cd deployment
```

## Настройка и развёртывание

```bash
cp .env.production.example .env.production
chmod 600 .env.production
chmod +x scripts/*.sh
```

Отредактируйте `.env.production`:

- замените `POSTGRES_PASSWORD`;
- укажите в `DATABASE_URL` пароль с URL-кодированием;
- проверьте `MONKEYOCR_MODEL_PATH`;
- выберите адрес привязки API;
- не включайте интеграцию с 1С до реализации реального адаптера.

Production-шаблон использует те же проверенные настройки выполнения модели,
что и предыдущие локальные запуски. Следующие безопасные ограничения можно
изменять в `.env.production` без пересборки образа обработчика:

```dotenv
MONKEYOCR_DTYPE=auto
MONKEYOCR_MAX_PIXELS=1003520
MONKEYOCR_MAX_NEW_TOKENS=
MONKEYOCR_PDF_RENDER_DPI=300
EXPECTED_WORKER_COUNT=2
```

При запуске в журнал выводятся фактический тип данных и механизм выполнения, а
для каждого вызова модели — размеры изображения приложения и процессора, число
визуальных токенов и лимит генерации.

Затем последовательно выполните:

```bash
scripts/check-host.sh
scripts/load-images.sh
scripts/check-host.sh
scripts/deploy.sh
scripts/status.sh
scripts/smoke-test.sh
```

Первая проверка сервера проверяет Linux, архитектуру, Docker, Compose, GPU
сервера, структуру каталога модели и свободное место. Проверка GPU внутри
контейнера откладывается до загрузки образа обработчика. `deploy.sh` повторяет
её как обязательную проверку и отказывается запускать систему, если NVIDIA
Container Toolkit не может предоставить контейнеру GPU.

Ни один сценарий не выполняет `docker compose build`, `docker compose pull`
или `docker compose down -v`.

## Управление системой

```bash
scripts/status.sh
scripts/logs.sh worker
scripts/logs.sh api
scripts/stop.sh
```

`stop.sh` сохраняет PostgreSQL, Redis, задания и рабочие тома. Повторный запуск
`deploy.sh` применяет миграции Alembic и запускает существующие тома.

Swagger отключён в production-режиме. Проверка состояния доступна по
`/health`. Проверочный тест отправляет только `smoke/document.pdf` и
`smoke/reference.json`; разметка правильных ответов через API не передаётся.

## Автономные образы

Ожидаемые теги:

```text
specification-checker-api:0.1.1-stage6.20260804
specification-checker-worker:0.1.3-stage6.20260805
postgres:18.4-alpine
redis:7.4.10-alpine
```

`scripts/load-images.sh` проверяет SHA-256 архива образов перед `docker load`,
а после загрузки проверяет наличие всех четырёх тегов. Сценарий никогда не
загружает образы из реестра.

API принимает как прежнюю загрузку `multipart/form-data`, так и ориентированный
на 1С запрос `application/json`. В режиме JSON поле `documents` является
упорядоченным массивом объектов с полями `filename`, необязательным
`content_type` и `content_base64`; `reference` остаётся обычным вложенным
JSON-объектом. Несколько документов преобразуются в один PDF до передачи
задания неизменённому конвейеру обработчика. По умолчанию один запрос может
содержать не более 10 документов; лимит меняется через
`API_MAX_DOCUMENTS_PER_JOB` в `.env.production`.
