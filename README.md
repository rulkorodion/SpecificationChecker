# Проверка спецификаций

Локальная система проверки спецификаций к договорам поставки, работающая по
принципу безопасного отказа. Она
распознаёт документ с помощью MonkeyOCRv2-B-Parsing, извлекает поля, сравнивает
их с эталонным JSON, отдельно проверяет подпись и печать покупателя средствами
OpenCV и формирует `report.json`, краткий консольный отчёт и решение:

- `APPROVE` — обязательные проверки пройдены;
- `RETURN` — найдено явное несоответствие или нарушение срока;
- `MANUAL_REVIEW` — доказательств недостаточно либо формат не поддерживается.

Эталонные данные и необязательная контрольная разметка не передаются в OCR или
модули извлечения. Контрольная разметка предназначена только для разработки и
оценки качества.

## Поддерживаемый формат

- PDF, JPG, JPEG, PNG, TIF и TIFF;
- одна или две страницы;
- документ с большим количеством страниц сразу направляется на
  `MANUAL_REVIEW`, OCR для него не запускается;
- документы, подписанные через оператора ЭДО, обрабатываются отдельным правилом;
- подпись и печать обычного документа проверяются независимо.

## Архитектура

```text
загрузчик документа -> быстрый CPU-модуль выбора страниц для PDF > 2 страниц
  -> selected_document.pdf (нужное приложение в логическом порядке страниц)
  -> рендеринг -> оценка качества
  -> определение структуры страницы с помощью MonkeyOCRv2
  -> распознавание фрагментов в исходном разрешении
  -> компактный OCR-кэш
  -> извлечение и нормализация полей
  -> определение подписи и печати средствами OpenCV
  -> механизм проверки правил
  -> report.json + console_report.txt
```

OCR реализован через единственный механизм `monkeyocr-v2`. Модель использует
официальную версию `zenosai/MonkeyOCRv2-B-Parsing`, закреплённую на ревизии
`de7a993bd0f39a97b122dac767e82ae04935bce4`.

### Выполнение MonkeyOCR

В приложении оставлен один проверенный способ выполнения модели: нативный
Transformers, SDPA, не более 1 003 520 пикселей на входной макет или фрагмент,
детерминированная генерация и включённый KV-кэш. Отклонённый эксперимент с
полной страницей и Flash Attention отсутствует в коде и конфигурации.

### Выбор страниц в PDF-пакетах

Одно- и двухстраничные документы проходят прежний конвейер без дополнительного
OCR и без преобразования файла. Для PDF длиннее двух страниц обработчик берёт
`application_number` из эталонных данных и запускает быстрый Tesseract только на
нижних колонтитулах страниц; верхний фрагмент используется как резервный. Полный
MonkeyOCR на этом этапе не запускается.

Модуль выбора страниц должен однозначно найти все отметки вида `Приложение № …, стр. X из
N`, проверяет отсутствие повторов и восстанавливает порядок `1..N`. Затем он
собирает `selected_document.pdf`; существующие модули MonkeyOCR, извлечения
данных и проверки правил получают этот файл без изменений. В каталоге результата сохраняется
`page_selection.json` с исходным числом страниц, выбранными физическими
страницами, логическим порядком и диагностикой. Если номер не найден либо выбор
неоднозначен, задача завершается ошибкой `page_selection_failed` — модуль ничего
не угадывает.

Образ обработчика содержит системные пакеты `tesseract-ocr` и
`tesseract-ocr-rus`. Веса MonkeyOCR по-прежнему подключаются отдельно и в образ
не входят.

Без пересборки образа можно менять только безопасные ограничения ресурсов и
качества:

```dotenv
MONKEYOCR_DTYPE=auto|bfloat16|float16
MONKEYOCR_MAX_PIXELS=1003520
MONKEYOCR_MAX_NEW_TOKENS=
MONKEYOCR_PDF_RENDER_DPI=300
```

`MONKEYOCR_MAX_PIXELS` должен быть положительным целым числом. Фактический тип
данных, размер входа процессора, число визуальных токенов и лимит генерации
записываются в метаданные результата.

## Структура проекта

```text
app/                         рабочий код приложения
  api/                       FastAPI
  integrations/              заменяемые адаптеры 1С
  persistence/               хранилище и сервис PostgreSQL
  queueing/                  Redis-очередь
  ocr/                       адаптер MonkeyOCR
alembic/versions/            последовательные миграции БД
config/                      YAML для базовой, локальной и production-конфигурации
docker/                      отдельные Dockerfile для API и GPU-обработчика
tests/                       автоматические тесты и синтетические данные
testdata/specifications/     локальный приватный тестовый корпус (не хранится в Git)
  documents/                реальные документы
  reference/                эталонные данные
  ground_truth/             разметка для оценки качества
  extracted_mock/
runtime/                     только изменяемые рабочие файлы, не код
```

Веса MonkeyOCR не находятся в репозитории и Docker-образе. Они хранятся во
внешнем кэше Hugging Face и подключаются к обработчику в каталог
`/models/huggingface`.

Реальные спецификации, эталонные JSON и контрольная разметка также исключены из Git,
поскольку содержат внутренние данные. Тесты, которым нужен приватный корпус,
автоматически пропускаются, если соответствующих локальных файлов нет.

## Программный интерфейс

Простейшая точка входа использует один лениво созданный сервис на весь процесс:

```python
from app import check_specification

result = check_specification(
    document_path="testdata/specifications/documents/Спецификация 40.pdf",
    reference="testdata/specifications/reference/specification_40.json",
    runtime_config={
        "output_dir": "runtime/reports/specification_40",
        "reuse_ocr_cache": True,
        "debug_level": "none",
    },
)

print(result["decision"])
print(result["comment_1c"])
```

`reference` может быть путём, словарём или экземпляром `ReferenceData`.
Возвращаемый словарь полностью JSON-сериализуем и содержит:

- `decision` — `APPROVE`, `RETURN` или `MANUAL_REVIEW`;
- `extracted_values` и подробные `field_details`;
- `checks` и `reasons`;
- `comment_1c`;
- общий и покомпонентный `confidence`;
- `metadata` с хешем документа, временем, версиями, устройством и стадиями.

`confidence.overall` — минимальная уверенность среди обязательных проверок.
Это диагностический показатель: он не заменяет статусы проверок и не участвует
в выборе решения механизмом правил.

Для серверного процесса рекомендуется явно создать один долгоживущий сервис:

```python
from app import SpecificationCheckerService

checker = SpecificationCheckerService(
    {
        "device": "auto",
        "reuse_ocr_cache": True,
        "debug_level": "none",
    }
)

# Необязательно: заранее загрузить модель при старте процесса.
checker.prepare()

first = checker.check_specification("incoming/40.pdf", reference_40)
second = checker.check_specification("incoming/41.pdf", reference_41)
```

Оба вызова используют один объект MonkeyOCR и один OCR-кэш. Внутренняя
блокировка обрабатывает документы последовательно, поскольку модель хранит
изменяемое состояние текущего документа и GPU-буферы. Бизнес-правила,
пороги и алгоритмы OCR остаются внутри существующего конвейера.

## HTTP API для интеграции с 1С

FastAPI не загружает OCR и не выполняет проверку. Он сохраняет вход в
PostgreSQL и общее файловое хранилище, кладёт в Redis только `job_id` и сразу
возвращает `PROCESSING`. Отдельный последовательный обработчик создаёт один
`SpecificationCheckerService`, поэтому MonkeyOCR загружается один раз и
переиспользуется для следующих документов.

### PostgreSQL и миграции

API требует PostgreSQL и получает строку подключения только из переменной
окружения `DATABASE_URL`. Файл `.env.example` служит шаблоном; секреты из него
не читаются автоматически и не должны попадать в Git.

После установки PostgreSQL создайте отдельного владельца и базу, например через
`psql -U postgres`:

```sql
CREATE ROLE spec_checker LOGIN PASSWORD 'replace_with_strong_password';
CREATE DATABASE spec_checker OWNER spec_checker;
```

Для PowerShell задайте подключение и примените миграции:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://spec_checker:replace_with_strong_password@127.0.0.1:5432/spec_checker"
.\.venv-gpu\Scripts\python.exe -m alembic upgrade head
.\.venv-gpu\Scripts\python.exe -m alembic current
```

Для Linux:

```bash
export DATABASE_URL='postgresql+psycopg://spec_checker:replace_with_strong_password@127.0.0.1:5432/spec_checker'
python -m alembic upgrade head
python -m alembic current
```

Откат только последней миграции при необходимости:

```powershell
.\.venv-gpu\Scripts\python.exe -m alembic downgrade -1
```

Первая миграция создаёт таблицу `specification_checks` со следующими данными:

- `job_id`, `external_id`, `status`;
- `decision`, `comment_1c`, полный `result` в PostgreSQL JSONB;
- `error_code`, `error_message`, `error_details` в JSONB;
- `document_path`, `document_sha256`;
- `created_at`, `started_at`, `completed_at`;
- `processing_time_seconds`.

На `external_id`, `status` и `created_at` созданы индексы. Допустимые состояния
записи ограничены значениями `PROCESSING`, `COMPLETED` и `FAILED`.

Ревизия `20260804_0002` переносит состояние обратной отправки в 1С в
PostgreSQL: `onec_delivery_status`, число попыток, payload, receipt,
`last_error` и UTC-время. Payload, receipt и error используют JSONB. Результат
OCR не удаляется и не пересчитывается при retry.

### Redis и переменные окружения

Обязательные настройки (см. также `.env.example`):

```text
DATABASE_URL=postgresql+psycopg://spec_checker:password@127.0.0.1:5432/spec_checker
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_QUEUE_NAME=specification_checks
REDIS_BLOCK_TIMEOUT_SECONDS=5
REDIS_SOCKET_TIMEOUT_SECONDS=10
WORKER_RECONNECT_DELAY_SECONDS=2
WORKER_JOB_TIMEOUT_SECONDS=3600
WORKER_LOCK_TTL_SECONDS=120
EXPECTED_WORKER_COUNT=2
JOB_STORAGE_ROOT=runtime/jobs
```

Для локальной разработки на Linux/WSL Redis можно запустить без службы:

```bash
redis-server --port 6379 --save "" --appendonly no
```

`WORKER_JOB_TIMEOUT_SECONDS` — мягкий диагностический лимит: превышение
пишется в журнал, но активный GPU-вызов принудительно не прерывается, чтобы не
оставить состояние CUDA и модели повреждённым. `WORKER_LOCK_TTL_SECONDS`
задаёт короткий TTL регистрации: отдельный heartbeat обновляет её во время
долгого OCR, а после аварийной остановки другой обработчик возвращает его
выполнявшееся задание в общую очередь.

### Порядок запуска

Сначала PostgreSQL, миграции и Redis, затем API и обработчик в разных терминалах:

```powershell
.\.venv-gpu\Scripts\python.exe -m alembic upgrade head
.\.venv-gpu\Scripts\python.exe -m uvicorn app.api.app:app --host 0.0.0.0 --port 8000
```

```powershell
.\.venv-gpu\Scripts\python.exe -m app.worker
```

После запуска доступны:

- Swagger UI с ручной отправкой документа: `http://127.0.0.1:8000/docs`;
- OpenAPI-схема: `http://127.0.0.1:8000/openapi.json`;
- отдельная проверка PostgreSQL, Redis и регистрации обработчика: `GET /health`;
- асинхронная постановка проверки: `POST /api/v1/checks`;
- чтение сохранённого задания: `GET /api/v1/checks/{job_id}`.

`/health` возвращает HTTP 200 только когда доступны PostgreSQL и Redis и в
Redis присутствует живая регистрация обработчика. Доступный Redis без обработчика не
выдаётся за работоспособную обработку. При ошибке постановки в Redis уже
созданная запись переводится из `PROCESSING` в `FAILED`.

`POST /api/v1/checks` принимает два эквивалентных транспорта. Для 1С основной
вариант — `application/json`: бинарные документы передаются строками Base64, а
`reference` остаётся обычным вложенным JSON-объектом.

```json
{
  "external_id": "1C-000040",
  "documents": [
    {
      "filename": "specification.pdf",
      "content_type": "application/pdf",
      "content_base64": "JVBERi0xLjQK..."
    },
    {
      "filename": "attachment.jpg",
      "content_type": "image/jpeg",
      "content_base64": "/9j/4AAQSkZJRgABAQ..."
    }
  ],
  "reference": {
    "specification_id": "1C-000040",
    "application_number": "208341",
    "total_quantity": "0.123",
    "total_with_vat": "8902.34",
    "transportation_costs_included": false,
    "delivery_deadline": "2026-08-30",
    "delivery_type": "PICKUP",
    "payment_type": "PREPAYMENT",
    "deferment_days": null
  }
}
```

Порядок элементов массива `documents` сохраняется. Один документ передаётся в
обработчик без преобразования. Несколько PDF/PNG/JPEG объединяются API в один PDF
в указанном порядке, после чего существующий обработчик выполняет единое
задание. OCR, Rule Engine и формат результата при этом не меняются. По
умолчанию разрешено до 10 документов, а лимит загрузки применяется к их
суммарному декодированному размеру. Число документов можно изменить через
`API_MAX_DOCUMENTS_PER_JOB`; MIME-тип можно не передавать, если расширение
файла корректно.

Старый способ ручной отправки также сохранён. `multipart/form-data` принимает:

- `document` — файл PDF, JPG, JPEG или PNG;
- `reference` — содержимое эталонного JSON как строка формы;
- `external_id` — необязательный внешний ID документа из 1С.

Пример для Git Bash или Linux:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/checks" \
  -F 'document=@testdata/specifications/documents/Спецификация 40.pdf;type=application/pdf' \
  -F 'reference=<testdata/specifications/reference/specification_40.json' \
  -F 'external_id=1C-000040'
```

Пример для PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/checks" `
  -F "document=@testdata/specifications/documents/Спецификация 40.pdf;type=application/pdf" `
  -F "reference=<testdata/specifications/reference/specification_40.json" `
  -F "external_id=1C-000040"
```

Ответ обёрнут в задание с уникальным `job_id`:

```json
{
  "job_id": "db4bd023-70d4-4643-8546-421e65d36af5",
  "external_id": "1C-000040",
  "status": "PROCESSING",
  "created_at": "2026-08-04T10:15:31.120000Z",
  "started_at": "2026-08-04T10:15:31.120000Z",
  "completed_at": null,
  "processing_time_seconds": null,
  "result": null,
  "error": null
}
```

Получить тот же результат позднее можно по `job_id`:

```bash
curl "http://127.0.0.1:8000/api/v1/checks/db4bd023-70d4-4643-8546-421e65d36af5"
```

Каталог каждого запроса имеет вид `runtime/jobs/{job_id}/`: API сохраняет в
нём `document.pdf|jpg|jpeg|png`, `reference.json` и `job.json`; обработчик пишет
обычный результат конвейера в `result/`. API и обработчик обязаны использовать
одинаковый `JOB_STORAGE_ROOT` (локальный диск одного сервера либо общий
защищённый volume). Worker разрешает путь только внутри каталога этого UUID и
перед OCR повторно сверяет SHA-256. Бинарный документ в Redis не передаётся.

```yaml
api:
  jobs_dir: runtime/jobs
  max_upload_size_mb: 25
  max_reference_size_kb: 256
  upload_chunk_size_kb: 1024
  max_documents_per_job: 10
```

PostgreSQL остаётся источником истины. Redis использует надёжную пару списков
`<queue>:pending` и `<queue>:processing`; получение выполняется атомарным
перемещением. При старте обработчик возвращает незавершённые доставки в очередь.
Повторно доставленный `job_id` со статусом `COMPLETED` или `FAILED`
подтверждается без вызова сервиса проверки. Временный `worker_outcome.json` позволяет
после восстановления PostgreSQL сохранить уже вычисленный результат без
повторного OCR и удаляется после успешной фиксации. Входные файлы сохраняются
для аудита; срок их хранения должен задаваться эксплуатационной политикой и
очищаться существующей командой `cleanup` либо внешним регламентным заданием.

### Изолированный адаптер 1С

Интеграция с 1С отключена по умолчанию и не влияет на текущий multipart POST.
Внутренний контракт зафиксирован Pydantic-моделями в
`app/integrations/contracts.py`:

- `CanonicalDocument` — доверенный локальный путь, имя, MIME-тип, размер и SHA-256;
- `CanonicalCheckInput` — документ, `external_id` и существующий `ReferenceData`;
- `CanonicalCheckResult` — `job_id`, `external_id`, финальный status, решение,
  `comment_1c`, причины, извлечённые значения, общая уверенность и компактные
  результаты отдельных проверок;
- `DeliveryReceipt` — подтверждение конкретного адаптера.

`OneCAdapter` отделяет преобразование входа, получение документа, построение
сетевого ответа и его отправку. `MockOneCAdapter` читает существующие
JSON или локальные файлы и атомарно пишет канонический ответ в локальную
исходящую очередь. `HttpOneCAdapter` реализует подтверждённый контракт обратного
вызова 1С.

Настройки:

```text
ONEC_INTEGRATION_ENABLED=false
ONEC_ADAPTER=mock
ONEC_API_URL=
ONEC_AUTH_TOKEN=
ONEC_TIMEOUT_SECONDS=15
ONEC_MAX_DELIVERY_ATTEMPTS=3
ONEC_MOCK_OUTBOX_DIR=runtime/onec/mock_outbox
```

Реальный HTTP-адаптер отправляет `POST ONEC_API_URL` с телом, зафиксированным
контрактом 1С (орфография имени `recomendation` сохранена намеренно):

```json
{
  "order_id": "УИД заказа из external_id",
  "recomendation": "полный человекочитаемый comment_1c"
}
```

Успешной считается только доставка с HTTP 200. По умолчанию клиент ожидает
ответ не более 15 секунд. Для включения используются
`ONEC_INTEGRATION_ENABLED=true`, `ONEC_ADAPTER=http` и полный URL вида
`http://host/.../API/v1/sendAiAnswer`. Авторизация не добавляется, пока
`ONEC_AUTH_TOKEN` пуст; при заданном токене отправляется Bearer-заголовок.

Для локальной проверки имитации отправки установите
`ONEC_INTEGRATION_ENABLED=true`. Обработчик сохранит результат в PostgreSQL до
вызова адаптера. Состояние доставки находится в PostgreSQL в колонках
`onec_delivery_*`; отдельных сопутствующих файлов для 1С больше нет. Ошибка
обратной отправки оставляет само задание `COMPLETED` и помечает доставку как
`PENDING`.

Ручная повторная отправка уже сохранённого payload без OCR:

```powershell
.\.venv-gpu\Scripts\python.exe -m app.integrations.retry --job-id <UUID>
```

Повторно доставленный через Redis финальный `job_id` также повторит только
отправку `PENDING`, а не проверку. Количество автоматических попыток ограничено
`ONEC_MAX_DELIVERY_ATTEMPTS`; явная операторская команда разрешает ещё одну
попытку. OCR, механизм правил, управление обработчиком и входной POST-контракт при
обратной доставке не меняются.

Упрощённый пример результата:

```json
{
  "decision": "APPROVE",
  "extracted_values": {
    "application_number": "76930",
    "total_quantity": "69.284",
    "buyer_seal_present": "PRESENT"
  },
  "checks": [
    {
      "code": "APPLICATION_NUMBER",
      "status": "PASS",
      "confidence": 0.99
    }
  ],
  "reasons": ["Все обязательные проверки пройдены"],
  "comment_1c": "Спецификация согласована. Все обязательные проверки пройдены",
  "confidence": {
    "overall": 0.95,
    "fields": {"application_number": 0.99}
  },
  "metadata": {
    "processing_time_seconds": 12.4,
    "page_count": 1
  }
}
```

## Docker-развёртывание

Требуются Docker Compose и NVIDIA Container Toolkit. GPU-обработчик использует
PyTorch 2.7.1/CUDA 12.8 с поддержкой Blackwell `sm_120`; Compose резервирует
одно GPU-устройство и ровно одну реплику обработчика. На Blackwell применяется
PyTorch SDPA, поскольку Flash Attention 2 не поддерживает вычислительную
архитектуру 12.0. Модель подключается внешним монтированием только для чтения.

Подготовьте локальный `.env`:

```powershell
Copy-Item .env.example .env
```

Обязательно измените `POSTGRES_PASSWORD` и укажите существующий внешний кэш:

```text
MONKEYOCR_MODEL_PATH=C:/Users/Rodion/AppData/Local/specification-checker/huggingface
```

В режиме разработки Swagger и имитационный адаптер 1С включены, срок хранения
файлов заданий — 7 дней.

```powershell
docker compose -f docker-compose.yml -f compose.development.yml up -d --build
docker compose -f docker-compose.yml -f compose.development.yml ps
docker compose -f docker-compose.yml -f compose.development.yml logs -f worker
```

Миграции выполняются отдельным одноразовым сервисом `migrate` до API и обработчика.
Повторный ручной запуск безопасен:

```powershell
docker compose -f docker-compose.yml -f compose.development.yml run --rm migrate
```

Локальная проверка:

```powershell
curl.exe http://127.0.0.1:8000/health
Start-Process http://127.0.0.1:8000/docs
.\.venv-gpu\Scripts\python.exe -m pytest -q
```

В production-режиме Swagger отключён, срок хранения составляет 30 дней,
имитация обратного вызова выключена.

```bash
docker compose -f docker-compose.yml -f compose.production.yml up -d --build
docker compose -f docker-compose.yml -f compose.production.yml ps
```

Сервисы:

- `api` — лёгкий Python-образ без Torch и модели;
- `worker` — CUDA-обработчик с одним `SpecificationCheckerService`;
- `postgres` — PostgreSQL 18;
- `redis` — Redis с AOF;
- `migrate` — `alembic upgrade head` и завершение.

Постоянные тома:

- `jobs_data` — общий `JOB_STORAGE_ROOT` API и обработчика;
- `postgres_data` — PostgreSQL 18 (`/var/lib/postgresql`);
- `redis_data` — AOF очереди;
- `runtime_cache` — OCR-кэш и кэш фрагментов обработчика;
- `runtime_logs` — рабочие журналы;
- `onec_outbox` — файлы только имитационного адаптера.

При переносе на сервер копируются исходники и защищённый `.env`. Модель заранее
загружается в отдельный серверный каталог и задаётся через
`MONKEYOCR_MODEL_PATH`; каталог весов не попадает в контекст сборки. Перед
обновлением создайте резервную копию тома PostgreSQL, затем соберите образы и
запустите `migrate`. Секреты production-среды не должны храниться в Git.

### Хранение файлов заданий

Единственный обработчик удаляет каталоги только финальных заданий старше заданного срока.
Перед удалением проверяется путь `JOB_STORAGE_ROOT/{job_id}`. В PostgreSQL
остаются задание, решение, полный результат, SHA-256 и состояние 1С; очищается только
`document_path` и файловый каталог.

```text
JOB_CLEANUP_ENABLED=true
JOB_RETENTION_DAYS=30
JOB_CLEANUP_INTERVAL_SECONDS=3600
JOB_CLEANUP_BATCH_SIZE=100
```

## Установка на Windows

Требуется Python 3.11–3.12. Для локальной проверки модуля выбора страниц также нужен
Tesseract OCR с русским языковым пакетом. Для локального запуска MonkeyOCR рекомендуется
GPU NVIDIA; текущая конфигурация проверена на RTX 3050 Laptop с 4 ГБ видеопамяти.

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-monkeyocr.txt
.\.venv-gpu\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
```

Однократная загрузка официальной версии весов:

```powershell
$env:HF_HOME = "$env:LOCALAPPDATA\specification-checker\huggingface"
.\.venv-gpu\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zenosai/MonkeyOCRv2-B-Parsing', revision='de7a993bd0f39a97b122dac767e82ae04935bce4')"
```

Во время проверки в production-среде сеть не используется: в конфигурации установлено
`local_files_only: true`.

## Установка на Linux-сервер

```bash
cd /srv/specification-checker
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-monkeyocr.txt
python -m pip install -e . --no-deps --no-build-isolation

export HF_HOME=/srv/specification-checker-models/huggingface
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zenosai/MonkeyOCRv2-B-Parsing', revision='de7a993bd0f39a97b122dac767e82ae04935bce4')"
```

Для постоянного серверного пути можно добавить в `ocr.monkeyocr_v2`:

```yaml
cache_dir: /srv/specification-checker-models/huggingface
local_files_only: true
```

Переменная `HF_HOME` и `cache_dir` должны указывать на один каталог.

## Проверка одного документа

Запуск через Git Bash:

```bash
./.venv-gpu/Scripts/python.exe -m app.cli check \
  --document "testdata/specifications/documents/Спецификация 40.pdf" \
  --reference "testdata/specifications/reference/specification_40.json" \
  --output-dir "runtime/reports/specification_40_monkey" \
  --ocr-backend monkeyocr-v2 \
  --device auto \
  --reuse-ocr-cache
```

`--force-ocr` принудительно запускает модель. `--reuse-ocr-cache` повторно
использует результат OCR, если документ, настройки рендеринга, версия адаптера,
версия весов и параметры модели не изменились.

## Пакетный OCR

Одна модель инициализируется один раз и последовательно обрабатывает документы:

```powershell
.\.venv-gpu\Scripts\python.exe -m app.cli ocr-batch `
  --documents "incoming\51.pdf" "incoming\52.pdf" "incoming\53.pdf" `
  --output "runtime\reports\ocr_batch\summary.json" `
  --device auto `
  --reuse-ocr-cache
```

## Рабочие файлы и диагностика

```text
runtime/
  reports/   итоговые отчёты
  cache/     компактный gzip OCR-кэш и кэш распознанных фрагментов
  logs/      ротируемые логи
  temp/      временные файлы, очищаемые после запуска
```

По умолчанию используется `--debug-level none`: промежуточные изображения не
сохраняются. Доступны:

- `--debug-level errors` — только материалы проблемных полей;
- `--debug-level full` — полная диагностика конкретного документа;
- `--save-debug` — алиас для `--debug-level full`.

Безопасная предварительная оценка очистки:

```powershell
.\.venv-gpu\Scripts\python.exe -m app.cli cleanup --all --dry-run
```

Фактическая очистка рабочего кэша и временных диагностических материалов:

```powershell
.\.venv-gpu\Scripts\python.exe -m app.cli cleanup --all
```

## Тестирование

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -m "not slow"
```

Тяжёлые OCR-прогоны не входят в быстрый набор. Модульные тесты нормализаторов,
модулей извлечения и механизма правил работают без запуска модели; сохранённый
OCR-кэш позволяет повторно проверять извлечение данных без генерации текста.

## Ограничения

- Четырёх или нескольких десятков PDF недостаточно для доказательства
  промышленной точности.
- Детекторы подписи и печати эвристические: они подтверждают визуальные
  признаки, но не устанавливают личность подписанта и подлинность оттиска.
- MonkeyOCR не предоставляет вероятность каждого токена; уверенность поля
  рассчитывается модулями извлечения по качеству, полноте и геометрии доказательств.
- При противоречивом или недостаточном результате система сохраняет
  fail-closed поведение и возвращает `MANUAL_REVIEW`.
