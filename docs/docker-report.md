# Docker и Compose

## Реализация

В Dockerfile используется Python 3.13 и uv 0.12.17.
Сначала копируются `pyproject.toml`, `uv.lock`, `.python-version` и выполняется
`uv sync --frozen --no-dev --no-install-project`. Затем копируются README и
исходники, выполняется `uv sync --frozen --no-dev --no-editable`, добавляется
артефакт модели. Таким образом, изменения кода не сбрасывают слой установки
внешних зависимостей. При запуске используется уже установленный `uvicorn`;
повторной синхронизации и установки dev-зависимостей при старте нет.

`.dockerignore` исключает локальное окружение, Git, данные, тесты и другие
файлы, которые не нужны образу. Модель включена в образ.

В Compose определены `api` и `db`. PostgreSQL проверяется через `pg_isready`.
API зависит от `db` с `condition: service_healthy`, поэтому запускается после
готовности базы. У API также есть healthcheck по `/ready`.
DATABASE_URL передаётся через Compose и использует `db:5432` — адрес базы
внутри сети контейнеров. Для подключения с Windows используется
`127.0.0.1:5433`. Данные сохраняются в существующем volume `prediction_db`.

## Проверка для отчёта

Запустите Docker Desktop. Если ранее запускали `uvicorn` вручную на порту 8000,
остановите его через Ctrl+C. Далее команды выполняются из корня проекта:

```powershell
cd C:\Users\Даша\Desktop\Courses\ML_PRO\dz1\mlops-playground
docker compose up -d --build
docker compose ps
```

При первом запуске скачиваются образы и зависимости. Подождите, пока оба
сервиса будут `healthy`; при необходимости повторите `docker compose ps`.
Если сервис не готов, выполните `docker compose logs --tail 50 api db`.
DATABASE_URL вручную задавать не нужно; локальные Python и uv для запуска
Compose не требуются.

Проверьте API:

```powershell
curl.exe -i http://127.0.0.1:8000/ready
curl.exe -i -X POST http://127.0.0.1:8000/v1/predict -H "Content-Type: application/json" --data-binary "@examples/predict.json"
```

Ожидается 200. Сохраните UUID из `request_id` ответа.
Покажите строки таблицы:

```powershell
docker compose exec -T db psql -U mlops -d mlops -c "SELECT request_id, requested_at, model_version, prediction, latency_ms, status_code, pg_typeof(features) AS features_type FROM prediction_requests ORDER BY requested_at DESC LIMIT 5;"
```

Новая строка должна содержать тот же UUID, версию, предсказание и latency_ms,
что ответ API, `status_code=200` и тип признаков `jsonb`.
Ранее записанные строки сохраняются — отличайте новый запрос по UUID.

Дополнительная проверка отсутствия dev-зависимостей:

```powershell
docker compose exec -T api python -c "import importlib.util; assert importlib.util.find_spec('pytest') is None; assert importlib.util.find_spec('httpx') is None; print('No dev dependencies: OK')"
```

Для демонстрации повторного использования слоёв можно выполнить:

```powershell
docker compose --progress plain build api
```

При неизменных файлах ранее собранные слои помечаются `CACHED`.
Порядок слоя зависимостей и слоя кода виден непосредственно в Dockerfile.

Если порт 8000 занят и другой сервер нужно оставить работающим:

```powershell
$env:API_PORT = "8002"
docker compose up -d --build
```

Тогда используйте порт 8002 в HTTP-запросах. По умолчанию API_PORT равен 8000.

## Что приложить

1. Dockerfile с двумя командами `uv sync --frozen --no-dev` и порядком COPY.
2. Фрагмент compose.yaml: healthcheck базы и depends_on API с условием.
3. Скриншот `docker compose ps` с двумя здоровыми сервисами.
4. Ответ `/v1/predict` и результат SELECT с совпадающим request_id.
5. Ссылку на коммит.

Текст после успешной проверки:

> Подготовлен Dockerfile на uv с отдельным слоем зависимостей перед кодом.
> Установка выполняется из uv.lock с --frozen --no-dev. Compose запускает
> FastAPI и PostgreSQL; API ожидает service_healthy базы. После команды
> docker compose up -d --build запрос predict вернул 200 и записался в БД.
> SELECT подтвердил запись с тем же request_id и типом признаков jsonb.

Остановить оба сервиса, сохранив данные:

```powershell
docker compose stop
```

Справочные материалы: [uv в Docker](https://docs.astral.sh/uv/guides/integration/docker/),
[порядок запуска Compose](https://docs.docker.com/compose/how-tos/startup-order/).
