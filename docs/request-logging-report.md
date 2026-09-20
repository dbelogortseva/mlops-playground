# Запись запросов /v1/predict в PostgreSQL

При непустом `DATABASE_URL` приложение создаёт таблицу `prediction_requests`
при старте. Каждый запрос по пути `/v1/predict` записывается одной строкой
перед отправкой ответа, включая ошибки 422, 500, 503 и неверный метод 405.
Запросы `/health`, `/ready`, `/docs` в эту таблицу не попадают.

| Поле | Тип PostgreSQL | Содержимое |
| --- | --- | --- |
| `request_id` | `uuid`, первичный ключ | Идентификатор запроса |
| `requested_at` | `timestamptz` | Время начала запроса, UTC |
| `model_version` | `text` | Версия модели; NULL, если модель не загружена |
| `features` | `jsonb` | Исходный JSON запроса, включая лишние/невалидные поля |
| `prediction` | `double precision` | Предсказание; NULL при ошибке |
| `latency_ms` | `double precision` | Время обработки без записи в БД |
| `status_code` | `smallint` | Фактический код HTTP-ответа |

Для успешного запроса `request_id`, `model_version`, `prediction` и `latency_ms`
точно совпадают с ответом API. Заголовок `X-Request-ID` возвращается также
при ошибках. Для ошибок время измеряется до получения ответа обработчика.
Пропущенные поля остаются пропущенными в журнале, а явные `null` — `null`.
Если тело невозможно представить как JSONB (например, сломанный JSON или
NaN), исходные байты сохраняются в объекте с полями `_raw_body_hex` и
`_encoding: "hex"`.

Без `DATABASE_URL` нет ни создания таблицы, ни подключения, ни записи.
Если переменная задана, но БД недоступна, сервис продолжает отвечать и пишет
ошибку в журнал сервера. Запись потерянного запроса автоматически не повторяется.
После ошибки создания таблицы восстановите доступ к БД и перезапустите API.
Это простое синхронное подтверждение записи без очереди повторной доставки.
SQL выполняется в отдельном потоке и использует параметры, а не подстановку
значений запроса в текст SQL.

## Проверка для отчёта

Нужен запущенный Docker Desktop. Команды рассчитаны на PowerShell.
Остановите старый сервер комбинацией Ctrl+C, чтобы запустить обновлённый код.

### 1. Запустить PostgreSQL и API

В первом терминале:

```powershell
cd C:\Users\Даша\Desktop\Courses\ML_PRO\dz1\mlops-playground
uv sync --locked
docker compose up -d --wait db
$env:DATABASE_URL = "postgresql://mlops:mlops_local@127.0.0.1:5433/mlops"
uv run uvicorn mlops_playground.api:app --port 8000
```

Таблица создаётся автоматически. Переменная задаётся именно в терминале,
где запускается сервер. Реквизиты в compose.yaml предназначены для локальной
учебной БД; порт 5433 доступен только на этом компьютере.

### 2. Отправить успешный и ошибочный запросы

Во втором терминале:

```powershell
cd C:\Users\Даша\Desktop\Courses\ML_PRO\dz1\mlops-playground
curl.exe -i -X POST http://127.0.0.1:8000/v1/predict -H "Content-Type: application/json" --data-binary "@examples/predict.json"
curl.exe -i -X POST http://127.0.0.1:8000/v1/predict -H "Content-Type: application/json" --data-binary "@examples/predict_invalid_bounds.json"
```

Ожидаются 200 с предсказанием и 422 для `month=13`. Сделайте скриншот
ответов, включая `request_id` / `X-Request-ID`.
Можно отправить те же запросы через http://127.0.0.1:8000/docs.

### 3. Показать строки таблицы

```powershell
docker compose exec -T db psql -U mlops -d mlops -c "SELECT request_id, requested_at, model_version, prediction, latency_ms, status_code FROM prediction_requests ORDER BY requested_at DESC LIMIT 2;"
```

Должны появиться две новые строки: с `status_code=200` и `status_code=422`.
Для 422 поле `prediction` пустое — это SQL NULL.
UUID успешной строки должен совпадать с `request_id` ответа.

Проверка JSONB и значений признаков:

```powershell
docker compose exec -T db psql -U mlops -d mlops -c "SELECT request_id, pg_typeof(features) AS features_type, features->>'month' AS month, features->>'lag_1' AS lag_1, status_code FROM prediction_requests ORDER BY requested_at DESC LIMIT 2;"
```

Ожидается тип `jsonb`; месяц в ошибочном запросе равен 13, в успешном — 12.
Для просмотра полного JSON последнего запроса:

```powershell
docker compose exec -T db psql -U mlops -d mlops -c "SELECT jsonb_pretty(features) FROM prediction_requests ORDER BY requested_at DESC LIMIT 1;"
```

Сделайте скриншоты таблицы и результата проверки типа `jsonb`.
В БД могут уже находиться строки моей автоматической проверки — сравнивайте
UUID и время, а не ожидайте, что общее число строк будет равно двум.

### 4. Проверить работу без DATABASE_URL

Во втором терминале запомните число строк:

```powershell
docker compose exec -T db psql -U mlops -d mlops -c "SELECT count(*) FROM prediction_requests;"
```

В первом терминале остановите API через Ctrl+C, удалите переменную и запустите снова:

```powershell
Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
uv run uvicorn mlops_playground.api:app --port 8000
```

Во втором терминале повторите успешный запрос и подсчёт строк:

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/v1/predict -H "Content-Type: application/json" --data-binary "@examples/predict.json"
docker compose exec -T db psql -U mlops -d mlops -c "SELECT count(*) FROM prediction_requests;"
```

Ожидается **200**, при этом число строк **не изменилось**.
Во время сравнения не отправляйте запросы из другого экземпляра API с
включённым логированием. Сделайте скриншоты счётчика до/после и ответа 200.

После проверки остановите API через Ctrl+C. БД можно остановить командой:

```powershell
docker compose stop db
```

Данные сохраняются в Docker volume и доступны после следующего `up`.

## Краткий текст для отчёта

> Добавлено логирование запросов `/v1/predict` в PostgreSQL. Таблица
> `prediction_requests` содержит UUID запроса, время, версию модели,
> признаки JSONB, предсказание, задержку и код HTTP-ответа. Логируются
> как успешные запросы, так и ошибки валидации. Для ошибок предсказание
> хранится как NULL. При отсутствии DATABASE_URL API продолжает возвращать
> предсказания, но не создаёт новых строк в таблице.

К тексту приложите выполненные проверки и ссылку на коммит.
Автотесты не являются отдельным требованием задания. Для повторения
дополнительной проверки с настоящей БД:

```powershell
$env:TEST_DATABASE_URL = "postgresql://mlops:mlops_local@127.0.0.1:5433/mlops"
uv run pytest -q
```

Проверено локально: 46 тестов прошли, включая запись в настоящую PostgreSQL
ответов 200, 422, 500 и 503. Без `TEST_DATABASE_URL` интеграционный тест
пропускается: 45 passed, 1 skipped.

Также проверен настоящий HTTP-сервер: с `DATABASE_URL` четыре запроса
добавили ровно четыре строки; без переменной те же запросы получили
ожидаемые ответы, а число строк в БД осталось прежним.

Для записи JSONB используется официальный адаптер
[Psycopg Jsonb](https://www.psycopg.org/psycopg3/docs/basic/adapt.html#json-adaptation).
