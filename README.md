# MLOps Playground

В репозитории хранится модель предсказания количества покупок электронной коммерции. 
Модель принимает на вход **68 признаков**.

Обязательные признаки:

| Признак | Описание |
|---|---|
| `day_of_week` | День недели |
| `month` | Месяц |
| `is_weekend` | Признак выходного дня |
| `dow_sin` | Синусоидальное кодирование дня недели |
| `dow_cos` | Косинусоидальное кодирование дня недели |
| `year_sin` | Синусоидальное кодирование годового цикла |
| `year_cos` | Косинусоидальное кодирование годового цикла |


## Задание 1.

Была выполнена доработка: само выполнение заданий не менялось, изменился только отчет (ранее он почти не содержал подробностей).

### Проверка

Требуются `uv`, Docker, `kind` и `kubectl`.

```powershell
uv sync --frozen
uv run python -c "import mlops_playground"
uv run pytest

docker build -t mlops-playground:1.0 .
docker compose up -d --build
docker compose ps

if (-not ((kind get clusters) -contains "mlops-playground")) { kind create cluster --name mlops-playground }
kubectl config use-context kind-mlops-playground
kind load docker-image mlops-playground:1.0 --name mlops-playground

kubectl create secret generic mlops-secrets --from-literal=POSTGRES_PASSWORD=postgres --from-literal=DATABASE_URL="postgresql://postgres:postgres@postgres:5432/mlops" --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/
kubectl rollout status deployment/postgres --timeout=180s
kubectl rollout restart deployment/mlops-api
kubectl rollout status deployment/mlops-api --timeout=180s

kubectl get deployments
kubectl get pods
```
Отдельная проверка Kubernetes.

В первом терминале:

```powershell
kubectl port-forward service/mlops-api 8080:80
```
Во втором терминале (из корня репозитория):
```powershell
curl.exe --fail --silent http://127.0.0.1:8080/health
curl.exe --fail --silent http://127.0.0.1:8080/ready
curl.exe --fail --silent -X POST http://127.0.0.1:8080/v1/predict -H "Content-Type: application/json" --data-binary "@good.json"
```



### Результаты

#### 1. Артефакт с паспортом

* Исходное обучение модели описано в файле `notebooks/Model.ipynb`. 
* Создание и сохранение joblib-бандла, состоящего из pipeline и metadata, находится в файле `notebooks/train_model.ipynb`.
* `model.joblib` находится в папке `artifact`.

#### 2. Проект на uv

* `good.json` - пример корректного тела запроса.

![Проект на uv](screenshots/1.jpg)

#### 3. Сервис

Реализован FastAPI-сервис для получения предсказаний модели. Сервис предоставляет эндпоинты `/health` для проверки работоспособности, `/ready` для проверки готовности модели, `/v1/predict` для выполнения предсказания и `/docs` с автоматически сформированной документацией OpenAPI.
Входная схема `/v1/predict` описывает признаки модели и проверяет их типы и допустимые значения. Обязательные признаки должны присутствовать в запросе, а признаки, в которых возможны пропуски, объявлены опциональными. Параметр `extra="forbid"` запрещает передачу неизвестных полей. При нарушении контракта сервис возвращает код `422`.

![Сервис](screenshots/3.jpg)

#### 4. Логи запросов в PostgreSQL

Каждый запрос к `/v1/predict`, включая ошибочные запросы с кодом `422`, сохраняется в таблице `predictions`. Запись содержит request_id, время запроса, версию модели, входные признаки в формате jsonb, предсказание, latency_ms и HTTP-код ответа. Без переменной `DATABASE_URL` сервис продолжает работать без записи логов.

![Логи запросов в PostgreSQL](screenshots/2.jpg)


#### 5. Тесты

Реализовано 12 тестов, проверяющих контракт API, smoke-сценарии, детерминизм модели и интеграцию с PostgreSQL. Некорректные и лишние поля приводят к ответу `422`, корректный запрос возвращает ожидаемые типы и допустимое предсказание, а одинаковые входные данные дают одинаковый результат. Команда `uv run pytest` завершается успешно.

![Тесты](screenshots/8.jpg)


#### 5. Docker и compose

В `compose.yaml` настроены API и PostgreSQL, healthcheck базы и ожидание её готовности перед запуском API.

#### 6. Kubernetes

Для приложения подготовлены Kubernetes-манифесты с двумя репликами API, startupProbe, livenessProbe, readinessProbe, а также ограничениями requests и limits. Образ загружен в `kind`, манифесты применены, PostgreSQL и API успешно прошли rollout. Через `port-forward` получен корректный ответ от `/v1/predict`.

![Kubernetes](screenshots/5.jpg)

![Kubernetes](screenshots/7.jpg)

![Kubernetes](screenshots/6.jpg)

#### Поды в k9s

![Kubernetes](screenshots/Untitled.png)

#### Журнал проблем

##### Kubernetes не мог создать контейнеры

После применения манифестов Pod API и PostgreSQL имели статус: `CreateContainerConfigError`
Причиной было отсутствие Secret mlops-secrets, на который ссылались Deployment. После создания Secret с ключами POSTGRES_PASSWORD и DATABASE_URL PostgreSQL запустился, а Deployment API успешно обновился до двух готовых реплик.

В остальном всё получилось повторить почти сразу.
