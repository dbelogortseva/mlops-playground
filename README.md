# MLOps Playground

Ориентироваться в README.md можно по заголовку с номером задания.

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

## Задание 2.

### Результаты

#### 1. Пайплайн для своего сервиса

Ссылка на зелёный прогон со всеми тремя job: https://github.com/dbelogortseva/mlops-playground/actions/runs/37151305931
Ссылка на страницу пакета с образом, в теге виден sha коммита: https://github.com/dbelogortseva/mlops-playground/pkgs/container/mlops-playground/1332248128?tag=sha-16fd24c4cd8ca2da718fbe449ffdd18253b799ce

#### 2. Процесс: ветка и pull request

Была создана новая ветка `week2_red_test`. В файле `tests/test_smoke.py` код ошибки 200 был заменен на 201. В результате не был пройден job test.

Ссылка на пулл-реквест: https://github.com/dbelogortseva/mlops-playground/pull/3.

#### 3. Три красных прогона с диагнозом

1. **Конфиг**
    В файле `k8s/configmap.yaml` параметр `MODEL_PATH: artifact/model.joblib` был заменен на `MODEL_PATH: artifact/model1.joblib`.

    Этап deploy выполнялся долго: максимальное время стоит 180 секунд, поэтому этап "сервис" в job deploy упал через 3 минуты после начала выполнения. По диагностике видно, что возникла ошибка с ненайденным файлом.

    ```powershell
    FileNotFoundError: [Errno 2] No such file or directory: 'artifact/model1.joblib'
    ```
   
    Ссылка на красный прогон: https://github.com/dbelogortseva/mlops-playground/actions/runs/37153558266/job/111292423208
    Ссылка на зеленый прогон: https://github.com/dbelogortseva/mlops-playground/actions/runs/37155343758/job/111297693887

2. **Секрет**
   
    В файле `k8s/deployment.yaml` параметр `secretKeyRef: name: mlops-secrets` был заменен на `secretKeyRef: name: `mlops-secrets-wrong`.

    Сломался job deploy, а именно этап "сервис". Этап "подтягиваем секреты" не сломался, потому что подтягивает секрет из `ci.yml`. 
    Этап "сервис" отрабатывал максимальное время, то есть 3 минуты. В диагностике также видно `CreateContainerConfigError`.
  
    Ссылка на красный прогон: https://github.com/dbelogortseva/mlops-playground/actions/runs/37155976221/job/111299469025
    Ссылка на зеленый прогон: https://github.com/dbelogortseva/mlops-playground/actions/runs/37156731230/job/111301730473

3. **Ресурсы**

    В файле `k8s/deployment.yaml` параметр `resources` был заменен на memory: `1000000000Mi`.
    Сломался job deploy, а именно этап "сервис". Этот этап отработал 3 минуты и после этого выдал ошибку. 
    В "диагностике" есть указание на то, что поды не запустились (у них был статус Pending), но напрямую о нехватке ресурсов это не говорит.

    Ссылка на красный прогон: https://github.com/dbelogortseva/mlops-playground/actions/runs/37157284357/job/111303361752
    Ссылка на зеленый прогон: https://github.com/dbelogortseva/mlops-playground/actions/runs/37160214846/job/111312088417

4. **Семь вопросов**
   
   * Кэширование build
      
     Первый прогон: https://github.com/dbelogortseva/mlops-playground/actions/runs/37160214846/job/111312024490 - длительность build 23 секунды

     Второй прогон: https://github.com/dbelogortseva/mlops-playground/actions/runs/37160609170/job/111313175818 - длительность build 30 секунд
   
     Кэш использовался для слоя установки зависимостей, так как пакеты повторно не устанавливались. Также `CACHED` отмечены у копирования `pyproject.toml`, `uv.lock`, исходного кода, модели и установки самого проекта.          Значит соответствующие входные файлы и команды не изменились.

   * ImagePullBackOff
  
     `kubectl apply -f k8s/` просит запустить `mlops-playground:1.0`. Такого образа в новом кластере нет. Появляются поды с ошибкой. `kubectl set image` меняет образ на `ghcr.io/...:sha-...`. Создаются правильные поды, они запускаются, а предыдущие удаляются. Зеленый прогон означает, что приложение запустилось и ответило на проверочный запрос.
     Чтобы избежать временной ошибки, можно подставлять нужное имя образа в манифест до его применения.

   * Пароль от БД

     Пароль в настройках GitHub -> GitHub Secret `DB_PASSWORD` -> Переменная `DB_PASSWORD` в шаге CI -> Kubernetes Secret `mlops-secrets` -> `POSTGRES_PASSWORD` в контейнере PostgreSQL и `DATABASE_URL` в контейнере FastAPI

     Приложение через настройки читает `DATABASE_URL` и использует его для подключения к PostgreSQL.

     Мы можем положить пароль в `configmap.yaml`, но тогда он будет в репозитории и его последующее удаление не сможет полностью избавиться от пароля, так как он останется в истории коммитов.
     Если же к базе можно будет подключиться по сети, то имея пароль, другие люди смогут видеть мою БД (а не свою локальную копию) и, например, удалить её.

   * Убрать `needs` у `build`
  
      ```powershell
      build:
         needs: tests
      ```
      Данная настройка означает, что для запуска джобы build нужен корректно завершенный прогон джобы tests. Если мы уберем `needs: tests`, то tests и build станут независимыми задачами.
      Если этап tests не завершился корректно, то build может спокойно собраться, тогда начнет работать этап deploy, который тоже может корректно собраться, несмотря на красный tests.

   * `pull request`

     Это сделано, чтобы разделить проверку предложенных изменений, сохранение готового образа и проверку развертывания.

     В pull request автоматически делаются только тесты, потому что пока происходит этап pull request, потому что не требуется публиковать новый образ для каждого изменения в коде. Но базовые проверки работоспособности кода и тесты нужны.

     За это отвечает строка:
     
     ```powershell
     on:
        push: {branches: [main]}
        pull_request:
     ```

   * `pg_advisory_xact_lock`
  
     `pg_advisory_xact_lock` обеспечивает последовательную инициализацию таблицы при одновременном запуске двух реплик API. Без него на пустой базе возможна гонка: оба экземпляра приложения попытаются одновременно создать таблицу `predictions`, и один запуск может завершиться ошибкой. Здесь имеются в виду две реплики FastAPI mlops-api, работающие с одной общей базой PostgreSQL.

   * Порядок жизни пода
      По этапам запуска статусы располагаются так:

      Pending -> CreateContainerConfigError -> CrashLoopBackOff

      При нехватке ресурсов Pod не получает узел; при отсутствующем Secret Kubernetes не может подготовить контейнер; при неправильном пути к модели контейнер запускается, но приложение падает и перезапускается.

     
   
    

     
  
     
