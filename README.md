# Heart Disease API

Учебный Flask-сервис классификации на основе [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/). GET- или POST-запрос передаёт X-вектор из 13 признаков, ответ содержит целевую переменную `target` (0 или 1), предсказанную RandomForestClassifier.

Структура построена по примеру `production-code-jun`: конфигурация вынесена в `config`, этапы ML — в `src`, модели и метрики — в `models`. По запросу проект переведён на Flask с GET `/predict`. Исходное условие домашнего задания требует FastAPI; текущая версия использует Flask.

```text
config/variables.py    пути, seed, размер тестовой выборки
data/heart.csv        исходные данные
src/preprocessing.py  валидация, удаление дубликатов, train/test split
src/train.py          обучение и сохранение Random Forest
src/inference.py      загрузка и предсказание
src/evaluate.py       метрики на тестовой выборке
src/pipeline.py       полный цикл обучения
src/app.py            точка входа Flask / Gunicorn
app/main.py           маршруты Flask и загрузка модели при старте
app/schema.py         схема и проверка входных признаков
models/               model.joblib и metrics.json
tests/                тесты API и pipeline
test_request.py       проверка сервиса HTTP GET и POST
Dockerfile
docker-compose.yml
Makefile
```

## Запуск в Docker

Из каталога `heart-disease-api`:

```bash
docker compose up -d --build
docker compose ps
```

Модель обучается во время сборки образа. HTTP-запросы обслуживает Gunicorn. Контейнер запускается от пользователя без root-прав; API загружает модель один раз при старте. Сервис доступен на http://127.0.0.1:8000. Автоматической страницы Swagger `/docs` нет. Проверка готовности: GET `/health` (также `/`).

```bash
python3 test_request.py
curl 'http://127.0.0.1:8000/predict?age=63&sex=1&cp=3&trestbps=145&chol=233&fbs=1&restecg=0&thalach=150&exang=0&oldpeak=2.3&slope=0&ca=0&thal=1'
```

Ответ имеет вид `{"target": 1}` (значение зависит от признаков и модели). В Postman выберите GET и вставьте этот URL; тело запроса не требуется. Ошибочные, лишние или отсутствующие параметры возвращают HTTP 422.

Остановка:

```bash
docker compose down
```

## Локальный запуск и тесты

Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m src.pipeline
.venv/bin/python -m pytest -q
.venv/bin/python -m flask --app src.app:app run --host 127.0.0.1 --port 8000
```

В другом терминале: `.venv/bin/python test_request.py`. Скрипт также принимает URL сервера первым аргументом. Запуски `python train.py` и `python src/train.py` также выполняют полный pipeline; прямой запуск `src/train.py` из IDE работает независимо от рабочего каталога. Команды Makefile: `make train`, `make test`, `make run`, `make up`, `make request`, `make down`.

Путь к модели можно переопределить переменной окружения `MODEL_PATH`. Если файл отсутствует или схема модели не соответствует API, запуск завершается ошибкой; сначала выполните pipeline.

## Признаки

| Параметр | Описание | Допустимые значения API |
|---|---|---|
| age | Возраст, лет | целое 1–120 |
| sex | Код пола | 0, 1 |
| cp | Код типа боли в груди | 0–3 |
| trestbps | Давление в покое, мм рт. ст. | > 0, ≤ 300 |
| chol | Холестерин, мг/дл | > 0, ≤ 1000 |
| fbs | Сахар натощак > 120 мг/дл | 0, 1 |
| restecg | Код ЭКГ в покое | 0–2 |
| thalach | Максимальная частота пульса | > 0, ≤ 250 |
| exang | Стенокардия при нагрузке | 0, 1 |
| oldpeak | Депрессия ST при нагрузке | 0–10 |
| slope | Код наклона ST | 0–2 |
| ca | Код количества крупных сосудов | 0–4 |
| thal | Код thal в исходном CSV | 0–3 |

Порядок query-параметров не влияет на результат. Порядок столбцов для модели задаётся явно. `target` не передаётся в запросе. Категориальные коды сохраняются как в CSV; ограничения API описывают входной контракт учебного сервиса.

## Обучение и результаты

Исходный CSV: 1025 строк, после удаления повторов — 302. Разделение 80/20 со стратификацией, `random_state=42`: 241 строка для обучения и 61 для теста. Модель: Random Forest, 200 деревьев. Дубликаты удаляются **до** разделения, чтобы исключить попадание одинаковых наблюдений в обучение и тест. Pipeline также проверяет пропуски, схему, диапазоны признаков и конфликтующие метки.

При проверке: accuracy = 0,7541; ROC-AUC = 0,8701. Полный classification report сохраняется в `models/metrics.json`. Эти оценки относятся к небольшой отложенной выборке. Ответ API — класс из датасета; проект предназначен для учебного задания.


## POST /predict с JSON

Готовый входной JSON: `examples/predict.json`. Из каталога проекта:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/predict.json
```

В Postman: POST → `http://127.0.0.1:8000/predict` → Body → raw → JSON. Вставьте содержимое `examples/predict.json`. Ответ: `{"target": 1}` для данного примера.

GET и POST используют одну модель и схему признаков. POST принимает признаки из тела JSON, GET — из URL. Неверные признаки или тело, которое не является объектом, возвращают JSON с HTTP 422; повреждённый JSON — 400, неправильный Content-Type — 415. Скрипт `test_request.py` проверяет оба метода и совпадение результатов.
