# Skin Analysis API

Система анализа кожи, разделенная на два микросервиса:
1. **Skin Analysis Service** - анализ изображений кожи и генерация рекомендаций
2. **Product Search Service** - поиск продуктов для ухода за кожей

## Архитектура

```
┌───────────────────┐    HTTP    ┌───────────────────┐
│   Skin Analysis   │───────────▶│  Product Search   │
│   Service         │            │   Service         │
│   (порт 8000)     │◀───────────│   (порт 8001)     │
└───────────────────┘            └───────────────────┘
```

## Запуск системы

### Предварительные требования
- Docker и Docker Compose
- Google Custom Search API ключи
- Gemini API ключ
- HuggingFace токен

### Настройка переменных окружения

Скопируйте `example.env` в `.env` и заполните необходимые значения:

```bash
cp example.env .env
# Отредактируйте .env файл с вашими ключами API
```

### Настройка Google Custom Search для получения цен

Для получения цен продуктов рекомендуется настроить Custom Search Engine следующим образом:

1. **Создайте Custom Search Engine:**
   - Перейдите в [Google Custom Search](https://cse.google.com/)
   - Создайте новый поисковик
   - Добавьте сайты: `ulta.com`, `sephora.com`, `amazon.com`, `nordstrom.com`

2. **Настройте Google Shopping:**
   - В настройках CSE выберите "Search the entire web"
   - Включите Google Shopping результаты
   - Укажите "skincare", "beauty", "cosmetics" как ключевые слова

3. **Активируйте Custom Search API:**
   - В [Google Cloud Console](https://console.cloud.google.com/) включите Custom Search API
   - Создайте API ключи
   - Включите billing для API

### Запуск сервисов

```bash
# Запуск всех сервисов
docker-compose up -d

# Или сборка и запуск
docker-compose up --build -d
```

### Проверка работоспособности

```bash
# Проверка основного API
curl http://localhost:8000/health

# Проверка сервиса поиска продуктов
curl http://localhost:8001/health

# Swagger UI
# http://localhost:8002
```

## API Endpoints

### Skin Analysis Service (порт 8000)

- `POST /v1/skin-analysis` - Начать анализ кожи
- `GET /v1/skin-analysis/status/{job_id}` - Получить статус анализа
- `GET /v1/skin-analysis/result/{job_id}` - Получить результат анализа
- `GET /health` - Проверка здоровья сервиса

### Product Search Service (порт 8001)

- `POST /v1/search-products` - Поиск продуктов
- `GET /health` - Проверка здоровья сервиса

## Структура проекта

```
├── app/                          # Основной сервис анализа кожи
│   ├── main.py                   # FastAPI приложение
│   ├── services/                 # Сервисы
│   │   ├── gemini_client.py      # Клиент Gemini AI
│   │   ├── product_search_client.py # Клиент поиска продуктов
│   │   └── ...
│   └── tools/                    # Инструменты
└── product_search_service/       # Микросервис поиска продуктов
    ├── main.py                   # FastAPI приложение
    ├── app/                      # Логика сервиса
    └── Dockerfile               # Docker конфигурация
```

## Разработка

### Запуск в режиме разработки

```bash
# Основной сервис
cd /app && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Сервис поиска продуктов
cd /product_search_service && uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Логирование

Оба сервиса используют структурированное логирование. Логи записываются в stdout и могут быть просмотрены через:

```bash
docker-compose logs api
docker-compose logs product_search
```

## Масштабирование

- **Skin Analysis Service** может масштабироваться горизонтально, но требует GPU ресурсов
- **Product Search Service** может масштабироваться горизонтально без специальных требований

Для масштабирования обновите `docker-compose.yml` и добавьте несколько реплик сервисов.
