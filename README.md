## Skin Analyze API

Пайплайн: MedGemma-4B-it (визуальный анализ) → Google Gemini (reasoning + tool calling) → поиск товаров через Google CSE → ответ.

### Переменные окружения
- GEMINI_API_KEY — ключ API для Google Gemini (обязательно)
- GEMINI_MODEL — имя модели Gemini (по умолчанию `gemini-1.5-pro`)
- GOOGLE_CSE_API_KEY — ключ Google Custom Search (для поиска товаров)
- GOOGLE_CSE_CX — CX идентификатор для Google Custom Search

### Быстрый старт (локально)
1) Python 3.10+
2) Установка зависимостей:
```bash
pip install -r requirements.txt
```
3) Создать `.env` c переменными из раздела выше.
4) Запуск API:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Эндпоинт
POST `/analyze`
- Вход: multipart `image` (файл) или JSON `{ "image_url": "...", "text": "..." }`
- Выход: JSON с диагнозом, описанием состояния кожи и списком товаров (до 5)

### Swagger UI (отдельный порт)
- В docker-compose поднимается контейнер Swagger UI на http://localhost:8001
- Он использует спецификацию API с `http://localhost:8000/openapi.json`
- При изменении кода API перезапускается (uvicorn `--reload`), Swagger UI отобразит новую схему после обновления страницы в браузере

### Docker
См. `docker-compose.yml`, `Dockerfile.api`.

- Сервис `api` теперь не зависит жестко от сервиса `vllm` и может стартовать самостоятельно (используется Gemini API).
- Сервис `vllm` оставлен опционально (при необходимости локального поднятия LLM), но пайплайн умолчательно работает без него.
