## Skin Analyze API

Пайплайн: MedGemma-4B-it (визуальный анализ) → Qwen3-30B-A3B-Thinking-2507 (reasoning + tool calling) → поиск товаров через Google CSE → ответ.

### Быстрый старт (локально)
1) Python 3.10+
2) Установка зависимостей:
```bash
pip install -r requirements.txt
```
3) Создать `.env` на основе `.env.example`.
4) Запуск API:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Эндпоинт
POST `/analyze`
- Вход: multipart `image` (файл) или JSON `{ "image_url": "...", "text": "..." }`
- Выход: JSON с диагнозом, описанием состояния кожи и списком товаров (до 5)

### Docker
См. `docker-compose.yml`, `Dockerfile.api`, `Dockerfile.vllm`.
