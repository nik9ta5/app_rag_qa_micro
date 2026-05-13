# app_rag_qa_micro

Микросервисная RAG-платформа с локальным LLM (llama.cpp), векторной БД Qdrant и сервисом эмбеддингов на FastAPI + Sentence Transformers. Проект в активной разработке: часть компонентов уже работает, часть — в планах (см. Roadmap).

## Что уже есть

- **LLM-сервис** на `llama.cpp` в отдельном контейнере (OpenAI-совместимый API).
- **Vector Store** на базе **Qdrant**.
- **Embedding Service** на **FastAPI** + `sentence-transformers`.
- **RAG core** (CLI-прототип) для запроса к Qdrant + генерации ответа LLM.
- **Init-скрипты** для создания директорий и коллекции в Qdrant.
- **Docker Compose** с базовой инфраструктурой.

## Репозиторий

Основные директории:

- `llm_service/` — модели, кеши, скрипты LLM-сервиса
- `embedding_service/` — FastAPI сервис эмбеддингов
- `rag_core_service/` — прототип RAG-оркестратора
- `vector_store/` — данные Qdrant
- `init_scripts/` — инициализация директорий и коллекции

## Быстрый запуск (dev)

1. Создать `.env` (пример в `example.env`)
2. Создать директории:
```bash
python init_scripts/dir_init.py
```

3. Запустить инфраструктуру:
```bash
docker-compose up -d
```

### LLM-сервис (llama.cpp)

Контейнер стартует в режиме `sleep`, далее нужно зайти и вручную запустить модель.

```bash
docker exec -it llm_service bash
export LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH
/app/llama-server --model /models/gemma3_1b_it_bf16.gguf --host 0.0.0.0 --port 8080 --n-gpu-layers -1 --ctx-size 4096   --flash-attn on --parallel 2 --cont-batching --threads 6 --threads-batch 6 --verbose
```

### Инициализация коллекции в Qdrant

```bash
python init_scripts/vector_store_init.py
```

## Embedding Service

FastAPI сервис с двумя ручками:

- `POST /embed` — один текст
- `POST /embed_batch` — список текстов

Используется модель:
```
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

## RAG core (CLI прототип)

Находится в `rag_core_service/main.py`.

Запускается напрямую, ожидает input в консоли и генерирует ответ.

## Преимущества текущей реализации

- **Полностью локальный стек**: LLM + Embeddings + Vector Store.
- **Отдельные сервисы** с clear separation of concerns.
- **OpenAI-compatible API** для LLM (через llama.cpp).
- **FastAPI** для легко расширяемых микросервисов.
- **Готова база для реального RAG**: Qdrant + эмбеддинги + retrieval + prompt.

---

## Roadmap (план работ)

### 1. Данные и парсинг
- Выбрать тему приложения (область знаний)
- Подготовить PDF-статьи
- Парсинг PDF через PyMuPDF
- Извлечь текст

### 2. Препроцессинг
- Разбиение на чанки через LangChain splitter
- Формирование метаданных
- Хранение чанков

### 3. Векторная база
- Инициализация коллекций в Qdrant
- Загрузка данных с метаданными
- Тестирование поиска

### 4. Embedding сервис
- Сервис FastAPI
- SentenceTransformer модель
- Полная интеграция в пайплайн

### 5. RAG pipeline
- Полный RAG-сервис (FastAPI)
- Промпты и контекст
- Безопасность

### 6. Evaluation
- Генерация тест-сета (Gemini API)
- Скрипт оценки качества

### 7. DevOps/Repository
- Структурировать код
- Оформить репозиторий (README, схемы, лицензии)
- Вынести секреты
- Вынести зависимости
- Финальный тест и запуск
- Архитектурная схема

---

## TODO (коротко)
- Завершить data pipeline
- Сформировать dataset для оценки
- Финализировать RAG API
- Оформить документацию
