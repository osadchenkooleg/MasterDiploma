# Makefile для основних задач

IMAGE_NAME = plagiarism-detector

# ---------- Local ----------
dev:                 ## Запустити сервер у режимі hot-reload
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:                ## Запустити тести
	pytest -q

format:              ## Вирівняти код black + isort
	isort .
	black .

# ---------- Docker ----------
build:               ## Зібрати docker-образ
	docker build -t $(IMAGE_NAME) .

run: build           ## Запустити контейнер на :8000
	docker run -p 8000:8000 --rm $(IMAGE_NAME)

# ---------- Help ----------
help:                ## Список доступних команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
