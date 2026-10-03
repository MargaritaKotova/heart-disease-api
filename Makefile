PYTHON ?= .venv/bin/python

.PHONY: install train run test build up down request
install:
	$(PYTHON) -m pip install -r requirements-dev.txt
train:
	$(PYTHON) -m src.pipeline
run:
	$(PYTHON) -m flask --app src.app:app run --host 127.0.0.1 --port 8000
test: train
	$(PYTHON) -m pytest -q
build:
	docker compose build
up:
	docker compose up -d --build
down:
	docker compose down
request:
	$(PYTHON) test_request.py
