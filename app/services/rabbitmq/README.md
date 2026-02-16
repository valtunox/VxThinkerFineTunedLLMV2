# RabbitMQ Service

## Overview

`RabbitMQService` provides message queue operations for the backend, aligned with the existing Redis, Celery, and Kafka services. It uses **pika** for sync connection; FastAPI routes run blocking calls in an executor.

## Features

- Connect / disconnect to RabbitMQ broker
- Declare queues (idempotent)
- Publish messages (string or JSON)
- Queue info (message count when available)
- Health check

## Configuration

- `RABBITMQ_URL`: Connection URL (default: `amqp://guest:guest@localhost:5672/`)

## Install

```bash
pip install pika
```

## Example Usage

```python
from app.services.rabbitmq.rabbitmq_service import get_rabbitmq_service

service = get_rabbitmq_service()
service.connect()
service.declare_queue("notifications")
service.publish("notifications", {"event": "user.signup", "user_id": "123"})
service.health_check()
service.close()
```

## API Routes (mounted under `/api/v1`)

- `POST /rabbitmq/publish` – Publish message (body: `queue`, `body`, `durable`)
- `POST /rabbitmq/queue/declare` – Declare queue (body: `queue`, `durable`)
- `GET /rabbitmq/queue/{queue_name}` – Queue info
- `GET /rabbitmq/health` – Health check

## Production

For production, run a RabbitMQ broker (e.g. via Docker or managed service) and set `RABBITMQ_URL`. The service connects on first use and reconnects when the connection is closed.
