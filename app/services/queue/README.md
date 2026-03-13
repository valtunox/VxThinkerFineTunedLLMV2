# Queue Service

Generic queue and outbox pipeline: **Celery + Redis + Postgres**. Designed to fit any use case (notifications, email, campaigns, resumes, etc.), not only one domain. Other transports (e.g. Kafka) can be added later as alternative backends.

## Responsibilities

- **Queue service**: owns the pipeline (poll outbox, retry, dead letter), broker config (Celery/Redis), and generic outbox backends. It does **not** contain domain logic (e.g. “how to send an email”).
- **Notifications service**: notification-only logic. Writes to `notification_outbox`, provides the **handler** (send email, WebSocket, Redis) and the dead-letter callback. It does **not** own the polling loop or status machine.

## Architecture

1. **Outbox table** (e.g. `notification_outbox`): written by the domain (notifications) in the same transaction as the business record.
2. **Outbox backend** (in queue): implements the generic contract (get pending, mark processing/completed/failed, retries). One backend per outbox table/domain.
3. **Handler** (in domain): given a payload and outbox entry, performs the domain action (e.g. send email, broadcast). Registered with the queue by queue name.
4. **Processor** (in queue): loop or Celery task that calls the backend to get pending/retry entries and invokes the registered handler, then updates status.

So: **notifications** = write to outbox + handler; **queue** = generic pipeline + notification backend (and any future backends for email, campaigns, etc.).

## Current stack

- **Transport**: Celery + Redis + Postgres (outbox). No Kafka in the notification path.
- **Notification flow**: create notification → write to `notification_outbox` → queue processor polls → notification handler (email, WebSocket, Redis).

## Adding another queue type (e.g. email)

1. Define an outbox table (or reuse a generic one with a `queue_name` column).
2. Implement `OutboxBackend` in `app.services.queue.outbox_backends` (e.g. `email_backend.py`).
3. Implement a handler in the email (or relevant) domain that does the actual send.
4. At startup, `register_outbox("email", email_backend, email_handler)`.
5. The existing processor loop can process all registered queues, or you can add a dedicated Celery task that calls `process_batch_for_queue("email")`.

## Files

- `broker.py` – Celery/Redis config.
- `generic_outbox_processor.py` – Generic processor and handler registry.
- `outbox_backends/base.py` – Outbox backend protocol.
- `outbox_backends/notification_backend.py` – Notification outbox table backend.
- `outbox_processor.py` – In-process loop; registers notification queue and runs `process_notification_outbox_batch`.
- `queue_pipeline_service.py` – Notification pipeline API (produce via outbox, consumer = outbox processor); no Kafka.
- `tasks.py` – Celery tasks (e.g. `process_outbox_batch`).
- `scheduler.py` – Optional Beat schedule for periodic outbox runs.
