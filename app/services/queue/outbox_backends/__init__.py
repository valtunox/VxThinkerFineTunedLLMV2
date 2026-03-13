"""
Outbox backends for the generic queue pipeline.
===============================================

Each backend implements the outbox table contract for a specific domain
(notifications, email, campaigns, resumes, etc.). The queue service owns the pipeline
(poll, retry, DLQ); domains provide the backend and the handler.

  - base: OutboxBackend protocol and handler type
  - notification_backend: notification_outbox table (used by notifications domain)
"""

from app.services.queue.outbox_backends.base import OutboxBackend, OutboxEntry, OutboxHandler
from app.services.queue.outbox_backends.notification_backend import NotificationOutboxBackend
from app.services.queue.outbox_backends.campaign_backend import CampaignOutboxBackend

__all__ = [
    "OutboxBackend",
    "OutboxEntry",
    "OutboxHandler",
    "NotificationOutboxBackend",
    "CampaignOutboxBackend",
]
