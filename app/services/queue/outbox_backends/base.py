"""
Generic outbox backend interface.
=================================

The queue service uses this protocol to poll, lock, and update outbox entries.
Domain-specific backends (notifications, email, campaigns, resumes, etc.) implement it
for their outbox table. Handlers are registered per queue/event_type and
contain only domain logic (e.g. send email, broadcast WebSocket).
"""

from typing import Any, Awaitable, Callable, Dict, List, Protocol


# Entry shape returned by get_pending / get_failed_retry (backend-defined)
OutboxEntry = Dict[str, Any]


# Handler: (payload, outbox_entry) -> success. Domain-only logic (e.g. send email, broadcast).
OutboxHandler = Callable[[Dict[str, Any], OutboxEntry], Awaitable[bool]]


class OutboxBackend(Protocol):
    """Protocol for an outbox table backend. Implement per domain (notification, email, etc.)."""

    @property
    def queue_name(self) -> str:
        """Identifier for this outbox (e.g. 'notification', 'email')."""
        ...

    async def get_pending_entries(
        self,
        limit: int = 10,
        priority_min: int = 1,
    ) -> List[OutboxEntry]:
        """Return pending entries ready for processing (e.g. status=PENDING, scheduled_at <= now)."""
        ...

    async def mark_processing(self, outbox_id: Any) -> bool:
        """Mark entry as processing. Return True if updated."""
        ...

    async def mark_completed(self, outbox_id: Any) -> bool:
        """Mark entry as completed."""
        ...

    async def mark_failed(
        self,
        outbox_id: Any,
        error_message: str,
        error_code: str | None = None,
        retry: bool = True,
    ) -> bool:
        """Mark failed; schedule retry or move to dead letter. Return True if retry scheduled."""
        ...

    async def get_failed_retry_entries(self, limit: int = 10) -> List[OutboxEntry]:
        """Return failed entries ready for retry (e.g. status=FAILED, next_retry_at <= now)."""
        ...

    async def reset_to_pending(self, outbox_id: Any) -> bool:
        """Reset entry to PENDING for retry. Return True if updated."""
        ...
