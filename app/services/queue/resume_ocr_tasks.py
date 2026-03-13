"""
Document OCR Background Tasks
===============================

Celery tasks for asynchronous document OCR processing with Kafka event streaming.

Pipeline:
  1. Document uploaded -> Kafka event ``document.uploaded``
  2. This task receives (file_path, document_id, organization_id, batch_id)
  3. Text extraction: pdfplumber -> pypdf -> Tesseract OCR fallback
  4. Extracted text stored in document record
  5. Kafka event ``document.processed`` published on success / ``document.failed`` on failure
  6. Redis progress updated per file

Designed for production: retries, timeouts, fallback chain, structured error handling.
"""

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from app.core.celery_app import celery_app, CELERY_AVAILABLE
except ImportError:
    celery_app = None
    CELERY_AVAILABLE = False

_thread_local = threading.local()


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    loop = getattr(_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _thread_local.loop = loop
    return loop


# ---------------------------------------------------------------------------
# Text extraction with 3-tier fallback: pdfplumber -> pypdf -> Tesseract OCR
# ---------------------------------------------------------------------------

def _extract_with_pdfplumber(file_path: str) -> Optional[str]:
    try:
        import pdfplumber
        texts: List[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
        combined = "\n\n".join(texts).strip()
        return combined if len(combined) >= 50 else None
    except Exception as exc:
        logger.debug("pdfplumber extraction failed: %s", exc)
        return None


def _extract_with_pypdf(file_path: str) -> Optional[str]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        texts: List[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
        combined = "\n\n".join(texts).strip()
        return combined if len(combined) >= 50 else None
    except Exception as exc:
        logger.debug("pypdf extraction failed: %s", exc)
        return None


def _extract_with_ocr(file_path: str) -> Dict[str, Any]:
    """Tesseract OCR fallback."""
    try:
        from app.services.ai.ml.ocr import extract_text
        result = extract_text(file_path)
        return {
            "text": result.text if result.success else None,
            "confidence": result.confidence,
            "page_count": result.page_count,
            "warnings": result.warnings,
            "success": result.success,
            "processing_time_sec": result.processing_time_sec,
            "extraction_method": "tesseract_ocr",
        }
    except Exception as exc:
        logger.warning("OCR extraction failed: %s", exc)
        return {
            "text": None, "confidence": 0.0, "page_count": 0,
            "warnings": [str(exc)], "success": False,
            "processing_time_sec": 0.0, "extraction_method": "tesseract_ocr",
        }


def extract_document_text(file_path: str) -> Dict[str, Any]:
    """
    3-tier fallback text extraction:
      1. pdfplumber (fast, structured PDFs)
      2. pypdf (fallback for different PDF encodings)
      3. Tesseract OCR (scanned documents / images)
    """
    text = _extract_with_pdfplumber(file_path)
    if text:
        return {
            "text": text, "confidence": 0.95, "page_count": 0,
            "warnings": [], "success": True, "extraction_method": "pdfplumber",
        }

    text = _extract_with_pypdf(file_path)
    if text:
        return {
            "text": text, "confidence": 0.85, "page_count": 0,
            "warnings": ["pdfplumber failed; used pypdf fallback"],
            "success": True, "extraction_method": "pypdf",
        }

    ocr_result = _extract_with_ocr(file_path)
    if ocr_result.get("text"):
        return ocr_result

    return {
        "text": None, "confidence": 0.0, "page_count": 0,
        "warnings": (ocr_result.get("warnings") or []) + ["All extraction methods failed"],
        "success": False, "extraction_method": "none",
    }


# ---------------------------------------------------------------------------
# Document field parser (regex-based)
# ---------------------------------------------------------------------------

def _parse_document_fields(text: str) -> Dict[str, Any]:
    """Extract structured fields from document text using pattern matching."""
    import re

    parsed: Dict[str, Any] = {}

    # Amounts / currency
    amounts = re.findall(r'\$[\d,]+\.?\d*', text)
    if amounts:
        parsed["amounts_found"] = amounts

    # Dates
    dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
    if dates:
        parsed["dates_found"] = dates

    # Emails
    email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        parsed["email"] = email_match.group(0)

    # Phone
    phone_match = re.search(r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}", text)
    if phone_match:
        parsed["phone"] = phone_match.group(0).strip()

    # Document type classification
    lower_text = text.lower()
    if any(w in lower_text for w in ["invoice", "bill to", "amount due"]):
        parsed["document_type"] = "invoice"
    elif any(w in lower_text for w in ["receipt", "paid", "transaction"]):
        parsed["document_type"] = "receipt"
    elif any(w in lower_text for w in ["contract", "agreement", "hereby"]):
        parsed["document_type"] = "contract"
    elif any(w in lower_text for w in ["balance sheet", "income statement"]):
        parsed["document_type"] = "financial_statement"
    else:
        parsed["document_type"] = "general"

    return parsed


# ---------------------------------------------------------------------------
# Kafka event publishing
# ---------------------------------------------------------------------------

def _publish_kafka_event(event_type: str, file_id: str, metadata: Dict[str, Any]):
    """Publish a document event to Kafka (best-effort, non-blocking)."""
    try:
        from app.services.kafka.kafka_service import (
            kafka_service, KafkaEvent, KafkaEventType,
            KafkaResourceType, KafkaEventStatus,
        )
        if kafka_service is None:
            return

        type_map = {
            "document.processed": KafkaEventType.DOCUMENT_PROCESSED,
            "document.uploaded": KafkaEventType.DOCUMENT_UPLOADED,
        }
        event = KafkaEvent(
            event_id=str(uuid.uuid4()),
            resource_id=file_id,
            resource_type=KafkaResourceType.DOCUMENT,
            user_id="system",
            event_type=type_map.get(event_type, KafkaEventType.DOCUMENT_PROCESSED),
            status=KafkaEventStatus.COMPLETED if "processed" in event_type else KafkaEventStatus.FAILED,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata,
        )
        kafka_service.produce_event(event, ["document-events"])
    except Exception as exc:
        logger.warning("Kafka event publish failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Redis progress update
# ---------------------------------------------------------------------------

def _update_redis_progress(batch_id: str, file_id: str, status: str, detail: Optional[Dict[str, Any]] = None):
    """Update per-file progress in Redis (best-effort)."""
    try:
        from app.services.redis.redis_service import RedisService
        redis = RedisService()
        loop = _get_or_create_event_loop()

        async def _update():
            key = f"document_batch:{batch_id}"
            existing = await redis.get_key(key)
            if existing is None:
                existing = {}
            elif isinstance(existing, str):
                existing = json.loads(existing)
            existing[file_id] = {"status": status, **(detail or {})}
            await redis.set_key(key, json.dumps(existing))

        loop.run_until_complete(_update())
    except Exception as exc:
        logger.debug("Redis progress update failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Celery task definition
# ---------------------------------------------------------------------------

if CELERY_AVAILABLE and celery_app is not None:

    @celery_app.task(
        name="document.process_ocr",
        bind=True,
        max_retries=3,
        default_retry_delay=30,
        soft_time_limit=180,
        time_limit=300,
        queue="default",
        acks_late=True,
    )
    def process_document_ocr(
        self,
        file_path: str,
        document_id: str,
        organization_id: str,
        file_id: str,
        original_filename: str,
        batch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Celery task: extract text from a document PDF, parse fields,
        publish Kafka events, and update Redis progress.
        """
        logger.info(
            "[DOC-OCR] Celery task STARTED file_id=%s document=%s org=%s",
            file_id, document_id, organization_id,
        )

        if batch_id:
            _update_redis_progress(batch_id, file_id, "ocr_processing")

        try:
            if not Path(file_path).is_file():
                raise FileNotFoundError(f"Document file not found: {file_path}")

            extraction = extract_document_text(file_path)
            parsed_fields: Dict[str, Any] = {}
            if extraction.get("text"):
                parsed_fields = _parse_document_fields(extraction["text"])

            if extraction.get("success"):
                _publish_kafka_event("document.processed", file_id, {
                    "document_id": document_id,
                    "organization_id": organization_id,
                    "extraction_method": extraction.get("extraction_method"),
                    "confidence": extraction.get("confidence", 0.0),
                    "text_length": len(extraction.get("text") or ""),
                    "batch_id": batch_id,
                })
                if batch_id:
                    _update_redis_progress(batch_id, file_id, "ocr_completed")
            else:
                _publish_kafka_event("document.failed", file_id, {
                    "document_id": document_id,
                    "warnings": extraction.get("warnings", []),
                })
                if batch_id:
                    _update_redis_progress(batch_id, file_id, "ocr_failed")

            return {
                "file_id": file_id,
                "document_id": document_id,
                "success": extraction.get("success", False),
                "confidence": extraction.get("confidence", 0.0),
                "extraction_method": extraction.get("extraction_method", "none"),
                "text_length": len(extraction.get("text") or ""),
                "parsed_fields": list(parsed_fields.keys()),
            }

        except FileNotFoundError:
            if batch_id:
                _update_redis_progress(batch_id, file_id, "ocr_failed", {"error": "file_not_found"})
            raise
        except Exception as exc:
            logger.exception("OCR task failed for file %s: %s", file_id, exc)
            if batch_id:
                _update_redis_progress(batch_id, file_id, "ocr_retry")
            raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Synchronous fallback
# ---------------------------------------------------------------------------

def process_document_ocr_sync(
    file_path: str,
    document_id: str,
    organization_id: str,
    file_id: str,
    original_filename: str,
    batch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Synchronous fallback for OCR processing when Celery is not available."""
    if not Path(file_path).is_file():
        return {"file_id": file_id, "success": False, "error": "file_not_found"}

    extraction = extract_document_text(file_path)
    parsed_fields: Dict[str, Any] = {}
    if extraction.get("text"):
        parsed_fields = _parse_document_fields(extraction["text"])

    return {
        "file_id": file_id,
        "document_id": document_id,
        "success": extraction.get("success", False),
        "confidence": extraction.get("confidence", 0.0),
        "extraction_method": extraction.get("extraction_method", "none"),
        "text_length": len(extraction.get("text") or ""),
        "parsed_fields": list(parsed_fields.keys()),
    }


def dispatch_document_ocr(
    file_path: str,
    document_id: str,
    organization_id: str,
    file_id: str,
    original_filename: str,
    batch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Dispatch OCR: Celery task if available, otherwise sync fallback."""
    if CELERY_AVAILABLE and celery_app is not None:
        try:
            result = process_document_ocr.apply_async(
                kwargs={
                    "file_path": file_path,
                    "document_id": document_id,
                    "organization_id": organization_id,
                    "file_id": file_id,
                    "original_filename": original_filename,
                    "batch_id": batch_id,
                },
                queue="default",
            )
            return {"dispatch": "celery", "task_id": result.id, "file_id": file_id}
        except Exception as exc:
            logger.warning("Celery dispatch failed, falling back to sync: %s", exc)

    sync_result = process_document_ocr_sync(
        file_path=file_path,
        document_id=document_id,
        organization_id=organization_id,
        file_id=file_id,
        original_filename=original_filename,
        batch_id=batch_id,
    )
    sync_result["dispatch"] = "sync_fallback"
    return sync_result
