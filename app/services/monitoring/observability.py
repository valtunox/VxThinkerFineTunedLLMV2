"""OpenTelemetry observability integration."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from core.logger import logger


class ObservabilityService:
    def __init__(self) -> None:
        self.enabled: bool = False
        self.configured: bool = False
        self.error: Optional[str] = None
        self.otlp_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        self.service_name: str = os.getenv("OTEL_SERVICE_NAME", "messaging_notifications")

    def configure(self) -> bool:
        if self.configured:
            return self.enabled

        self.configured = True

        enabled = os.getenv("OTEL_ENABLED", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
        if not enabled:
            self.enabled = False
            return False

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

            resource = Resource.create({"service.name": self.service_name})

            tracer_provider = TracerProvider(resource=resource)
            span_exporter = OTLPSpanExporter(endpoint=f"{self.otlp_endpoint.rstrip('/')}/v1/traces")
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
            trace.set_tracer_provider(tracer_provider)

            metric_exporter = OTLPMetricExporter(endpoint=f"{self.otlp_endpoint.rstrip('/')}/v1/metrics")
            metric_reader = PeriodicExportingMetricReader(metric_exporter)
            metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))

            self.enabled = True
            return True
        except Exception as e:
            self.enabled = False
            self.error = str(e)
            logger.warning(f"⚠️  OpenTelemetry observability not configured: {e}")
            return False

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "otlp_endpoint": self.otlp_endpoint,
            "service_name": self.service_name,
            "error": self.error,
        }


observability_service = ObservabilityService()
observability_service.configure()
