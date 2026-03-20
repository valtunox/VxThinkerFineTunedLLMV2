"""
Structured logging helpers for VaLLM Specialist Model.
Provides performance_logger and ai_logger used by embedding and ML services.
"""

import logging
import time
from typing import Any, Dict, Optional


class PerformanceLogger:
    """Logs performance metrics (durations, throughput)."""

    def __init__(self):
        self.logger = logging.getLogger("vallm.performance")

    def log_duration(self, operation: str, duration_ms: float, **kwargs):
        self.logger.info(f"[PERF] {operation}: {duration_ms:.1f}ms {kwargs if kwargs else ''}")

    def log_throughput(self, operation: str, count: int, duration_s: float, **kwargs):
        rate = count / duration_s if duration_s > 0 else 0
        self.logger.info(f"[PERF] {operation}: {count} items in {duration_s:.1f}s ({rate:.1f}/s)")


class AILogger:
    """Logs AI/ML operations (embeddings, scoring, predictions)."""

    def __init__(self):
        self.logger = logging.getLogger("vallm.ai")

    def log_embedding_generation(self, **kwargs):
        self.logger.info(f"[AI] Embedding generation: {kwargs}")

    def log_search(self, query: str, results_count: int, duration_ms: float, **kwargs):
        self.logger.info(f"[AI] Search: q={query[:60]}... results={results_count} {duration_ms:.1f}ms")

    def log_prediction(self, model: str, **kwargs):
        self.logger.info(f"[AI] Prediction: model={model} {kwargs}")

    def log_scoring(self, **kwargs):
        self.logger.info(f"[AI] Scoring: {kwargs}")


performance_logger = PerformanceLogger()
ai_logger = AILogger()
