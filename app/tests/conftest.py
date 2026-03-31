from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.ai.ml import llm_router as llm_router_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
DATASETS_DIR = APP_DIR / "data" / "datasets"
CLOUD_DIR = DATASETS_DIR / "cloud"
AUTOMATION_DIR = DATASETS_DIR / "automation"
CUSTOMER_SERVICE_DIR = DATASETS_DIR / "customer_service"
SKILLS_DIR = DATASETS_DIR / "skills"
UPLOADED_DIR = DATASETS_DIR / "uploaded"


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def load_csv_sample(path: Path, rows: int = 200) -> pd.DataFrame:
    return pd.read_csv(path, nrows=rows)


def dataframe_text(df: pd.DataFrame) -> str:
    return " ".join(df.astype(str).fillna("").agg(" ".join, axis=1).tolist()).lower()


def contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


@pytest.fixture
def router_client_factory(monkeypatch):
    def _build(
        *,
        generated_answer: str = "Use Terraform and verify IAM role permissions.",
        internal_results: List[Dict[str, Any]] | None = None,
        web_results: List[Dict[str, Any]] | None = None,
    ) -> TestClient:
        async def fake_internal(query: str, top_k: int):
            return internal_results or [
                {
                    "document": "AWS IAM access keys should be rotated every 90 days and monitored.",
                    "score": 0.82,
                    "metadata": {
                        "file_name": "cloud_security_qa.csv",
                        "source_label": "Cloud and DevOps",
                        "dataset_file": "cloud/cloud_security_qa.csv",
                    },
                }
            ]

        async def fake_web(query: str, use_web_search: bool, max_results: int):
            if not use_web_search:
                return []
            return web_results or [
                {
                    "title": "IAM security best practices",
                    "url": "https://example.com/iam-best-practices",
                    "snippet": "Rotate credentials, prefer short-lived credentials, and audit usage.",
                    "provider": "duckduckgo",
                    "rank": 1,
                }
            ]

        async def fake_generate(http_request, prompt: str):
            return generated_answer

        monkeypatch.setattr(llm_router_module, "_search_internal_knowledge", fake_internal)
        monkeypatch.setattr(llm_router_module, "_search_web", fake_web)
        monkeypatch.setattr(llm_router_module, "_generate_specialist_answer", fake_generate)

        app = FastAPI()
        app.state.model = None
        app.state.tokenizer = None
        app.include_router(llm_router_module.router, prefix="/api/models/v1")
        return TestClient(app)

    return _build
