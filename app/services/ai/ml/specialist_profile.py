"""
Shared IT specialist profile for training, retrieval, and runtime guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


SPECIALIST_PROFILE_NAME = "it_specialist"
SPECIALIST_ALLOWED_DATASET_DIRS = (
    "cloud",
    "automation",
    "customer_service",
    "skills",
    "uploaded",
)
SPECIALIST_PRIMARY_DATASET = "cloud/cloud_deployments.csv"
SPECIALIST_DOCUMENT_SUFFIXES = (".pdf", ".docx", ".doc", ".txt", ".md", ".html")

SPECIALIST_SYSTEM_PROMPT = (
    "You are VaLLM IT Specialist. You only answer questions about IT, cloud "
    "infrastructure, DevOps, deployments, technical support, networking, "
    "systems administration, software troubleshooting, customer support for "
    "technology products, programming, databases, security, and computer skills. "
    "If a request is mainly about healthcare, investing, personal finance, or "
    "other non-IT domains, explain briefly that the model is intentionally "
    "specialized for IT work and ask the user to reframe the request in an IT context."
)

SPECIALIST_DECLINE_MESSAGE = (
    "This model is intentionally specialized for IT work. It can help with cloud, "
    "deployments, DevOps, infrastructure, technical support, programming, "
    "networking, security, and related computer topics, but it is not meant to "
    "answer healthcare or finance questions as a generalist model."
)

IT_DOMAIN_KEYWORDS = {
    "api",
    "application",
    "aws",
    "azure",
    "backup",
    "bug",
    "ci/cd",
    "cloud",
    "computer",
    "container",
    "database",
    "debug",
    "deployment",
    "devops",
    "docker",
    "endpoint",
    "git",
    "gcp",
    "helpdesk",
    "infrastructure",
    "incident",
    "integration",
    "k8s",
    "kubernetes",
    "linux",
    "monitoring",
    "network",
    "node",
    "observability",
    "pipeline",
    "programming",
    "python",
    "redis",
    "sre",
    "server",
    "software",
    "support ticket",
    "terraform",
    "troubleshooting",
    "vpn",
    "windows",
}

BLOCKED_FINANCE_KEYWORDS = {
    "accounting",
    "bond",
    "credit score",
    "deferred revenue",
    "dividend",
    "equity",
    "forex",
    "gaap",
    "hedge fund",
    "ifrs",
    "investment",
    "loan",
    "mortgage",
    "mutual fund",
    "portfolio",
    "revenue recognition",
    "retirement",
    "stock",
    "tax filing",
    "trading",
}

BLOCKED_HEALTHCARE_KEYWORDS = {
    "clinical",
    "diagnosis",
    "doctor",
    "dosage",
    "hospital",
    "medical",
    "medication",
    "nurse",
    "patient",
    "pharmacy",
    "prescription",
    "symptom",
    "treatment",
    "vaccine",
}


@dataclass(frozen=True)
class ScopeAssessment:
    allowed: bool
    reason: str
    matched_it_terms: tuple[str, ...] = ()
    matched_blocked_terms: tuple[str, ...] = ()
    blocked_domain: str | None = None


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _match_keywords(text: str, keywords: Iterable[str]) -> List[str]:
    normalized = _normalize_text(text)
    return sorted(term for term in keywords if term in normalized)


def assess_specialist_scope(query: str) -> ScopeAssessment:
    """
    Allow IT/computer topics and reject clearly non-IT healthcare/finance requests.

    Generic questions are allowed unless the request is clearly dominated by one of
    the blocked domains. This keeps the guardrail targeted instead of over-blocking.
    """

    normalized = _normalize_text(query)
    it_terms = _match_keywords(normalized, IT_DOMAIN_KEYWORDS)
    finance_terms = _match_keywords(normalized, BLOCKED_FINANCE_KEYWORDS)
    healthcare_terms = _match_keywords(normalized, BLOCKED_HEALTHCARE_KEYWORDS)

    if healthcare_terms and not it_terms:
        return ScopeAssessment(
            allowed=False,
            reason="Query looks healthcare-specific rather than IT-specific.",
            matched_it_terms=tuple(it_terms),
            matched_blocked_terms=tuple(healthcare_terms),
            blocked_domain="healthcare",
        )

    if finance_terms and not it_terms:
        return ScopeAssessment(
            allowed=False,
            reason="Query looks finance-specific rather than IT-specific.",
            matched_it_terms=tuple(it_terms),
            matched_blocked_terms=tuple(finance_terms),
            blocked_domain="finance",
        )

    return ScopeAssessment(
        allowed=True,
        reason="Query is within the IT specialist scope.",
        matched_it_terms=tuple(it_terms),
        matched_blocked_terms=tuple(finance_terms or healthcare_terms),
        blocked_domain=None,
    )


def allowed_dataset_group_names() -> tuple[str, ...]:
    return SPECIALIST_ALLOWED_DATASET_DIRS


def dataset_group_for_path(path: Path, dataset_root: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return None

    if not relative.parts:
        return None

    group = relative.parts[0].lower()
    return group if group in SPECIALIST_ALLOWED_DATASET_DIRS else None


def is_specialist_dataset_path(path: Path, dataset_root: Path) -> bool:
    return dataset_group_for_path(path, dataset_root) is not None


def filter_specialist_csv_paths(paths: Sequence[Path], dataset_root: Path) -> List[Path]:
    filtered = [path for path in paths if path.is_file() and path.suffix.lower() == ".csv" and is_specialist_dataset_path(path, dataset_root)]
    return sorted(filtered)


def filter_specialist_document_paths(paths: Sequence[Path], dataset_root: Path) -> List[Path]:
    filtered = [
        path
        for path in paths
        if path.is_file()
        and path.suffix.lower() in SPECIALIST_DOCUMENT_SUFFIXES
        and is_specialist_dataset_path(path, dataset_root)
    ]
    return sorted(filtered)


def specialist_source_label(group_name: str) -> str:
    labels = {
        "cloud": "Cloud and DevOps",
        "automation": "Automation and Workflows",
        "customer_service": "IT Customer Support",
        "skills": "IT Skills and Troubleshooting",
        "uploaded": "Uploaded IT Knowledge",
    }
    return labels.get(group_name, group_name.replace("_", " ").title())
