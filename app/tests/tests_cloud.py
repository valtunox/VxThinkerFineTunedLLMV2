"""
VaLLM Specialist Model — Cloud / DevOps / SRE Dataset Tests
=============================================================

Author: Joel Otepa Wembo
https://joelwembo.com

DESCRIPTION
===========
Comprehensive test suite for cloud, DevOps, and SRE datasets.
Validates CSV structure, row counts, category distributions,
answer quality, and cross-dataset search relevance.

No running server is required — tests operate directly on the
CSV files in app/data/datasets/cloud/.

PREREQUISITES
=============
    pip install pytest pandas numpy

USAGE
=====
    python -m pytest app/tests/tests_cloud.py -v -s --tb=short
"""

import sys
import os
import atexit
from pathlib import Path
from typing import Any, Dict

import pytest
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP_DIR = _PROJECT_ROOT / "app"
for _p in [str(_PROJECT_ROOT), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATASETS_DIR = _APP_DIR / "data" / "datasets"
CLOUD_DIR = DATASETS_DIR / "cloud"

# All expected cloud CSV files
CLOUD_CSV_FILES = [
    "cloud_architecture_qa.csv",
    "cloud_automation_qa.csv",
    "cloud_billing_qa.csv",
    "cloud_certifications_qa.csv",
    "cloud_compliance_qa.csv",
    "cloud_costs_qa.csv",
    "cloud_deployments.csv",
    "cloud_interview_qa.csv",
    "cloud_migrations_qa.csv",
    "cloud_performance_qa.csv",
    "cloud_security_qa.csv",
    "cloud_troubleshooting_qa.csv",
    "devops_qa.csv",
    "networking_qa.csv",
    "sre_qa.csv",
]

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
_SCORECARD: Dict[str, Dict[str, Any]] = {}
_TEST_COUNTER = 0


def _record_score(test_name: str, score: int, details: str = ""):
    global _TEST_COUNTER
    _TEST_COUNTER += 1
    score = max(1, min(10, score))
    bar = "#" * score + "." * (10 - score)
    label = (
        "PERFECT" if score == 10 else
        "EXCELLENT" if score >= 8 else
        "GOOD" if score >= 6 else
        "FAIR" if score >= 4 else
        "POOR"
    )
    _SCORECARD[f"Test {_TEST_COUNTER}"] = {
        "name": test_name, "score": score, "label": label, "details": details,
    }
    print(f"\n  {'=' * 65}")
    print(f"  [Test {_TEST_COUNTER}] {test_name}")
    print(f"  Score: {score}/10 [{bar}] {label}")
    if details:
        print(f"  Details: {details}")
    print(f"  {'=' * 65}")


def _print_final_scorecard():
    if not _SCORECARD:
        return
    total = sum(v["score"] for v in _SCORECARD.values())
    count = len(_SCORECARD)
    avg = total / count if count else 0
    print("\n\n" + "=" * 70)
    print("  CLOUD / DEVOPS / SRE TESTS — FINAL SCORECARD")
    print("=" * 70)
    for key, val in _SCORECARD.items():
        bar = "#" * val["score"] + "." * (10 - val["score"])
        print(f"  {key:>8} | {val['score']:>2}/10 [{bar}] {val['label']:<10} | {val['name']}")
    print("-" * 70)
    print(f"  {'TOTAL':>8} | {total}/{count * 10}  Average: {avg:.1f}/10")
    overall = (
        "EXCELLENT" if avg >= 8 else "GOOD" if avg >= 6 else
        "NEEDS IMPROVEMENT" if avg >= 4 else "CRITICAL ISSUES"
    )
    print(f"  Overall Assessment: {overall}")
    print("=" * 70)


atexit.register(_print_final_scorecard)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_csv(name: str) -> pd.DataFrame:
    """Load a cloud CSV into a DataFrame."""
    path = CLOUD_DIR / name
    return pd.read_csv(path, on_bad_lines="warn")


def _keyword_search(df: pd.DataFrame, keyword: str, columns=None) -> pd.DataFrame:
    """Simple case-insensitive keyword search across specified columns."""
    if columns is None:
        columns = [c for c in df.columns if df[c].dtype == object]
    mask = pd.Series(False, index=df.index)
    for col in columns:
        mask |= df[col].astype(str).str.contains(keyword, case=False, na=False)
    return df[mask]


# ============================================================================
# 1. CLOUD DIRECTORY & FILE EXISTENCE
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCloudFilesExist:
    """Verify cloud dataset directory and all expected CSV files exist."""

    def test_cloud_directory_exists(self):
        """Cloud dataset directory should exist."""
        assert CLOUD_DIR.exists(), f"Cloud directory not found: {CLOUD_DIR}"
        assert CLOUD_DIR.is_dir()
        _record_score("Cloud Directory Exists", 10, str(CLOUD_DIR))

    def test_all_csv_files_exist(self):
        """All 15 expected cloud CSV files should be present."""
        missing = [f for f in CLOUD_CSV_FILES if not (CLOUD_DIR / f).exists()]
        found = len(CLOUD_CSV_FILES) - len(missing)
        assert not missing, f"Missing files: {missing}"
        _record_score("All Cloud CSVs Present", 10, f"{found}/{len(CLOUD_CSV_FILES)} files")

    def test_csv_files_not_empty(self):
        """Every cloud CSV should be non-empty (> 0 bytes)."""
        empty_files = []
        for f in CLOUD_CSV_FILES:
            p = CLOUD_DIR / f
            if p.exists() and p.stat().st_size == 0:
                empty_files.append(f)
        assert not empty_files, f"Empty files: {empty_files}"
        _record_score("Cloud CSVs Non-Empty", 10, f"All {len(CLOUD_CSV_FILES)} files have data")


# ============================================================================
# 2. COLUMN STRUCTURE
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCloudColumnStructure:
    """Verify CSV column structure matches expected schema."""

    @pytest.mark.parametrize("csv_name", [
        "devops_qa.csv", "sre_qa.csv", "networking_qa.csv",
        "cloud_architecture_qa.csv", "cloud_automation_qa.csv",
        "cloud_billing_qa.csv", "cloud_compliance_qa.csv",
        "cloud_costs_qa.csv", "cloud_security_qa.csv",
        "cloud_troubleshooting_qa.csv", "cloud_deployments.csv",
        "cloud_interview_qa.csv", "cloud_migrations_qa.csv",
        "cloud_performance_qa.csv", "cloud_certifications_qa.csv",
    ])
    def test_qa_columns(self, csv_name):
        """QA files should have question, answer, and category columns."""
        df = _load_csv(csv_name)
        required = {"question", "answer", "category"}
        actual = set(df.columns.str.lower().str.strip())
        missing = required - actual
        assert not missing, f"{csv_name} missing columns: {missing}. Has: {list(df.columns)}"

    def test_column_report(self):
        """Report columns for all cloud CSVs."""
        report_lines = []
        for f in CLOUD_CSV_FILES:
            df = _load_csv(f)
            report_lines.append(f"  {f}: {list(df.columns)}")
        report = "\n".join(report_lines)
        print(f"\n  Column report:\n{report}")
        _record_score("Cloud Column Structure", 10, f"All {len(CLOUD_CSV_FILES)} files checked")


# ============================================================================
# 3. ROW COUNTS
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCloudRowCounts:
    """Verify datasets have substantial row counts."""

    def test_devops_qa_row_count(self):
        """devops_qa.csv should have > 50K rows."""
        df = _load_csv("devops_qa.csv")
        count = len(df)
        print(f"  devops_qa.csv: {count:,} rows")
        assert count > 50_000, f"Expected > 50K rows, got {count:,}"
        score = 10 if count > 70_000 else 8 if count > 60_000 else 6
        _record_score("devops_qa Row Count", score, f"{count:,} rows")

    def test_sre_qa_row_count(self):
        """sre_qa.csv should have > 30K rows."""
        df = _load_csv("sre_qa.csv")
        count = len(df)
        print(f"  sre_qa.csv: {count:,} rows")
        assert count > 30_000, f"Expected > 30K rows, got {count:,}"
        score = 10 if count > 50_000 else 8 if count > 40_000 else 6
        _record_score("sre_qa Row Count", score, f"{count:,} rows")

    def test_networking_qa_row_count(self):
        """networking_qa.csv should have > 20K rows."""
        df = _load_csv("networking_qa.csv")
        count = len(df)
        print(f"  networking_qa.csv: {count:,} rows")
        assert count > 20_000, f"Expected > 20K rows, got {count:,}"
        score = 10 if count > 25_000 else 8
        _record_score("networking_qa Row Count", score, f"{count:,} rows")

    def test_cloud_billing_qa_row_count(self):
        """cloud_billing_qa.csv should have > 15K rows."""
        df = _load_csv("cloud_billing_qa.csv")
        count = len(df)
        print(f"  cloud_billing_qa.csv: {count:,} rows")
        assert count > 15_000, f"Expected > 15K rows, got {count:,}"
        score = 10 if count > 20_000 else 8
        _record_score("cloud_billing_qa Row Count", score, f"{count:,} rows")

    def test_total_cloud_rows(self):
        """Total rows across all cloud CSVs should be > 200K."""
        total = 0
        for f in CLOUD_CSV_FILES:
            df = _load_csv(f)
            total += len(df)
        print(f"  Total cloud rows: {total:,}")
        assert total > 200_000, f"Expected > 200K total, got {total:,}"
        score = 10 if total > 250_000 else 8
        _record_score("Total Cloud Rows", score, f"{total:,} rows across {len(CLOUD_CSV_FILES)} files")


# ============================================================================
# 4. DOMAIN-SPECIFIC SEARCH QUERIES
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCloudDomainSearch:
    """Test keyword search relevance across cloud domains."""

    def test_kubernetes_search(self):
        """Search for Kubernetes-related Q&A."""
        df = _load_csv("devops_qa.csv")
        results = _keyword_search(df, "kubernetes")
        count = len(results)
        print(f"  Kubernetes results in devops_qa: {count}")
        assert count > 0, "No Kubernetes results found"
        score = 10 if count > 500 else 8 if count > 100 else 6
        _record_score("Kubernetes Search", score, f"{count} matches")

    def test_terraform_search(self):
        """Search for Terraform IaC content."""
        df = _load_csv("devops_qa.csv")
        results = _keyword_search(df, "terraform")
        count = len(results)
        print(f"  Terraform results: {count}")
        assert count > 0, "No Terraform results found"
        score = 10 if count > 200 else 8 if count > 50 else 6
        _record_score("Terraform Search", score, f"{count} matches")

    def test_cicd_search(self):
        """Search for CI/CD pipeline content."""
        df = _load_csv("devops_qa.csv")
        results_ci = _keyword_search(df, "ci/cd")
        results_pipeline = _keyword_search(df, "pipeline")
        combined = len(results_ci) + len(results_pipeline)
        print(f"  CI/CD results: {len(results_ci)}, pipeline results: {len(results_pipeline)}")
        assert combined > 0, "No CI/CD or pipeline results found"
        score = 10 if combined > 200 else 8 if combined > 50 else 6
        _record_score("CI/CD Search", score, f"{combined} matches")

    def test_monitoring_search(self):
        """Search for monitoring/observability content."""
        df = _load_csv("sre_qa.csv")
        results = _keyword_search(df, "monitoring")
        count = len(results)
        print(f"  Monitoring results in sre_qa: {count}")
        assert count > 0, "No monitoring results found"
        score = 10 if count > 100 else 8 if count > 30 else 6
        _record_score("Monitoring Search", score, f"{count} matches")

    def test_networking_search(self):
        """Search for networking content."""
        df = _load_csv("networking_qa.csv")
        results = _keyword_search(df, "TCP")
        count = len(results)
        print(f"  TCP results in networking_qa: {count}")
        assert count > 0, "No TCP results found"
        score = 10 if count > 100 else 8 if count > 20 else 6
        _record_score("Networking Search", score, f"{count} matches")

    def test_sre_error_budget_search(self):
        """Search for SRE error budget content."""
        df = _load_csv("sre_qa.csv")
        results = _keyword_search(df, "error budget")
        count = len(results)
        print(f"  Error budget results in sre_qa: {count}")
        assert count > 0, "No error budget results found"
        score = 10 if count > 100 else 8 if count > 20 else 6
        _record_score("SRE Error Budget Search", score, f"{count} matches")

    def test_billing_costs_search(self):
        """Search for billing and cost optimization."""
        df = _load_csv("cloud_billing_qa.csv")
        results = _keyword_search(df, "cost")
        count = len(results)
        print(f"  Cost results in cloud_billing_qa: {count}")
        assert count > 0, "No cost results found"
        score = 10 if count > 200 else 8 if count > 50 else 6
        _record_score("Billing/Costs Search", score, f"{count} matches")


# ============================================================================
# 5. CATEGORY DISTRIBUTION
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCloudCategoryDistribution:
    """Test that datasets have diverse, well-distributed categories."""

    def test_devops_categories(self):
        """devops_qa should have multiple categories (kubernetes, docker, terraform, etc.)."""
        df = _load_csv("devops_qa.csv")
        cats = df["category"].dropna().unique()
        print(f"  devops_qa categories ({len(cats)}): {sorted(cats)[:15]}...")
        assert len(cats) >= 3, f"Expected >= 3 categories, got {len(cats)}"
        score = 10 if len(cats) >= 8 else 8 if len(cats) >= 5 else 6
        _record_score("DevOps Category Count", score, f"{len(cats)} categories")

    def test_sre_categories(self):
        """sre_qa should cover SRE sub-domains."""
        df = _load_csv("sre_qa.csv")
        cats = df["category"].dropna().unique()
        print(f"  sre_qa categories ({len(cats)}): {sorted(cats)[:15]}...")
        assert len(cats) >= 2, f"Expected >= 2 categories, got {len(cats)}"
        score = 10 if len(cats) >= 5 else 8 if len(cats) >= 3 else 6
        _record_score("SRE Category Count", score, f"{len(cats)} categories")

    def test_no_dominant_single_category(self):
        """No single category should represent > 80% of devops_qa."""
        df = _load_csv("devops_qa.csv")
        dist = df["category"].value_counts(normalize=True)
        max_pct = dist.iloc[0]
        top_cat = dist.index[0]
        print(f"  Most common category: {top_cat} ({max_pct:.1%})")
        assert max_pct < 0.80, f"Category '{top_cat}' dominates with {max_pct:.1%}"
        score = 10 if max_pct < 0.30 else 8 if max_pct < 0.50 else 6
        _record_score("Category Balance", score, f"top={top_cat} at {max_pct:.1%}")


# ============================================================================
# 6. ANSWER QUALITY
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCloudAnswerQuality:
    """Verify answers are substantive and non-empty."""

    def test_no_empty_answers(self):
        """Answers should not be empty or NaN."""
        df = _load_csv("devops_qa.csv")
        empty = df["answer"].isna().sum() + (df["answer"].astype(str).str.strip() == "").sum()
        pct_valid = 1 - empty / len(df)
        print(f"  devops_qa: {empty} empty answers out of {len(df)} ({pct_valid:.1%} valid)")
        assert pct_valid > 0.95, f"Too many empty answers: {empty}"
        score = 10 if pct_valid > 0.99 else 8 if pct_valid > 0.97 else 6
        _record_score("Non-Empty Answers", score, f"{pct_valid:.1%} valid")

    def test_answer_length_reasonable(self):
        """Average answer length should be > 50 characters."""
        df = _load_csv("devops_qa.csv")
        avg_len = df["answer"].astype(str).str.len().mean()
        print(f"  devops_qa average answer length: {avg_len:.0f} chars")
        assert avg_len > 50, f"Average answer too short: {avg_len:.0f} chars"
        score = 10 if avg_len > 150 else 8 if avg_len > 100 else 6
        _record_score("Answer Length", score, f"avg={avg_len:.0f} chars")


# ============================================================================
# 7. CROSS-DATASET SEARCH
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCloudCrossDatasetSearch:
    """Test that queries find results across multiple cloud files."""

    def test_cross_dataset_deploy_query(self):
        """'deploy' should appear across deployments, devops, and automation files."""
        files_with_hits = []
        for csv_name in CLOUD_CSV_FILES:
            df = _load_csv(csv_name)
            hits = _keyword_search(df, "deploy")
            if len(hits) > 0:
                files_with_hits.append((csv_name, len(hits)))
                print(f"  {csv_name}: {len(hits)} hits for 'deploy'")
        assert len(files_with_hits) >= 2, f"Expected hits in >= 2 files, got {len(files_with_hits)}"
        score = 10 if len(files_with_hits) >= 5 else 8 if len(files_with_hits) >= 3 else 6
        _record_score("Cross-Dataset Deploy Query", score,
                       f"hits in {len(files_with_hits)} files")

    def test_cross_dataset_security_query(self):
        """'security' should appear across multiple cloud files."""
        files_with_hits = []
        for csv_name in CLOUD_CSV_FILES:
            df = _load_csv(csv_name)
            hits = _keyword_search(df, "security")
            if len(hits) > 0:
                files_with_hits.append((csv_name, len(hits)))
        print(f"  'security' found in {len(files_with_hits)} files")
        assert len(files_with_hits) >= 2, f"Expected hits in >= 2 files, got {len(files_with_hits)}"
        score = 10 if len(files_with_hits) >= 5 else 8 if len(files_with_hits) >= 3 else 6
        _record_score("Cross-Dataset Security Query", score,
                       f"hits in {len(files_with_hits)} files")


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VaLLM Specialist Model — Cloud / DevOps / SRE Dataset Tests")
    print("=" * 70)
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
