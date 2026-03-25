"""
VaLLM Specialist Model — Customer Service Dataset Tests
=========================================================

Author: Joel Otepa Wembo
https://joelwembo.com

DESCRIPTION
===========
Test suite for customer service datasets: FAQ, support tickets,
and escalation rules. Validates CSV structure, row counts,
search relevance, sentiment distribution, and escalation priorities.

No running server is required.

PREREQUISITES
=============
    pip install pytest pandas numpy

USAGE
=====
    python -m pytest app/tests/tests_customer_service.py -v -s --tb=short
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
CS_DIR = DATASETS_DIR / "customer_service"

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
    print("  CUSTOMER SERVICE TESTS — FINAL SCORECARD")
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

def _keyword_search(df: pd.DataFrame, keyword: str, columns=None) -> pd.DataFrame:
    if columns is None:
        columns = [c for c in df.columns if df[c].dtype == object]
    mask = pd.Series(False, index=df.index)
    for col in columns:
        mask |= df[col].astype(str).str.contains(keyword, case=False, na=False)
    return df[mask]


# ============================================================================
# 1. FILE EXISTENCE
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCustomerServiceFilesExist:
    """Verify customer service dataset files exist."""

    def test_directory_exists(self):
        assert CS_DIR.exists() and CS_DIR.is_dir()
        _record_score("CS Directory Exists", 10, str(CS_DIR))

    def test_faq_exists(self):
        p = CS_DIR / "faq.csv"
        assert p.exists(), f"faq.csv not found: {p}"
        _record_score("faq.csv Exists", 10, f"{p.stat().st_size / 1024:.1f} KB")

    def test_support_tickets_exists(self):
        p = CS_DIR / "support_tickets.csv"
        assert p.exists(), f"support_tickets.csv not found: {p}"
        _record_score("support_tickets.csv Exists", 10, f"{p.stat().st_size / 1024:.1f} KB")

    def test_escalation_rules_exists(self):
        p = CS_DIR / "escalation_rules.csv"
        assert p.exists(), f"escalation_rules.csv not found: {p}"
        _record_score("escalation_rules.csv Exists", 10, f"{p.stat().st_size / 1024:.1f} KB")


# ============================================================================
# 2. COLUMN STRUCTURE
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCustomerServiceColumns:
    """Verify correct columns per file."""

    def test_faq_columns(self):
        df = pd.read_csv(CS_DIR / "faq.csv", nrows=5)
        required = {"question", "answer", "category", "keywords"}
        actual = set(df.columns.str.lower().str.strip())
        missing = required - actual
        assert not missing, f"faq.csv missing columns: {missing}. Has: {list(df.columns)}"
        _record_score("FAQ Columns", 10, f"columns: {list(df.columns)}")

    def test_support_tickets_columns(self):
        df = pd.read_csv(CS_DIR / "support_tickets.csv", nrows=5)
        required = {"ticket_id", "customer_query", "response", "category", "sentiment"}
        actual = set(df.columns.str.lower().str.strip())
        missing = required - actual
        assert not missing, f"support_tickets.csv missing columns: {missing}. Has: {list(df.columns)}"
        _record_score("Support Tickets Columns", 10, f"columns: {list(df.columns)}")

    def test_escalation_rules_columns(self):
        df = pd.read_csv(CS_DIR / "escalation_rules.csv", nrows=5)
        required = {"trigger_condition", "priority", "escalation_target", "sla_hours", "category"}
        actual = set(df.columns.str.lower().str.strip())
        missing = required - actual
        assert not missing, f"escalation_rules.csv missing columns: {missing}. Has: {list(df.columns)}"
        _record_score("Escalation Rules Columns", 10, f"columns: {list(df.columns)}")


# ============================================================================
# 3. ROW COUNTS
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCustomerServiceRowCounts:
    """Verify datasets have expected row counts."""

    def test_faq_row_count(self):
        df = pd.read_csv(CS_DIR / "faq.csv")
        count = len(df)
        print(f"  faq.csv: {count:,} rows")
        assert count >= 100, f"Expected >= 100 FAQ entries, got {count}"
        score = 10 if count >= 250 else 8 if count >= 150 else 6
        _record_score("FAQ Row Count", score, f"{count:,} rows")

    def test_support_tickets_row_count(self):
        # Large file — only count lines to avoid reading entire file into memory
        path = CS_DIR / "support_tickets.csv"
        # Read a small sample to confirm it loads, then count
        sample = pd.read_csv(path, nrows=100)
        assert len(sample) == 100, "Failed to read 100-row sample"
        # Estimate total from file size or use wc-style count
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()
            line_count = sum(1 for _ in f)
        print(f"  support_tickets.csv: ~{line_count:,} data rows")
        assert line_count > 100_000, f"Expected > 100K rows, got {line_count:,}"
        score = 10 if line_count > 500_000 else 8 if line_count > 200_000 else 6
        _record_score("Support Tickets Row Count", score, f"~{line_count:,} rows")

    def test_escalation_rules_row_count(self):
        df = pd.read_csv(CS_DIR / "escalation_rules.csv")
        count = len(df)
        print(f"  escalation_rules.csv: {count:,} rows")
        assert count >= 50, f"Expected >= 50 escalation rules, got {count}"
        score = 10 if count >= 150 else 8 if count >= 100 else 6
        _record_score("Escalation Rules Row Count", score, f"{count:,} rows")


# ============================================================================
# 4. SEARCH QUERIES
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCustomerServiceSearch:
    """Test keyword search relevance across CS datasets."""

    def test_password_reset_search(self):
        df = pd.read_csv(CS_DIR / "faq.csv")
        results = _keyword_search(df, "password")
        count = len(results)
        print(f"  'password' in faq: {count} matches")
        assert count > 0, "No FAQ entries for password"
        _record_score("Password Reset Search", 10 if count >= 2 else 8, f"{count} matches")

    def test_billing_dispute_search(self):
        df = pd.read_csv(CS_DIR / "faq.csv")
        results = _keyword_search(df, "billing")
        count = len(results)
        print(f"  'billing' in faq: {count} matches")
        assert count > 0, "No FAQ entries for billing"
        _record_score("Billing Dispute Search", 10 if count >= 2 else 8, f"{count} matches")

    def test_account_locked_search(self):
        df = pd.read_csv(CS_DIR / "faq.csv")
        results = _keyword_search(df, "account")
        count = len(results)
        print(f"  'account' in faq: {count} matches")
        assert count > 0, "No FAQ entries for account"
        _record_score("Account Locked Search", 10 if count >= 2 else 8, f"{count} matches")

    def test_refund_search(self):
        df = pd.read_csv(CS_DIR / "faq.csv")
        results = _keyword_search(df, "refund")
        count = len(results)
        print(f"  'refund' in faq: {count} matches")
        assert count > 0, "No FAQ entries for refund"
        _record_score("Refund Request Search", 10 if count >= 1 else 6, f"{count} matches")


# ============================================================================
# 5. SENTIMENT DISTRIBUTION
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestSentimentDistribution:
    """Verify support tickets have diverse sentiments."""

    def test_sentiment_values(self):
        """Support tickets should have identifiable sentiment labels."""
        df = pd.read_csv(CS_DIR / "support_tickets.csv", nrows=10_000)
        sentiments = df["sentiment"].dropna().unique()
        print(f"  Sentiments found: {sorted(sentiments)}")
        assert len(sentiments) >= 2, f"Expected >= 2 sentiment types, got {len(sentiments)}"
        score = 10 if len(sentiments) >= 3 else 8
        _record_score("Sentiment Diversity", score, f"{len(sentiments)} types: {sorted(sentiments)}")

    def test_sentiment_not_single_dominant(self):
        """No single sentiment should dominate > 80% of tickets."""
        df = pd.read_csv(CS_DIR / "support_tickets.csv", nrows=10_000)
        dist = df["sentiment"].value_counts(normalize=True)
        max_pct = dist.iloc[0]
        top_sent = dist.index[0]
        print(f"  Most common: {top_sent} ({max_pct:.1%})")
        for sent, pct in dist.items():
            print(f"    {sent}: {pct:.1%}")
        assert max_pct < 0.80, f"Sentiment '{top_sent}' dominates: {max_pct:.1%}"
        score = 10 if max_pct < 0.50 else 8 if max_pct < 0.65 else 6
        _record_score("Sentiment Balance", score, f"top={top_sent} at {max_pct:.1%}")


# ============================================================================
# 6. ESCALATION PRIORITIES
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestEscalationPriorities:
    """Verify escalation rules cover expected priority levels."""

    def test_priority_levels(self):
        df = pd.read_csv(CS_DIR / "escalation_rules.csv")
        priorities = df["priority"].dropna().str.lower().unique()
        print(f"  Priority levels: {sorted(priorities)}")
        assert "critical" in priorities, "Missing 'critical' priority"
        assert len(priorities) >= 2, f"Expected >= 2 priority levels, got {len(priorities)}"
        score = 10 if len(priorities) >= 3 else 8
        _record_score("Escalation Priority Levels", score, f"{sorted(priorities)}")

    def test_sla_hours_valid(self):
        """SLA hours should be positive numbers."""
        df = pd.read_csv(CS_DIR / "escalation_rules.csv")
        sla = pd.to_numeric(df["sla_hours"], errors="coerce")
        invalid = sla.isna().sum()
        all_positive = (sla.dropna() > 0).all()
        print(f"  SLA range: {sla.min():.1f} - {sla.max():.1f} hours")
        print(f"  Invalid SLA entries: {invalid}")
        assert all_positive, "Some SLA hours are not positive"
        score = 10 if invalid == 0 else 8
        _record_score("SLA Hours Valid", score, f"range {sla.min():.1f}-{sla.max():.1f}h, {invalid} invalid")


# ============================================================================
# 7. FAQ KEYWORD COVERAGE
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestFAQKeywordCoverage:
    """Verify FAQ entries have keyword metadata."""

    def test_faq_keywords_populated(self):
        """Most FAQ entries should have keywords."""
        df = pd.read_csv(CS_DIR / "faq.csv")
        has_kw = df["keywords"].dropna().astype(str).str.strip().ne("").sum()
        pct = has_kw / len(df)
        print(f"  FAQs with keywords: {has_kw}/{len(df)} ({pct:.1%})")
        assert pct > 0.80, f"Only {pct:.1%} have keywords"
        score = 10 if pct > 0.95 else 8 if pct > 0.90 else 6
        _record_score("FAQ Keyword Coverage", score, f"{pct:.1%} populated")

    def test_faq_categories_diverse(self):
        """FAQ should cover multiple categories."""
        df = pd.read_csv(CS_DIR / "faq.csv")
        cats = df["category"].dropna().unique()
        print(f"  FAQ categories ({len(cats)}): {sorted(cats)[:10]}")
        assert len(cats) >= 3, f"Expected >= 3 categories, got {len(cats)}"
        score = 10 if len(cats) >= 8 else 8 if len(cats) >= 5 else 6
        _record_score("FAQ Category Diversity", score, f"{len(cats)} categories")


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VaLLM Specialist Model — Customer Service Dataset Tests")
    print("=" * 70)
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
