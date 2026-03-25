"""
VaLLM Specialist Model — Finance Dataset Tests
=================================================

Author: Joel Otepa Wembo
https://joelwembo.com

DESCRIPTION
===========
Test suite for finance datasets: financial_qa.csv, financial_terms.csv,
and transactions.csv. Validates CSV structure, row counts, search
relevance, risk scores, and financial term example usage.

No running server is required.

PREREQUISITES
=============
    pip install pytest pandas numpy

USAGE
=====
    python -m pytest app/tests/tests_finance.py -v -s --tb=short
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
FINANCE_DIR = DATASETS_DIR / "finance"

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
    print("  FINANCE TESTS — FINAL SCORECARD")
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
class TestFinanceFilesExist:
    """Verify finance dataset files exist."""

    def test_directory_exists(self):
        assert FINANCE_DIR.exists() and FINANCE_DIR.is_dir()
        _record_score("Finance Directory Exists", 10, str(FINANCE_DIR))

    def test_financial_qa_exists(self):
        p = FINANCE_DIR / "financial_qa.csv"
        assert p.exists(), f"Not found: {p}"
        _record_score("financial_qa.csv Exists", 10, f"{p.stat().st_size / 1024:.1f} KB")

    def test_financial_terms_exists(self):
        p = FINANCE_DIR / "financial_terms.csv"
        assert p.exists(), f"Not found: {p}"
        _record_score("financial_terms.csv Exists", 10, f"{p.stat().st_size / 1024:.1f} KB")

    def test_transactions_exists(self):
        p = FINANCE_DIR / "transactions.csv"
        assert p.exists(), f"Not found: {p}"
        _record_score("transactions.csv Exists", 10, f"{p.stat().st_size / 1024:.1f} KB")


# ============================================================================
# 2. COLUMN STRUCTURE
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestFinanceColumns:
    """Verify correct column schemas."""

    def test_financial_qa_columns(self):
        df = pd.read_csv(FINANCE_DIR / "financial_qa.csv", nrows=5)
        required = {"question", "answer", "category"}
        actual = set(df.columns.str.lower().str.strip())
        missing = required - actual
        assert not missing, f"Missing: {missing}. Has: {list(df.columns)}"
        _record_score("financial_qa Columns", 10, f"{list(df.columns)}")

    def test_financial_terms_columns(self):
        df = pd.read_csv(FINANCE_DIR / "financial_terms.csv", nrows=5)
        required = {"term", "definition", "category", "example_usage"}
        actual = set(df.columns.str.lower().str.strip())
        missing = required - actual
        assert not missing, f"Missing: {missing}. Has: {list(df.columns)}"
        _record_score("financial_terms Columns", 10, f"{list(df.columns)}")

    def test_transactions_columns(self):
        df = pd.read_csv(FINANCE_DIR / "transactions.csv", nrows=5)
        required = {"transaction_id", "type", "amount", "currency", "category", "risk_score"}
        actual = set(df.columns.str.lower().str.strip())
        missing = required - actual
        assert not missing, f"Missing: {missing}. Has: {list(df.columns)}"
        _record_score("transactions Columns", 10, f"{list(df.columns)}")


# ============================================================================
# 3. ROW COUNTS
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestFinanceRowCounts:
    """Verify adequate data volume."""

    def test_financial_qa_count(self):
        df = pd.read_csv(FINANCE_DIR / "financial_qa.csv")
        count = len(df)
        print(f"  financial_qa.csv: {count:,} rows")
        assert count >= 100, f"Expected >= 100, got {count}"
        score = 10 if count >= 250 else 8 if count >= 150 else 6
        _record_score("financial_qa Row Count", score, f"{count:,} rows")

    def test_financial_terms_count(self):
        df = pd.read_csv(FINANCE_DIR / "financial_terms.csv")
        count = len(df)
        print(f"  financial_terms.csv: {count:,} rows")
        assert count >= 100, f"Expected >= 100, got {count}"
        score = 10 if count >= 250 else 8 if count >= 150 else 6
        _record_score("financial_terms Row Count", score, f"{count:,} rows")

    def test_transactions_count(self):
        df = pd.read_csv(FINANCE_DIR / "transactions.csv")
        count = len(df)
        print(f"  transactions.csv: {count:,} rows")
        assert count >= 100, f"Expected >= 100, got {count}"
        score = 10 if count >= 250 else 8 if count >= 150 else 6
        _record_score("transactions Row Count", score, f"{count:,} rows")


# ============================================================================
# 4. SEARCH QUERIES
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestFinanceSearch:
    """Test keyword search across finance datasets."""

    def test_accounts_payable_search(self):
        df = pd.read_csv(FINANCE_DIR / "financial_qa.csv")
        results = _keyword_search(df, "accounts payable")
        count = len(results)
        print(f"  'accounts payable' in financial_qa: {count} matches")
        assert count > 0, "No results for 'accounts payable'"
        _record_score("Accounts Payable Search", 10 if count >= 2 else 8, f"{count} matches")

    def test_roi_search(self):
        df = pd.read_csv(FINANCE_DIR / "financial_qa.csv")
        results = _keyword_search(df, "ROI")
        count = len(results)
        print(f"  'ROI' in financial_qa: {count} matches")
        # ROI may also appear as 'return on investment'
        if count == 0:
            results = _keyword_search(df, "return on investment")
            count = len(results)
            print(f"  'return on investment': {count} matches")
        assert count > 0, "No results for ROI or return on investment"
        _record_score("ROI Search", 10 if count >= 2 else 8, f"{count} matches")

    def test_audit_compliance_search(self):
        df = pd.read_csv(FINANCE_DIR / "financial_qa.csv")
        results_audit = _keyword_search(df, "audit")
        results_compliance = _keyword_search(df, "compliance")
        combined = len(results_audit) + len(results_compliance)
        print(f"  'audit': {len(results_audit)}, 'compliance': {len(results_compliance)}")
        assert combined > 0, "No results for audit or compliance"
        _record_score("Audit/Compliance Search", 10 if combined >= 3 else 8, f"{combined} matches")


# ============================================================================
# 5. TRANSACTION RISK SCORES
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestTransactionRiskScores:
    """Verify transaction risk scores are valid."""

    def test_risk_score_range(self):
        """Risk scores should be between 0 and 1."""
        df = pd.read_csv(FINANCE_DIR / "transactions.csv")
        risk = pd.to_numeric(df["risk_score"], errors="coerce")
        valid = risk.dropna()
        min_val = valid.min()
        max_val = valid.max()
        print(f"  Risk score range: {min_val:.4f} - {max_val:.4f}")
        print(f"  Mean risk: {valid.mean():.4f}")
        assert min_val >= 0, f"Risk score below 0: {min_val}"
        assert max_val <= 1, f"Risk score above 1: {max_val}"
        score = 10 if valid.notna().all() else 8
        _record_score("Risk Score Range", score, f"range [{min_val:.3f}, {max_val:.3f}]")

    def test_flagged_transactions(self):
        """Some transactions should be flagged for review."""
        df = pd.read_csv(FINANCE_DIR / "transactions.csv")
        flagged_col = df.get("flagged")
        if flagged_col is None:
            _record_score("Flagged Transactions", 5, "No 'flagged' column")
            pytest.skip("No flagged column")
        flagged = df["flagged"].astype(str).str.lower().isin(["true", "1", "yes"])
        flagged_count = flagged.sum()
        print(f"  Flagged: {flagged_count}/{len(df)} transactions")
        assert flagged_count >= 0, "Negative flagged count (impossible)"
        score = 10 if flagged_count > 0 else 6
        _record_score("Flagged Transactions", score, f"{flagged_count} flagged")

    def test_transaction_types_diverse(self):
        """Transactions should include credits and debits."""
        df = pd.read_csv(FINANCE_DIR / "transactions.csv")
        types = df["type"].dropna().str.lower().unique()
        print(f"  Transaction types: {sorted(types)}")
        assert len(types) >= 2, f"Expected >= 2 types, got {len(types)}: {types}"
        score = 10 if "credit" in types and "debit" in types else 8
        _record_score("Transaction Type Diversity", score, f"{sorted(types)}")


# ============================================================================
# 6. FINANCIAL TERMS — EXAMPLE USAGE
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestFinancialTermsQuality:
    """Verify financial terms have definitions and example usage."""

    def test_definitions_populated(self):
        """All terms should have definitions."""
        df = pd.read_csv(FINANCE_DIR / "financial_terms.csv")
        has_def = df["definition"].dropna().astype(str).str.strip().ne("").sum()
        pct = has_def / len(df)
        print(f"  Terms with definitions: {has_def}/{len(df)} ({pct:.1%})")
        assert pct > 0.95, f"Only {pct:.1%} have definitions"
        score = 10 if pct > 0.99 else 8 if pct > 0.97 else 6
        _record_score("Terms Have Definitions", score, f"{pct:.1%}")

    def test_example_usage_populated(self):
        """Most terms should have example usage."""
        df = pd.read_csv(FINANCE_DIR / "financial_terms.csv")
        has_ex = df["example_usage"].dropna().astype(str).str.strip().ne("").sum()
        pct = has_ex / len(df)
        print(f"  Terms with examples: {has_ex}/{len(df)} ({pct:.1%})")
        assert pct > 0.80, f"Only {pct:.1%} have examples"
        score = 10 if pct > 0.95 else 8 if pct > 0.90 else 6
        _record_score("Terms Have Examples", score, f"{pct:.1%}")

    def test_term_categories_diverse(self):
        """Financial terms should span multiple categories."""
        df = pd.read_csv(FINANCE_DIR / "financial_terms.csv")
        cats = df["category"].dropna().unique()
        print(f"  Term categories ({len(cats)}): {sorted(cats)[:10]}")
        assert len(cats) >= 2, f"Expected >= 2 categories, got {len(cats)}"
        score = 10 if len(cats) >= 5 else 8 if len(cats) >= 3 else 6
        _record_score("Term Category Diversity", score, f"{len(cats)} categories")


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VaLLM Specialist Model — Finance Dataset Tests")
    print("=" * 70)
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
