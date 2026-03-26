"""
VA LLM Specialist Model — Healthcare Dataset Tests
====================================================

Author: Joel Otepa Wembo
https://joelwembo.com

DESCRIPTION
===========
Test suite for healthcare datasets: clinical_notes.csv, icd_codes.csv,
and medical_qa.csv. Validates CSV structure, row counts, search
relevance, ICD code format, and clinical notes required fields.

No running server is required.

PREREQUISITES
=============
    pip install pytest pandas numpy

USAGE
=====
    python -m pytest app/tests/tests_healthcare.py -v -s --tb=short
"""

import re
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
HEALTH_DIR = DATASETS_DIR / "healthcare"

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
    print("  HEALTHCARE TESTS — FINAL SCORECARD")
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
class TestHealthcareFilesExist:
    """Verify healthcare dataset files exist."""

    def test_directory_exists(self):
        assert HEALTH_DIR.exists() and HEALTH_DIR.is_dir()
        _record_score("Healthcare Directory Exists", 10, str(HEALTH_DIR))

    def test_clinical_notes_exists(self):
        p = HEALTH_DIR / "clinical_notes.csv"
        assert p.exists(), f"Not found: {p}"
        _record_score("clinical_notes.csv Exists", 10, f"{p.stat().st_size / 1024:.1f} KB")

    def test_icd_codes_exists(self):
        p = HEALTH_DIR / "icd_codes.csv"
        assert p.exists(), f"Not found: {p}"
        _record_score("icd_codes.csv Exists", 10, f"{p.stat().st_size / 1024:.1f} KB")

    def test_medical_qa_exists(self):
        p = HEALTH_DIR / "medical_qa.csv"
        assert p.exists(), f"Not found: {p}"
        _record_score("medical_qa.csv Exists", 10, f"{p.stat().st_size / 1024:.1f} KB")


# ============================================================================
# 2. COLUMN STRUCTURE
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestHealthcareColumns:
    """Verify correct column schemas per file."""

    def test_clinical_notes_columns(self):
        df = pd.read_csv(HEALTH_DIR / "clinical_notes.csv", nrows=5)
        required = {"patient_id", "symptoms", "diagnosis", "treatment", "medications", "follow_up"}
        actual = set(df.columns.str.lower().str.strip())
        missing = required - actual
        assert not missing, f"Missing: {missing}. Has: {list(df.columns)}"
        _record_score("clinical_notes Columns", 10, f"{list(df.columns)}")

    def test_icd_codes_columns(self):
        df = pd.read_csv(HEALTH_DIR / "icd_codes.csv", nrows=5)
        required = {"code", "description", "category", "common_symptoms", "typical_treatment"}
        actual = set(df.columns.str.lower().str.strip())
        missing = required - actual
        assert not missing, f"Missing: {missing}. Has: {list(df.columns)}"
        _record_score("icd_codes Columns", 10, f"{list(df.columns)}")

    def test_medical_qa_columns(self):
        df = pd.read_csv(HEALTH_DIR / "medical_qa.csv", nrows=5)
        required = {"question", "answer", "category"}
        actual = set(df.columns.str.lower().str.strip())
        missing = required - actual
        assert not missing, f"Missing: {missing}. Has: {list(df.columns)}"
        _record_score("medical_qa Columns", 10, f"{list(df.columns)}")


# ============================================================================
# 3. ROW COUNTS
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestHealthcareRowCounts:
    """Verify adequate data volume."""

    def test_clinical_notes_count(self):
        df = pd.read_csv(HEALTH_DIR / "clinical_notes.csv")
        count = len(df)
        print(f"  clinical_notes.csv: {count:,} rows")
        assert count >= 100, f"Expected >= 100, got {count}"
        score = 10 if count >= 250 else 8 if count >= 150 else 6
        _record_score("clinical_notes Row Count", score, f"{count:,} rows")

    def test_icd_codes_count(self):
        df = pd.read_csv(HEALTH_DIR / "icd_codes.csv")
        count = len(df)
        print(f"  icd_codes.csv: {count:,} rows")
        assert count >= 100, f"Expected >= 100, got {count}"
        score = 10 if count >= 250 else 8 if count >= 150 else 6
        _record_score("icd_codes Row Count", score, f"{count:,} rows")

    def test_medical_qa_count(self):
        df = pd.read_csv(HEALTH_DIR / "medical_qa.csv")
        count = len(df)
        print(f"  medical_qa.csv: {count:,} rows")
        assert count >= 100, f"Expected >= 100, got {count}"
        score = 10 if count >= 250 else 8 if count >= 150 else 6
        _record_score("medical_qa Row Count", score, f"{count:,} rows")


# ============================================================================
# 4. SEARCH QUERIES
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestHealthcareSearch:
    """Test keyword search across healthcare datasets."""

    def test_diabetes_search(self):
        df = pd.read_csv(HEALTH_DIR / "medical_qa.csv")
        results = _keyword_search(df, "diabetes")
        count = len(results)
        print(f"  'diabetes' in medical_qa: {count} matches")
        assert count > 0, "No results for diabetes"
        _record_score("Diabetes Search", 10 if count >= 3 else 8, f"{count} matches")

    def test_cardiac_search(self):
        """Search for cardiac / heart content."""
        df = pd.read_csv(HEALTH_DIR / "medical_qa.csv")
        results_cardiac = _keyword_search(df, "cardiac")
        results_heart = _keyword_search(df, "heart")
        combined = len(set(results_cardiac.index) | set(results_heart.index))
        print(f"  cardiac/heart in medical_qa: {combined} matches")
        assert combined > 0, "No cardiac/heart results"
        _record_score("Cardiac Search", 10 if combined >= 3 else 8, f"{combined} matches")

    def test_medication_interaction_search(self):
        """Search for medication-related content."""
        df = pd.read_csv(HEALTH_DIR / "medical_qa.csv")
        results = _keyword_search(df, "medication")
        count = len(results)
        print(f"  'medication' in medical_qa: {count} matches")
        # Also check clinical notes
        df_cn = pd.read_csv(HEALTH_DIR / "clinical_notes.csv")
        cn_results = _keyword_search(df_cn, "medication", columns=["medications"])
        total = count + len(cn_results)
        print(f"  'medication' total across files: {total}")
        assert total > 0, "No medication results"
        _record_score("Medication Search", 10 if total >= 5 else 8, f"{total} matches")


# ============================================================================
# 5. ICD CODE FORMAT
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestICDCodeFormat:
    """Verify ICD codes follow standard format (e.g., I21.0, E11.9)."""

    def test_icd_code_pattern(self):
        """ICD-10 codes should match pattern: letter + digits, optionally dot + digits."""
        df = pd.read_csv(HEALTH_DIR / "icd_codes.csv")
        icd_pattern = re.compile(r"^[A-Z]\d{2}(\.\d{1,2})?$")
        codes = df["code"].dropna().astype(str).str.strip()
        valid = codes.apply(lambda c: bool(icd_pattern.match(c)))
        valid_count = valid.sum()
        pct = valid_count / len(codes)
        print(f"  Valid ICD-10 codes: {valid_count}/{len(codes)} ({pct:.1%})")

        # Show some examples
        invalid_examples = codes[~valid].head(5).tolist()
        if invalid_examples:
            print(f"  Invalid examples: {invalid_examples}")

        assert pct > 0.80, f"Only {pct:.1%} codes match ICD-10 pattern"
        score = 10 if pct > 0.95 else 8 if pct > 0.90 else 6
        _record_score("ICD Code Format", score, f"{pct:.1%} valid")

    def test_icd_codes_unique(self):
        """ICD codes should be mostly unique."""
        df = pd.read_csv(HEALTH_DIR / "icd_codes.csv")
        total = len(df)
        unique = df["code"].nunique()
        pct_unique = unique / total
        print(f"  Unique codes: {unique}/{total} ({pct_unique:.1%})")
        assert pct_unique > 0.70, f"Too many duplicate codes: {pct_unique:.1%}"
        score = 10 if pct_unique > 0.95 else 8 if pct_unique > 0.85 else 6
        _record_score("ICD Code Uniqueness", score, f"{unique} unique out of {total}")

    def test_icd_categories_diverse(self):
        """ICD codes should span multiple medical categories."""
        df = pd.read_csv(HEALTH_DIR / "icd_codes.csv")
        cats = df["category"].dropna().unique()
        print(f"  ICD categories ({len(cats)}): {sorted(cats)[:10]}")
        assert len(cats) >= 3, f"Expected >= 3 categories, got {len(cats)}"
        score = 10 if len(cats) >= 8 else 8 if len(cats) >= 5 else 6
        _record_score("ICD Category Diversity", score, f"{len(cats)} categories")


# ============================================================================
# 6. CLINICAL NOTES REQUIRED FIELDS
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestClinicalNotesQuality:
    """Verify clinical notes have required medical information."""

    def test_diagnosis_populated(self):
        """Every clinical note should have a diagnosis."""
        df = pd.read_csv(HEALTH_DIR / "clinical_notes.csv")
        has_diag = df["diagnosis"].dropna().astype(str).str.strip().ne("").sum()
        pct = has_diag / len(df)
        print(f"  Notes with diagnosis: {has_diag}/{len(df)} ({pct:.1%})")
        assert pct > 0.90, f"Only {pct:.1%} have diagnosis"
        score = 10 if pct > 0.98 else 8 if pct > 0.95 else 6
        _record_score("Diagnosis Populated", score, f"{pct:.1%}")

    def test_treatment_populated(self):
        """Clinical notes should include treatment plans."""
        df = pd.read_csv(HEALTH_DIR / "clinical_notes.csv")
        has_treat = df["treatment"].dropna().astype(str).str.strip().ne("").sum()
        pct = has_treat / len(df)
        print(f"  Notes with treatment: {has_treat}/{len(df)} ({pct:.1%})")
        assert pct > 0.90, f"Only {pct:.1%} have treatment"
        score = 10 if pct > 0.98 else 8 if pct > 0.95 else 6
        _record_score("Treatment Populated", score, f"{pct:.1%}")

    def test_medications_populated(self):
        """Clinical notes should list medications."""
        df = pd.read_csv(HEALTH_DIR / "clinical_notes.csv")
        has_meds = df["medications"].dropna().astype(str).str.strip().ne("").sum()
        pct = has_meds / len(df)
        print(f"  Notes with medications: {has_meds}/{len(df)} ({pct:.1%})")
        assert pct > 0.85, f"Only {pct:.1%} have medications"
        score = 10 if pct > 0.95 else 8 if pct > 0.90 else 6
        _record_score("Medications Populated", score, f"{pct:.1%}")

    def test_follow_up_populated(self):
        """Clinical notes should include follow-up instructions."""
        df = pd.read_csv(HEALTH_DIR / "clinical_notes.csv")
        has_fu = df["follow_up"].dropna().astype(str).str.strip().ne("").sum()
        pct = has_fu / len(df)
        print(f"  Notes with follow-up: {has_fu}/{len(df)} ({pct:.1%})")
        assert pct > 0.85, f"Only {pct:.1%} have follow-up"
        score = 10 if pct > 0.95 else 8 if pct > 0.90 else 6
        _record_score("Follow-Up Populated", score, f"{pct:.1%}")


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VA LLM Specialist Model — Healthcare Dataset Tests")
    print("=" * 70)
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
