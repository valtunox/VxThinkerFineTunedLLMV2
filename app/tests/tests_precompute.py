"""
VaLLM Specialist Model — Precompute Functionality Tests
=========================================================

Author: Joel Otepa Wembo
https://joelwembo.com

DESCRIPTION
===========
Test suite for the precompute module: row_to_text conversion,
CSV file discovery, dataset directory existence, output paths,
and vectorstore directory writability.

No running server is required.

PREREQUISITES
=============
    pip install pytest pandas numpy

USAGE
=====
    python -m pytest app/tests/tests_precompute.py -v -s --tb=short
"""

import sys
import os
import atexit
import tempfile
from pathlib import Path
from typing import Any, Dict, List

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
VECTORSTORE_DIR = _APP_DIR / "data" / "vectorstore"

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
    print("  PRECOMPUTE TESTS — FINAL SCORECARD")
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


# ============================================================================
# 1. ROW_TO_TEXT CONVERSION
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestRowToText:
    """Test the row_to_text function that converts CSV rows to text documents."""

    @pytest.fixture(scope="class")
    def row_to_text_func(self):
        """Import row_to_text from precompute module."""
        try:
            from services.ai.ml.precompute import row_to_text
            return row_to_text
        except ImportError:
            try:
                from app.services.ai.ml.precompute import row_to_text
                return row_to_text
            except ImportError:
                pytest.skip("Cannot import row_to_text from precompute")

    def test_basic_conversion(self, row_to_text_func):
        """A simple dict should produce a pipe-separated string."""
        row = {"question": "What is Kubernetes?", "answer": "Container orchestration", "category": "cloud"}
        result = row_to_text_func(row)
        assert "question: What is Kubernetes?" in result
        assert "answer: Container orchestration" in result
        assert "category: cloud" in result
        assert "|" in result
        print(f"  Result: {result}")
        _record_score("Basic row_to_text", 10, f"{len(result)} chars")

    def test_none_values_skipped(self, row_to_text_func):
        """None values should be skipped in output."""
        row = {"question": "Test?", "answer": None, "category": "finance"}
        result = row_to_text_func(row)
        assert "answer" not in result
        assert "question: Test?" in result
        print(f"  Result (with None): {result}")
        _record_score("None Values Skipped", 10, "None field excluded")

    def test_nan_values_skipped(self, row_to_text_func):
        """NaN string values should be skipped."""
        row = {"question": "Test?", "answer": "nan", "category": "cloud"}
        result = row_to_text_func(row)
        assert "answer" not in result
        print(f"  Result (with nan): {result}")
        _record_score("NaN Values Skipped", 10, "nan field excluded")

    def test_empty_string_skipped(self, row_to_text_func):
        """Empty/whitespace-only strings should be skipped."""
        row = {"question": "Test?", "answer": "  ", "category": ""}
        result = row_to_text_func(row)
        assert "answer" not in result
        assert "category" not in result
        print(f"  Result (with blanks): {result}")
        _record_score("Empty Strings Skipped", 10, "blank fields excluded")


# ============================================================================
# 2. CSV FILE DISCOVERY
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCSVFileDiscovery:
    """Test that the datasets directory contains discoverable CSV files."""

    def test_datasets_directory_exists(self):
        """The main datasets directory should exist."""
        assert DATASETS_DIR.exists() and DATASETS_DIR.is_dir(), f"Not found: {DATASETS_DIR}"
        _record_score("Datasets Directory Exists", 10, str(DATASETS_DIR))

    def test_csv_files_discoverable(self):
        """Recursive glob should find CSV files across all industry sub-dirs."""
        csv_files = sorted(p for p in DATASETS_DIR.rglob("*.csv") if p.is_file())
        print(f"  Total CSV files found: {len(csv_files)}")
        # List by subdirectory
        by_dir: Dict[str, int] = {}
        for p in csv_files:
            rel = p.relative_to(DATASETS_DIR)
            parent = str(rel.parent) if str(rel.parent) != "." else "(root)"
            by_dir[parent] = by_dir.get(parent, 0) + 1
        for d, count in sorted(by_dir.items()):
            print(f"    {d}: {count} files")

        assert len(csv_files) >= 20, f"Expected >= 20 CSV files, got {len(csv_files)}"
        score = 10 if len(csv_files) >= 30 else 8 if len(csv_files) >= 25 else 6
        _record_score("CSV File Discovery", score, f"{len(csv_files)} CSV files in {len(by_dir)} dirs")

    def test_all_industry_subdirs_exist(self):
        """Expected industry subdirectories should be present."""
        expected = {"cloud", "finance", "healthcare", "customer_service", "automation", "skills"}
        actual = {d.name for d in DATASETS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")}
        missing = expected - actual
        print(f"  Expected dirs: {sorted(expected)}")
        print(f"  Found dirs: {sorted(actual)}")
        if missing:
            print(f"  Missing: {missing}")
        assert not missing, f"Missing industry directories: {missing}"
        _record_score("Industry Subdirectories", 10, f"All {len(expected)} present")


# ============================================================================
# 3. OUTPUT PATHS & VECTORSTORE WRITABILITY
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestPrecomputeOutput:
    """Test precompute output directory and file paths."""

    def test_vectorstore_directory_exists(self):
        """Vectorstore output directory should exist (created by EmbeddingService or manually)."""
        exists = VECTORSTORE_DIR.exists()
        if not exists:
            # It should be created automatically, but check
            print(f"  Vectorstore dir does not exist yet: {VECTORSTORE_DIR}")
            _record_score("Vectorstore Dir Exists", 6, "Not yet created (run precompute)")
        else:
            _record_score("Vectorstore Dir Exists", 10, str(VECTORSTORE_DIR))

    def test_vectorstore_directory_writable(self):
        """Vectorstore directory should be writable."""
        if not VECTORSTORE_DIR.exists():
            VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        # Try writing a temp file
        test_file = VECTORSTORE_DIR / ".write_test"
        try:
            test_file.write_text("test")
            assert test_file.exists()
            _record_score("Vectorstore Writable", 10, "Write test passed")
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_expected_output_files(self):
        """After precompute, index.faiss and documents.pkl should exist."""
        index_path = VECTORSTORE_DIR / "index.faiss"
        pkl_path = VECTORSTORE_DIR / "documents.pkl"
        index_exists = index_path.exists()
        pkl_exists = pkl_path.exists()
        print(f"  index.faiss: {'exists' if index_exists else 'MISSING'}"
              f"  ({index_path.stat().st_size / 1024:.0f} KB)" if index_exists else "")
        print(f"  documents.pkl: {'exists' if pkl_exists else 'MISSING'}"
              f"  ({pkl_path.stat().st_size / 1024:.0f} KB)" if pkl_exists else "")

        if index_exists and pkl_exists:
            score = 10
            details = (f"index.faiss={index_path.stat().st_size / 1024:.0f}KB, "
                       f"documents.pkl={pkl_path.stat().st_size / 1024:.0f}KB")
        elif index_exists or pkl_exists:
            score = 6
            details = "Partial output (run precompute to generate both)"
        else:
            score = 4
            details = "No precomputed files (run: python -m app.services.ai.ml.precompute)"
        _record_score("Precompute Output Files", score, details)

    def test_precompute_module_importable(self):
        """The precompute module should be importable."""
        try:
            from services.ai.ml.precompute import row_to_text
            _record_score("Precompute Module Import", 10, "Imported via services.ai.ml.precompute")
        except ImportError:
            try:
                from app.services.ai.ml.precompute import row_to_text
                _record_score("Precompute Module Import", 10, "Imported via app.services.ai.ml.precompute")
            except ImportError as e:
                _record_score("Precompute Module Import", 3, f"Import failed: {e}")
                pytest.fail(f"Cannot import precompute module: {e}")


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VaLLM Specialist Model — Precompute Functionality Tests")
    print("=" * 70)
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
