"""
VA LLM Specialist Model — Skills Dataset Tests
================================================

Author: Joel Otepa Wembo
https://joelwembo.com

DESCRIPTION
===========
Test suite for skills datasets (8 CSV files under app/data/datasets/skills/).
Validates file existence, column structure, skill level validity,
tool coverage, and keyword search relevance.

No running server is required.

PREREQUISITES
=============
    pip install pytest pandas numpy

USAGE
=====
    python -m pytest app/tests/tests_skills.py -v -s --tb=short
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
SKILLS_DIR = DATASETS_DIR / "skills"

# All expected skills CSV files
SKILLS_CSV_FILES = [
    "cloud_engineering_skills.csv",
    "data_engineering_skills.csv",
    "devops_skills.csv",
    "networking_skills.csv",
    "programming_skills.csv",
    "security_skills.csv",
    "soft_skills.csv",
    "sre_skills.csv",
]

VALID_LEVELS = {"beginner", "intermediate", "advanced", "expert"}

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
    print("  SKILLS TESTS — FINAL SCORECARD")
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

def _load_all_skills() -> pd.DataFrame:
    """Load and concatenate all skills CSV files."""
    frames = []
    for f in SKILLS_CSV_FILES:
        p = SKILLS_DIR / f
        if p.exists():
            frames.append(pd.read_csv(p, on_bad_lines="warn"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
class TestSkillsFilesExist:
    """Verify all 8 skills CSV files exist."""

    def test_skills_directory_exists(self):
        assert SKILLS_DIR.exists() and SKILLS_DIR.is_dir()
        _record_score("Skills Directory Exists", 10, str(SKILLS_DIR))

    def test_all_8_csv_files_exist(self):
        missing = [f for f in SKILLS_CSV_FILES if not (SKILLS_DIR / f).exists()]
        found = len(SKILLS_CSV_FILES) - len(missing)
        assert not missing, f"Missing files: {missing}"
        _record_score("All 8 Skills CSVs Present", 10, f"{found}/{len(SKILLS_CSV_FILES)} files")

    def test_csv_files_have_data(self):
        """Each CSV should have at least 5 rows."""
        small_files = []
        for f in SKILLS_CSV_FILES:
            p = SKILLS_DIR / f
            if p.exists():
                df = pd.read_csv(p)
                if len(df) < 5:
                    small_files.append((f, len(df)))
        if small_files:
            print(f"  Small files: {small_files}")
        assert not small_files, f"Files with < 5 rows: {small_files}"
        _record_score("Skills CSVs Have Data", 10, f"All {len(SKILLS_CSV_FILES)} files >= 5 rows")


# ============================================================================
# 2. COLUMN STRUCTURE
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestSkillsColumnStructure:
    """Verify skill_name, description, subcategory, level, tools columns."""

    @pytest.mark.parametrize("csv_name", SKILLS_CSV_FILES)
    def test_skills_columns(self, csv_name):
        p = SKILLS_DIR / csv_name
        df = pd.read_csv(p, nrows=5)
        required = {"skill_name", "description", "subcategory", "level", "tools"}
        actual = set(df.columns.str.lower().str.strip())
        missing = required - actual
        assert not missing, f"{csv_name} missing columns: {missing}. Has: {list(df.columns)}"

    def test_column_report(self):
        """Report on all skills CSVs."""
        total_skills = 0
        for f in SKILLS_CSV_FILES:
            df = pd.read_csv(SKILLS_DIR / f)
            total_skills += len(df)
            print(f"  {f}: {len(df)} skills, columns={list(df.columns)}")
        _record_score("Skills Column Structure", 10, f"{total_skills} total skills across {len(SKILLS_CSV_FILES)} files")


# ============================================================================
# 3. SKILL LEVELS VALIDATION
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestSkillLevels:
    """Verify skill levels are valid (beginner, intermediate, advanced, expert)."""

    def test_all_levels_valid(self):
        """All skill level values must be in the allowed set."""
        df = _load_all_skills()
        levels = df["level"].dropna().str.lower().str.strip().unique()
        invalid = set(levels) - VALID_LEVELS
        print(f"  Levels found: {sorted(levels)}")
        if invalid:
            print(f"  Invalid levels: {invalid}")
        assert not invalid, f"Invalid levels found: {invalid}. Allowed: {VALID_LEVELS}"
        _record_score("Valid Skill Levels", 10, f"levels: {sorted(levels)}")

    def test_level_distribution(self):
        """Skill levels should have reasonable distribution (not all one level)."""
        df = _load_all_skills()
        dist = df["level"].str.lower().str.strip().value_counts(normalize=True)
        print("  Level distribution:")
        for lev, pct in dist.items():
            print(f"    {lev}: {pct:.1%}")
        max_pct = dist.iloc[0]
        assert max_pct < 0.80, f"Level '{dist.index[0]}' dominates: {max_pct:.1%}"
        score = 10 if max_pct < 0.50 else 8 if max_pct < 0.65 else 6
        _record_score("Level Distribution", score, f"top={dist.index[0]} at {max_pct:.1%}")


# ============================================================================
# 4. TOOL COVERAGE
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestToolCoverage:
    """Verify skills reference a diverse set of tools."""

    def test_tools_populated(self):
        """Most skills should have a tools field."""
        df = _load_all_skills()
        has_tools = df["tools"].dropna().astype(str).str.strip().ne("").sum()
        pct = has_tools / len(df)
        print(f"  Skills with tools: {has_tools}/{len(df)} ({pct:.1%})")
        assert pct > 0.85, f"Only {pct:.1%} have tools"
        score = 10 if pct > 0.95 else 8 if pct > 0.90 else 6
        _record_score("Tools Populated", score, f"{pct:.1%}")

    def test_tool_diversity(self):
        """Skills should reference many distinct tools."""
        df = _load_all_skills()
        all_tools = set()
        for tools_str in df["tools"].dropna().astype(str):
            for t in tools_str.split(","):
                t = t.strip().lower()
                if t and t != "nan":
                    all_tools.add(t)
        print(f"  Distinct tools referenced: {len(all_tools)}")
        print(f"  Sample: {sorted(list(all_tools))[:15]}")
        assert len(all_tools) >= 20, f"Expected >= 20 distinct tools, got {len(all_tools)}"
        score = 10 if len(all_tools) >= 50 else 8 if len(all_tools) >= 30 else 6
        _record_score("Tool Diversity", score, f"{len(all_tools)} distinct tools")


# ============================================================================
# 5. SEARCH QUERIES FOR SKILLS
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestSkillsSearch:
    """Test keyword search across skills datasets."""

    def test_kubernetes_skills_search(self):
        df = _load_all_skills()
        results = _keyword_search(df, "kubernetes")
        count = len(results)
        print(f"  'kubernetes' across all skills: {count} matches")
        assert count > 0, "No Kubernetes skills found"
        _record_score("Kubernetes Skills Search", 10 if count >= 3 else 8, f"{count} matches")

    def test_python_skills_search(self):
        df = _load_all_skills()
        results = _keyword_search(df, "python")
        count = len(results)
        print(f"  'python' across all skills: {count} matches")
        assert count > 0, "No Python skills found"
        _record_score("Python Skills Search", 10 if count >= 2 else 8, f"{count} matches")

    def test_security_skills_search(self):
        df = _load_all_skills()
        results = _keyword_search(df, "security")
        count = len(results)
        print(f"  'security' across all skills: {count} matches")
        assert count > 0, "No security skills found"
        _record_score("Security Skills Search", 10 if count >= 3 else 8, f"{count} matches")

    def test_total_skills_count(self):
        """Report total skills across all files."""
        df = _load_all_skills()
        total = len(df)
        print(f"  Total skills across all files: {total}")
        assert total >= 100, f"Expected >= 100 total skills, got {total}"
        score = 10 if total >= 200 else 8 if total >= 150 else 6
        _record_score("Total Skills Count", score, f"{total} skills")


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VA LLM Specialist Model — Skills Dataset Tests")
    print("=" * 70)
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
