from pathlib import Path

from app.tests.conftest import CLOUD_DIR, contains_any, csv_row_count, dataframe_text, load_csv_sample


EXPECTED_CLOUD_FILES = {
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
}


def test_cloud_dataset_directory_exists():
    assert CLOUD_DIR.exists()
    assert CLOUD_DIR.is_dir()


def test_all_expected_cloud_files_exist():
    actual = {path.name for path in CLOUD_DIR.glob("*.csv")}

    missing = EXPECTED_CLOUD_FILES - actual
    assert not missing, f"Missing cloud CSV files: {sorted(missing)}"


def test_cloud_files_are_non_empty():
    for file_name in EXPECTED_CLOUD_FILES:
        path = CLOUD_DIR / file_name
        assert path.exists(), f"Missing file: {file_name}"
        assert path.stat().st_size > 0, f"Empty file: {file_name}"


def test_cloud_deployments_has_required_columns():
    df = load_csv_sample(CLOUD_DIR / "cloud_deployments.csv", rows=25)

    expected = {"question", "answer", "category"}
    assert expected.issubset(set(df.columns))


def test_cloud_deployments_contains_major_providers():
    df = load_csv_sample(CLOUD_DIR / "cloud_deployments.csv", rows=500)
    text = dataframe_text(df)

    assert contains_any(text, ["aws", "azure", "gcp"])


def test_core_cloud_corpus_has_meaningful_volume():
    key_files = [
        CLOUD_DIR / "cloud_deployments.csv",
        CLOUD_DIR / "cloud_security_qa.csv",
        CLOUD_DIR / "devops_qa.csv",
        CLOUD_DIR / "sre_qa.csv",
    ]
    counts = [csv_row_count(path) for path in key_files]

    assert counts[0] >= 1000
    assert sum(counts) >= 5000


def test_cloud_sample_text_contains_it_topics():
    sample_files = [
        CLOUD_DIR / "cloud_security_qa.csv",
        CLOUD_DIR / "devops_qa.csv",
        CLOUD_DIR / "networking_qa.csv",
    ]
    combined = " ".join(dataframe_text(load_csv_sample(path, rows=100)) for path in sample_files)

    assert contains_any(combined, ["kubernetes", "aws", "terraform", "network", "incident", "deployment"])
