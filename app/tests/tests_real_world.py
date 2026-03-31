from app.tests.conftest import (
    AUTOMATION_DIR,
    CLOUD_DIR,
    CUSTOMER_SERVICE_DIR,
    DATASETS_DIR,
    SKILLS_DIR,
    csv_row_count,
    load_csv_sample,
)


def test_it_specialist_dataset_groups_exist():
    expected = {"cloud", "automation", "customer_service", "skills", "uploaded"}
    actual = {path.name for path in DATASETS_DIR.iterdir() if path.is_dir()}

    assert expected.issubset(actual)


def test_large_support_ticket_corpus_exists():
    support_tickets = CUSTOMER_SERVICE_DIR / "support_tickets.csv"

    assert support_tickets.exists()
    assert support_tickets.stat().st_size > 10 * 1024 * 1024


def test_cloud_and_automation_corpora_have_real_volume():
    total_rows = sum(
        csv_row_count(path)
        for path in [
            CLOUD_DIR / "cloud_deployments.csv",
            CLOUD_DIR / "devops_qa.csv",
            AUTOMATION_DIR / "automation_workflows.csv",
            CUSTOMER_SERVICE_DIR / "faq.csv",
        ]
    )

    assert total_rows >= 10000


def test_cloud_deployment_records_have_actionable_fields():
    df = load_csv_sample(CLOUD_DIR / "cloud_deployments.csv", rows=100)
    combined = " ".join(df.astype(str).fillna("").agg(" ".join, axis=1).tolist()).lower()

    assert df["question"].notna().any()
    assert df["answer"].notna().any()
    assert df["category"].notna().any()
    assert any(term in combined for term in ["deploy", "aws", "ec2", "kubernetes", "terraform"])


def test_skills_corpus_contains_multiple_it_disciplines():
    skill_csvs = list(SKILLS_DIR.glob("*.csv"))
    names = {path.name for path in skill_csvs}

    assert {"devops_skills.csv", "networking_skills.csv", "security_skills.csv", "sre_skills.csv"}.issubset(names)
