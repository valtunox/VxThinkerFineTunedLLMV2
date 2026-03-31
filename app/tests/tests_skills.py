from app.tests.conftest import SKILLS_DIR, contains_any, dataframe_text, load_csv_sample


def test_skills_directory_contains_csv_and_markdown_assets():
    csv_files = list(SKILLS_DIR.glob("*.csv"))
    doc_files = list(SKILLS_DIR.glob("*.md")) + list(SKILLS_DIR.glob("*.txt"))

    assert csv_files
    assert doc_files


def test_core_skill_csv_files_exist():
    expected = {"devops_skills.csv", "networking_skills.csv", "security_skills.csv", "sre_skills.csv"}
    actual = {path.name for path in SKILLS_DIR.glob("*.csv")}

    assert expected.issubset(actual)


def test_devops_and_security_skill_samples_reference_it_topics():
    devops_text = dataframe_text(load_csv_sample(SKILLS_DIR / "devops_skills.csv", rows=100))
    security_text = dataframe_text(load_csv_sample(SKILLS_DIR / "security_skills.csv", rows=100))
    combined = f"{devops_text} {security_text}"

    assert contains_any(combined, ["terraform", "ci/cd", "docker", "kubernetes", "security", "incident"])


def test_troubleshooting_docs_contain_actionable_language():
    docs = [
        SKILLS_DIR / "linux_administration.md",
        SKILLS_DIR / "networking_troubleshooting.md",
        SKILLS_DIR / "windows_troubleshooting.md",
    ]
    combined = " ".join(path.read_text(encoding="utf-8", errors="ignore").lower() for path in docs)

    assert contains_any(combined, ["solution", "check", "logs", "network", "service", "troubleshoot"])
