from app.tests.conftest import CLOUD_DIR, SKILLS_DIR, contains_any, dataframe_text, load_csv_sample


def test_cloud_troubleshooting_corpus_mentions_incident_workflows():
    df = load_csv_sample(CLOUD_DIR / "cloud_troubleshooting_qa.csv", rows=200)
    text = dataframe_text(df)

    assert contains_any(text, ["incident", "debug", "latency", "outage", "logs", "restart"])


def test_sre_and_networking_qas_contain_operational_keywords():
    sre_text = dataframe_text(load_csv_sample(CLOUD_DIR / "sre_qa.csv", rows=200))
    networking_text = dataframe_text(load_csv_sample(CLOUD_DIR / "networking_qa.csv", rows=200))
    combined = f"{sre_text} {networking_text}"

    assert contains_any(combined, ["availability", "latency", "packet", "dns", "retry", "service"])


def test_programming_and_database_troubleshooting_docs_exist():
    docs = [
        SKILLS_DIR / "programming_debugging.md",
        SKILLS_DIR / "database_troubleshooting.md",
    ]

    for path in docs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_troubleshooting_docs_include_practical_steps():
    docs = [
        SKILLS_DIR / "programming_debugging.md",
        SKILLS_DIR / "database_troubleshooting.md",
        SKILLS_DIR / "hardware_troubleshooting.txt",
    ]
    combined = " ".join(path.read_text(encoding="utf-8", errors="ignore").lower() for path in docs)

    assert contains_any(combined, ["check", "verify", "logs", "restart", "monitor", "solution"])
