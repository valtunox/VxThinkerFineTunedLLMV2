from app.tests.conftest import CUSTOMER_SERVICE_DIR, contains_any, csv_row_count, dataframe_text, load_csv_sample


def test_customer_service_directory_exists():
    assert CUSTOMER_SERVICE_DIR.exists()
    assert CUSTOMER_SERVICE_DIR.is_dir()


def test_customer_service_core_files_exist():
    expected = {"faq.csv", "support_tickets.csv", "escalation_rules.csv"}
    actual = {path.name for path in CUSTOMER_SERVICE_DIR.glob("*.csv")}

    assert expected.issubset(actual)


def test_customer_service_files_have_reasonable_volume():
    faq_rows = csv_row_count(CUSTOMER_SERVICE_DIR / "faq.csv")
    ticket_rows = csv_row_count(CUSTOMER_SERVICE_DIR / "support_tickets.csv")
    escalation_rows = csv_row_count(CUSTOMER_SERVICE_DIR / "escalation_rules.csv")

    assert faq_rows >= 100
    assert ticket_rows >= 1000
    assert escalation_rows >= 10


def test_customer_service_content_is_it_focused():
    faq_text = dataframe_text(load_csv_sample(CUSTOMER_SERVICE_DIR / "faq.csv", rows=200))
    ticket_text = dataframe_text(load_csv_sample(CUSTOMER_SERVICE_DIR / "support_tickets.csv", rows=200))
    combined = f"{faq_text} {ticket_text}"

    assert contains_any(
        combined,
        ["vpn", "cloud", "server", "terraform", "network", "cpu", "backup", "database", "deployment"],
    )


def test_escalation_rules_have_actionable_columns():
    df = load_csv_sample(CUSTOMER_SERVICE_DIR / "escalation_rules.csv", rows=20)

    assert len(df.columns) >= 3
    assert contains_any(" ".join(df.columns).lower(), ["priority", "severity", "team", "escalation", "sla"])
