from pathlib import Path

from app.services.ai.ml.specialist_profile import (
    assess_specialist_scope,
    filter_specialist_csv_paths,
)
from app.services.ai.ml.web_search import (
    _DuckDuckGoHTMLParser,
    _normalize_duckduckgo_href,
    web_search_service,
)


def test_it_scope_allows_cloud_devops_queries():
    assessment = assess_specialist_scope("How do I troubleshoot a failing Kubernetes deployment on AWS?")

    assert assessment.allowed is True
    assert "kubernetes" in assessment.matched_it_terms


def test_it_scope_rejects_healthcare_queries():
    assessment = assess_specialist_scope("What medication should a patient take for a fever?")

    assert assessment.allowed is False
    assert assessment.blocked_domain == "healthcare"


def test_specialist_dataset_filter_keeps_only_it_directories(tmp_path: Path):
    cloud_file = tmp_path / "cloud" / "cloud_deployments.csv"
    finance_file = tmp_path / "finance" / "finance_records.csv"
    skills_file = tmp_path / "skills" / "devops_skills.csv"

    cloud_file.parent.mkdir(parents=True)
    finance_file.parent.mkdir(parents=True)
    skills_file.parent.mkdir(parents=True)

    for file_path in (cloud_file, finance_file, skills_file):
        file_path.write_text("col\nvalue\n", encoding="utf-8")

    filtered = filter_specialist_csv_paths([cloud_file, finance_file, skills_file], tmp_path)

    assert cloud_file in filtered
    assert skills_file in filtered
    assert finance_file not in filtered


def test_duckduckgo_redirect_url_is_decoded():
    href = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc"

    assert _normalize_duckduckgo_href(href) == "https://example.com/doc"


def test_duckduckgo_html_parser_extracts_results():
    parser = _DuckDuckGoHTMLParser()
    parser.feed(
        """
        <div class="result results_links results_links_deep web-result">
          <h2 class="result__title">
            <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fguide">Kubernetes Guide</a>
          </h2>
          <a class="result__snippet" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fguide">
            Troubleshooting pods in production
          </a>
        </div>
        """
    )

    assert len(parser.results) == 1
    assert parser.results[0]["title"] == "Kubernetes Guide"
    assert parser.results[0]["url"] == "https://example.com/guide"
    assert "Troubleshooting pods" in parser.results[0]["snippet"]


def test_web_search_prompt_formatting():
    formatted = web_search_service.format_results_for_prompt(
        [
            {
                "title": "AWS IAM Best Practices",
                "url": "https://example.com/iam",
                "snippet": "Rotate access keys regularly.",
                "provider": "duckduckgo",
                "rank": 1,
            }
        ]
    )

    assert "AWS IAM Best Practices" in formatted
    assert "Rotate access keys regularly." in formatted
