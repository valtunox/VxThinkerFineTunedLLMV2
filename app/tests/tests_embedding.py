from app.services.ai.ml.llm_router import (
    _build_references,
    _confidence_score,
    _extract_code_blocks,
    _extract_shell_commands,
    _reasoning_summary,
    _to_query_results,
)


def test_reference_builder_merges_internal_and_web_results():
    internal = [
        {
            "document": "Rotate IAM access keys and audit usage.",
            "score": 0.8,
            "metadata": {"file_name": "cloud_security_qa.csv"},
        }
    ]
    web = [
        {
            "title": "AWS IAM best practices",
            "url": "https://example.com/iam",
            "snippet": "Use short-lived credentials where possible.",
            "provider": "duckduckgo",
            "rank": 1,
        }
    ]

    refs = _build_references(internal, web)

    assert len(refs) == 2
    assert refs[0]["provider"] == "internal"
    assert refs[1]["provider"] == "duckduckgo"


def test_to_query_results_maps_internal_documents():
    results = _to_query_results(
        [
            {
                "document": "Kubernetes uses liveness and readiness probes.",
                "score": 0.73,
                "metadata": {"dataset_file": "cloud/cloud_troubleshooting_qa.csv"},
            }
        ]
    )

    assert len(results) == 1
    assert "liveness" in results[0].content
    assert results[0].score == 0.73


def test_code_block_and_shell_command_extraction():
    answer = """
    Use the following:

    ```bash
    kubectl get pods -n prod
    kubectl describe pod api-123 -n prod
    ```

    Then inspect the events.
    """

    blocks = _extract_code_blocks(answer)
    commands = _extract_shell_commands(answer)

    assert len(blocks) == 1
    assert "kubectl get pods" in blocks[0]
    assert commands[:2] == ["kubectl get pods -n prod", "kubectl describe pod api-123 -n prod"]


def test_confidence_and_reasoning_reflect_available_context():
    internal = [{"document": "Terraform state locking prevents concurrent applies.", "score": 0.9, "metadata": {}}]
    web = [{"title": "Terraform docs", "url": "https://example.com", "snippet": "Use remote state.", "provider": "duckduckgo", "rank": 1}]

    confidence = _confidence_score(internal, web)
    reasoning = _reasoning_summary(internal, web)

    assert confidence > 0.5
    assert "internal IT retrieval matches" in reasoning
    assert "live web references" in reasoning
