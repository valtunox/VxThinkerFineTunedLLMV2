from app.schemas.query import (
    DeveloperQueryRequest,
    QueryRequest,
    QueryResponse,
    TerminalQueryRequest,
)


def test_query_request_defaults():
    request = QueryRequest(query="Explain Kubernetes liveness probes")

    assert request.top_k == 5
    assert request.include_reasoning is True
    assert request.use_web_search is True
    assert request.web_search_max_results == 5


def test_query_response_supports_answer_and_references():
    response = QueryResponse(
        query="How do I debug an EKS deployment?",
        answer="Check pod events and the deployment rollout history.",
        references=[{"provider": "internal", "title": "cloud_troubleshooting_qa.csv"}],
    )

    assert "deployment rollout" in response.answer
    assert response.references[0]["provider"] == "internal"


def test_developer_and_terminal_requests_enable_web_search():
    dev_request = DeveloperQueryRequest(query="Create Terraform for an S3 bucket")
    terminal_request = TerminalQueryRequest(query="Show PowerShell to list Windows services")

    assert dev_request.use_web_search is True
    assert terminal_request.use_web_search is True


def test_query_route_returns_it_answer(router_client_factory):
    client = router_client_factory(
        generated_answer="Check the failing pod events, rollout status, and recent IAM changes."
    )

    response = client.post(
        "/api/models/v1/query",
        json={
            "query": "Why is my Kubernetes deployment failing after an IAM policy change?",
            "include_reasoning": True,
            "use_web_search": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Kubernetes" in payload["query"]
    assert "failing pod events" in payload["answer"]
    assert payload["confidence"] > 0
    assert payload["results"]
    assert payload["references"]
    assert payload["reasoning"]


def test_query_route_rejects_non_it_scope(router_client_factory):
    client = router_client_factory()

    response = client.post(
        "/api/models/v1/query",
        json={"query": "What medication should I take for chest pain?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "IT work" in payload["answer"] or "IT specialist" in payload["answer"]
    assert payload["confidence"] == 1.0
