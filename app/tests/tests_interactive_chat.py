def test_developer_route_returns_code_and_references(router_client_factory):
    client = router_client_factory(
        generated_answer="""
        Use Terraform:

        ```hcl
        resource "aws_s3_bucket" "logs" {
          bucket = "team-logs"
        }
        ```
        """
    )

    response = client.post(
        "/api/models/v1/developer",
        json={
            "query": "Create Terraform for an S3 bucket used for log retention.",
            "language": "hcl",
            "framework": "terraform",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["code_snippets"]
    assert "aws_s3_bucket" in payload["answer"]
    assert payload["references"]


def test_terminal_route_extracts_commands(router_client_factory):
    client = router_client_factory(
        generated_answer="""
        Run:

        ```bash
        kubectl get deploy -n prod
        kubectl rollout status deploy/api -n prod
        ```
        """
    )

    response = client.post(
        "/api/models/v1/terminal",
        json={
            "query": "Give me shell commands to verify a Kubernetes deployment rollout.",
            "shell": "bash",
            "os_type": "linux",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["commands"][:2] == [
        "kubectl get deploy -n prod",
        "kubectl rollout status deploy/api -n prod",
    ]
    assert not payload["warnings"]


def test_query_route_can_disable_web_search(router_client_factory):
    client = router_client_factory(generated_answer="Use kubectl describe pod and inspect recent events.")

    response = client.post(
        "/api/models/v1/query",
        json={
            "query": "How do I debug a crashing pod in Kubernetes?",
            "use_web_search": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert any(ref["provider"] == "internal" for ref in payload["references"])
    assert all(ref["provider"] != "duckduckgo" for ref in payload["references"])


def test_terminal_route_rejects_non_it_requests(router_client_factory):
    client = router_client_factory()

    response = client.post(
        "/api/models/v1/terminal",
        json={"query": "Give me shell commands to calculate a mortgage amortization schedule."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["warnings"]
    assert "IT-only specialist model" in payload["warnings"][0]
