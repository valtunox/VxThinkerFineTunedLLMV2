"""
VaLLM Integration Tests - 7 tests, provisioning only.

Tests cover all 6 provisioning intents and one V1 RAG provision query.
No incident, cost, or troubleshooting tests.

Start the app first:
    uvicorn app.app:app --host 0.0.0.0 --port 8745

Run tests:
    python -m app.tests.tests               # batch mode (7 tests)
    python -m app.tests.tests --interactive  # interactive chat
    python -m app.tests.tests --interactive "deploy ec2 instance 20gb"
    python -m app.tests.tests --url http://localhost:8745  # custom URL
"""
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests

BASE_URL = "http://localhost:8745"


# ---------------------------------------------------------------------------
# Test definitions: 7 tests, provisioning only (6 provision-intent + 1 V1 RAG)
# ---------------------------------------------------------------------------
TESTS: List[Dict[str, Any]] = [
    # ---- Provisioning Intents (6 types) ----
    # Queries modelled on actual training prompts from cloud_deployments.csv
    {
        "id": 1,
        "name": "Provision VM (EC2 t3.medium in us-west-2)",
        "endpoint": "/api/cloud/provision-intent",
        "payload": {"query": "Deploy EC2 instance t3.medium 50GB gp3 Ubuntu in us-west-2"},
        "expected_query_type": "provisioning",
        "expected_intent": "provision_vm",
        "expected_payload_keys": ["instance_type", "region", "cloud_provider", "os", "volume_size"],
        "description": "Matches training pattern: 'Deploy ec2 instance 30gb t2 micro'. Short, direct, includes instance_type + size + OS + region.",
    },
    {
        "id": 2,
        "name": "Provision Kubernetes (EKS 3-node cluster)",
        "endpoint": "/api/cloud/provision-intent",
        "payload": {"query": "Create an EKS cluster, 3 nodes, m5.large, kubernetes 1.29 in us-east-1"},
        "expected_query_type": "provisioning",
        "expected_intent": "provision_kubernetes",
        "expected_payload_keys": ["cluster_name", "node_count", "node_type", "kubernetes_version", "region"],
        "description": "Matches training pattern: 'Create an EKS cluster, 4 nodes, m5.xlarge'. Uses comma-separated params.",
    },
    {
        "id": 3,
        "name": "Provision Docker (Nginx container)",
        "endpoint": "/api/cloud/provision-intent",
        "payload": {"query": "Run a nginx Docker container, port 80:80"},
        "expected_query_type": "provisioning",
        "expected_intent": "provision_docker",
        "expected_payload_keys": ["docker_image", "container_name", "ports"],
        "description": "Matches training pattern: 'Run a nginx Docker container'. Short with image name + port.",
    },
    {
        "id": 4,
        "name": "Provision Database (PostgreSQL 16)",
        "endpoint": "/api/cloud/provision-intent",
        "payload": {"query": "Deploy PostgreSQL database, version 16, name analytics_db, user admin"},
        "expected_query_type": "provisioning",
        "expected_intent": "provision_database",
        "expected_payload_keys": ["database_engine", "database_name", "database_user", "port"],
        "description": "Matches training pattern: 'Deploy Postgres database, version 16'. Includes engine, name, user.",
    },
    {
        "id": 5,
        "name": "Provision FastAPI application",
        "endpoint": "/api/cloud/provision-intent",
        "payload": {"query": "Deploy FastAPI app billing-api, port 8000, http port 80"},
        "expected_query_type": "provisioning",
        "expected_intent": "provision_fastapi",
        "expected_payload_keys": ["app_name", "app_port", "http_port"],
        "description": "Matches training pattern: 'Deploy FastAPI to app-770.example.com, port 8000'. Includes app_name + ports.",
    },
    {
        "id": 6,
        "name": "Provision Static Website",
        "endpoint": "/api/cloud/provision-intent",
        "payload": {"query": "Deploy static website to nginx on docs.example.com, port 80"},
        "expected_query_type": "provisioning",
        "expected_intent": "provision_static_website",
        "expected_payload_keys": ["server_name", "http_port"],
        "description": "Matches training pattern: 'Deploy static website to nginx on web-103.example.com'. Includes server_name + port.",
    },
    # ---- V1 RAG + Reasoning (provision only) ----
    {
        "id": 7,
        "name": "V1 RAG query with reasoning (provision intent)",
        "endpoint": "/api/models/v1/query",
        "payload": {
            "query": "Deploy a small EC2 instance with 30GB disk in us-east-1 with Ubuntu",
            "include_reasoning": True,
            "top_k": 5,
        },
        "expected_response_keys": ["response", "reasoning", "context"],
        "expected_reasoning_intents": ["provision"],
        "description": "Full RAG pipeline test. 'Deploy' triggers provision intent in reasoning engine.",
    },
]


class VaLLMClient:
    """Client for VaLLM API testing."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def post(self, endpoint: str, payload: dict, timeout: int = 60) -> Optional[dict]:
        try:
            r = self.session.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"_error": str(e)}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_provision_intent(test: dict, data: dict) -> List[str]:
    """Validate a provision-intent response against expected values."""
    errors = []

    if "_error" in data:
        return [f"Request failed: {data['_error']}"]

    # Check query_type
    actual_qt = data.get("query_type")
    expected_qt = test.get("expected_query_type")
    if expected_qt and actual_qt != expected_qt:
        errors.append(f"query_type: expected '{expected_qt}', got '{actual_qt}'")

    # Check intent
    actual_intent = data.get("intent")
    expected_intent = test.get("expected_intent")
    if expected_intent is not None and actual_intent != expected_intent:
        errors.append(f"intent: expected '{expected_intent}', got '{actual_intent}'")
    elif expected_intent is None and actual_intent is not None:
        errors.append(f"intent: expected None, got '{actual_intent}'")

    # Check payload keys exist (for provisioning intents)
    expected_keys = test.get("expected_payload_keys", [])
    payload = data.get("payload") or {}
    for key in expected_keys:
        if key not in payload:
            errors.append(f"payload missing key: '{key}'")

    # Check confidence > 0 for provisioning
    if expected_qt == "provisioning":
        confidence = data.get("confidence", 0)
        if confidence < 0.2:
            errors.append(f"confidence too low: {confidence:.4f} (threshold: 0.2)")

    return errors


def validate_v1_query(test: dict, data: dict) -> List[str]:
    """Validate a V1 query response."""
    errors = []

    if "_error" in data:
        return [f"Request failed: {data['_error']}"]

    # Check required keys
    for key in test.get("expected_response_keys", []):
        if key not in data:
            errors.append(f"missing response key: '{key}'")

    # Check reasoning intent
    expected_intents = test.get("expected_reasoning_intents", [])
    if expected_intents and "reasoning" in data:
        actual_intent = (data["reasoning"].get("intent") or "").lower()
        if not any(ei in actual_intent for ei in expected_intents):
            errors.append(
                f"reasoning intent: expected one of {expected_intents}, got '{actual_intent}'"
            )

    # Check confidence
    if "reasoning" in data:
        confidence = data["reasoning"].get("confidence", 0)
        if confidence < 0.1:
            errors.append(f"reasoning confidence too low: {confidence:.4f}")

    # Check context returned
    if "context" in data and len(data.get("context", [])) == 0:
        errors.append("context is empty (expected at least 1 document)")

    return errors


# ---------------------------------------------------------------------------
# Main runners
# ---------------------------------------------------------------------------
def run_tests(base_url: str = BASE_URL) -> None:
    """Run all 7 provisioning tests against the live API."""
    print("\n" + "=" * 78)
    print("  VaLLM Model Accuracy Tests - 7 Tests (provisioning only)")
    print("=" * 78)
    print(f"  Base URL: {base_url}")
    print(f"  Tests:    {len(TESTS)}")
    print("=" * 78)

    client = VaLLMClient(base_url)

    print("\nChecking service health...", end=" ")
    if not client.health_check():
        print("FAILED")
        print("\nService not available. Start the app first:")
        print("  uvicorn app.app:app --host 0.0.0.0 --port 8745")
        return
    print("OK\n")

    passed = 0
    failed = 0
    results = []

    for test in TESTS:
        test_id = test["id"]
        test_name = test["name"]
        endpoint = test["endpoint"]
        payload = test["payload"]

        print(f"[{test_id:02d}/7] {test_name}")
        print(f"        Endpoint: {endpoint}")
        print(f"        Query:    {payload.get('query', payload.get('command', ''))[:70]}...")

        start = time.perf_counter()
        data = client.post(endpoint, payload)
        duration_ms = (time.perf_counter() - start) * 1000

        # Validate based on endpoint type
        if endpoint == "/api/cloud/provision-intent":
            errors = validate_provision_intent(test, data)
        else:
            errors = validate_v1_query(test, data)

        if errors:
            failed += 1
            status = "FAIL"
            print(f"        Status:   FAIL ({duration_ms:.0f}ms)")
            for err in errors:
                print(f"          - {err}")
        else:
            passed += 1
            status = "PASS"
            print(f"        Status:   PASS ({duration_ms:.0f}ms)")

            # Print key results
            if endpoint == "/api/cloud/provision-intent":
                print(f"          query_type={data.get('query_type')}, "
                      f"intent={data.get('intent')}, "
                      f"confidence={data.get('confidence', 0):.4f}")
                payload = data.get("payload") or {}
                if payload:
                    # Show full payload: storage, instance_type, size, cloud_provider, and every field
                    def _show(v: Any) -> bool:
                        if v is None: return False
                        s = str(v).strip().lower()
                        return s not in ("", "nan", "none")
                    print("          payload:")
                    for k in sorted(payload.keys()):
                        v = payload[k]
                        if _show(v):
                            print(f"            {k}: {v}")
            elif "reasoning" in (data or {}):
                r = data["reasoning"]
                print(f"          intent={r.get('intent')}, "
                      f"confidence={r.get('confidence', 0):.2f}, "
                      f"steps={len(r.get('steps', []))}, "
                      f"context_docs={len(data.get('context', []))}")
                # Include LLM response for troubleshoot (and provision) V1 queries
                resp_text = (data.get("response") or "").strip()
                if resp_text:
                    print("          LLM response:")
                    excerpt = resp_text[:1500] + ("..." if len(resp_text) > 1500 else "")
                    for line in excerpt.split("\n"):
                        print(f"            {line}")
                    if len(resp_text) > 1500:
                        print(f"            ... ({len(resp_text)} chars total)")

        results.append({
            "id": test_id,
            "name": test_name,
            "status": status,
            "duration_ms": round(duration_ms, 1),
            "errors": errors,
        })
        print()

    # Summary
    print("=" * 78)
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(TESTS)} total")
    print(f"  Pass Rate: {passed / len(TESTS) * 100:.0f}%")
    print("=" * 78)

    if failed > 0:
        print("\n  Failed tests:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"    [{r['id']:02d}] {r['name']}")
                for err in r["errors"]:
                    print(f"         - {err}")

    print()


# ---------------------------------------------------------------------------
# Smart routing: detect if a query is provisioning or general
# ---------------------------------------------------------------------------
_PROVISION_KEYWORDS = [
    "deploy", "provision", "create", "launch", "set up", "setup", "spin up",
    "host", "start", "run", "install",
    "ec2", "instance", "vm", "virtual machine", "server",
    "kubernetes", "eks", "aks", "gke", "k8s", "cluster", "node",
    "docker", "container", "nginx", "image",
    "database", "postgres", "mysql", "rds", "mongodb", "db",
    "fastapi", "flask", "django", "api", "app",
    "website", "static site", "static website", "s3 bucket",
]


def _is_provision_query(query: str) -> bool:
    """Heuristic: does this look like a provisioning request?"""
    q = query.lower()
    matches = sum(1 for kw in _PROVISION_KEYWORDS if kw in q)
    return matches >= 2


def _print_provision_response(data: dict) -> None:
    """Pretty-print a provision-intent response."""
    qt = data.get("query_type", "unknown")
    intent = data.get("intent")
    confidence = data.get("confidence", 0)
    payload = data.get("payload") or {}
    match_prompt = data.get("match_prompt", "")

    print(f"\n  Query Type:  {qt}")
    if intent:
        print(f"  Intent:      {intent}")
    print(f"  Confidence:  {confidence:.4f}")

    if match_prompt:
        preview = match_prompt[:120]
        if len(match_prompt) > 120:
            preview += "..."
        print(f"  Matched:     {preview}")

    if payload:
        print(f"\n  Payload (Golang-ready):")
        for k, v in payload.items():
            if v and str(v).lower() not in ("", "nan"):
                print(f"    {k}: {v}")
    elif qt != "provisioning":
        print(f"\n  This is a non-provisioning query ({qt}).")
        print("  The agent would use an LLM to answer this type of question.")


def _print_rag_response(data: dict) -> None:
    """Pretty-print a V1 RAG + reasoning response."""
    response_text = data.get("response", "")
    reasoning = data.get("reasoning") or {}
    context = data.get("context") or []

    if response_text:
        print(f"\n  Response:")
        for line in response_text[:800].split("\n"):
            print(f"    {line}")
        if len(response_text) > 800:
            print(f"    ... ({len(response_text)} chars total)")

    if reasoning:
        intent = reasoning.get("intent", "unknown")
        confidence = reasoning.get("confidence", 0)
        steps = reasoning.get("steps") or []
        print(f"\n  Reasoning:")
        print(f"    Intent:     {intent}")
        print(f"    Confidence: {confidence:.2%}")
        if steps:
            print(f"    Steps:")
            for idx, step in enumerate(steps, 1):
                print(f"      {idx}. {step}")

    if context:
        print(f"\n  Context: {len(context)} documents")
        for idx, item in enumerate(context[:3], 1):
            doc = (item.get("document") or "").strip().replace("\n", " ")
            preview = doc[:180]
            if len(doc) > 180:
                preview += "..."
            score = item.get("score", 0.0)
            doc_type = item.get("type", "unknown")
            print(f"    {idx}. [{doc_type}] score={score:.4f}")
            print(f"       {preview}")


def _process_chat_query(client: VaLLMClient, user_input: str) -> None:
    """Route a single query, call API, and pretty-print the result."""
    # Explicit /cloud prefix forces provision-intent
    if user_input.startswith("/cloud "):
        query = user_input[7:].strip()
        endpoint = "/api/cloud/provision-intent"
        payload = {"query": query}
        mode = "provision-intent"
    # Explicit /rag prefix forces RAG
    elif user_input.startswith("/rag "):
        query = user_input[5:].strip()
        endpoint = "/api/models/v1/query"
        payload = {"query": query, "include_reasoning": True, "top_k": 5}
        mode = "rag"
    # Auto-detect
    elif _is_provision_query(user_input):
        endpoint = "/api/cloud/provision-intent"
        payload = {"query": user_input}
        mode = "provision-intent"
    else:
        endpoint = "/api/models/v1/query"
        payload = {"query": user_input, "include_reasoning": True, "top_k": 5}
        mode = "rag"

    print(f"\n  [{mode}] -> {endpoint}")

    start = time.perf_counter()
    data = client.post(endpoint, payload)
    duration_ms = (time.perf_counter() - start) * 1000

    if "_error" in data:
        print(f"\n  Error: {data['_error']}")
        return

    print(f"  Responded in {duration_ms:.0f}ms")

    if mode == "provision-intent":
        _print_provision_response(data)
    else:
        _print_rag_response(data)


def interactive_mode(base_url: str = BASE_URL, initial_query: str = "") -> None:
    """
    Interactive chat mode - talk to VaLLM in your terminal.

    Usage:
        python -m app.tests.tests --interactive
        python -m app.tests.tests --interactive "deploy ec2 instance 20gb"

    The chat auto-detects provisioning vs. general queries.
    Prefix with /cloud or /rag to force a specific endpoint.
    """
    print("\n" + "=" * 78)
    print("  VaLLM Interactive Chat")
    print("=" * 78)
    print("  Ask anything about cloud infrastructure. I'll route your query")
    print("  to the right endpoint automatically.")
    print()
    print("  Tips:")
    print("    - Provisioning queries are auto-detected and sent to /api/cloud/provision-intent")
    print("    - General/troubleshooting queries are sent to /api/models/v1/query (RAG + reasoning)")
    print("    - Prefix with /cloud to force provision-intent")
    print("    - Prefix with /rag   to force RAG query")
    print()
    print("  Commands:")
    print("    /health  - Check service health")
    print("    /tests   - Run the 7 automated provisioning tests")
    print("    /quit    - Exit")
    print("=" * 78)

    client = VaLLMClient(base_url)

    print(f"\n  Connecting to {base_url}...", end=" ")
    if not client.health_check():
        print("FAILED")
        print("\n  Service not available. Start the app first:")
        print("    uvicorn app.app:app --host 0.0.0.0 --port 8745")
        return
    print("OK")

    turn = 0

    # If initial query was passed from CLI, process it first
    if initial_query:
        turn += 1
        print(f"\n{'=' * 78}")
        print(f"  [{turn}] You: {initial_query}")
        _process_chat_query(client, initial_query)

    while True:
        try:
            print()
            if turn == 0:
                prompt = "  What would you like to do? > "
            else:
                prompt = "  Do you have another question? > "

            user_input = input(prompt).strip()

            if not user_input:
                continue

            lower = user_input.lower()
            if lower in ("/quit", "/exit", "quit", "exit", "q"):
                print(f"\n  Session ended. {turn} question(s) answered. Goodbye!")
                break

            if lower == "/health":
                status = "OK" if client.health_check() else "Unhealthy"
                print(f"  Health: {status}")
                continue

            if lower == "/tests":
                print()
                run_tests(base_url)
                continue

            turn += 1
            print(f"\n{'=' * 78}")
            print(f"  [{turn}] You: {user_input}")
            _process_chat_query(client, user_input)

        except KeyboardInterrupt:
            print(f"\n\n  Session ended. {turn} question(s) answered. Goodbye!")
            break
        except EOFError:
            print(f"\n\n  Session ended. {turn} question(s) answered. Goodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VaLLM Model Accuracy Tests (7 tests, provisioning only) & Interactive Chat",
        epilog=(
            "Examples:\n"
            "  python -m app.tests.tests                                    # run 7 tests\n"
            "  python -m app.tests.tests --interactive                      # chat mode\n"
            '  python -m app.tests.tests --interactive "deploy ec2 20gb"    # chat with initial query\n'
            "  python -m app.tests.tests --url http://localhost:8745        # custom URL\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--interactive", nargs="?", const="", default=None,
                        metavar="QUERY",
                        help="Start interactive chat (optionally with an initial query)")
    parser.add_argument("--url", default=BASE_URL, help=f"Base URL (default: {BASE_URL})")
    args = parser.parse_args()

    if args.interactive is not None:
        interactive_mode(args.url, args.interactive)
    else:
        run_tests(args.url)
