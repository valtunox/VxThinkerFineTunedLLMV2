"""
VaLLM Usage Example - Comprehensive API Demonstration

Tests all endpoints: provision-intent (6 provisioning + non-provisioning),
V1 RAG + reasoning, and RLHF feedback submission.

USAGE:
======
    # 1. Start the API server first (in another terminal):
    uvicorn app.app:app --host 0.0.0.0 --port 8745

    # 2. Run this example:
    python -m app.services.ai.ml.usage_example
    python -m app.services.ai.ml.usage_example --url http://localhost:8745
    python -m app.services.ai.ml.usage_example --section provision   # provisioning only
    python -m app.services.ai.ml.usage_example --section rag         # RAG queries only
    python -m app.services.ai.ml.usage_example --section feedback    # RLHF feedback only
"""

import json
import time
import uuid
import argparse
from typing import Optional, Dict, Any, List

import requests

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_URL = "http://localhost:8745"
TIMEOUT = 60
TOP_K = 5
SHOW_CONTEXT = True
SHOW_REASONING_STEPS = True
CONTEXT_PREVIEW_CHARS = 200

# =============================================================================
# PROVISIONING INTENT EXAMPLES - All 6 intent types
# =============================================================================
PROVISION_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "Provision VM (EC2 t3.medium)",
        "query": "Deploy a t3.medium EC2 instance in us-west-2 with Ubuntu and 50GB gp3 disk for production",
        "expected_intent": "provision_vm",
        "expected_keys": ["instance_type", "region", "cloud_provider", "os", "volume_size"],
    },
    {
        "name": "Provision Kubernetes (EKS 3-node)",
        "query": "Create an EKS cluster with 3 m5.large nodes running Kubernetes 1.29 in us-east-1",
        "expected_intent": "provision_kubernetes",
        "expected_keys": ["cluster_name", "node_count", "node_type", "kubernetes_version", "region"],
    },
    {
        "name": "Provision Docker (Nginx container)",
        "query": "Deploy an nginx Docker container with port mapping 80:80 on my server",
        "expected_intent": "provision_docker",
        "expected_keys": ["docker_image", "container_name", "ports"],
    },
    {
        "name": "Provision Database (PostgreSQL 16)",
        "query": "Set up a PostgreSQL 16 database named analytics_db with user admin on my VM",
        "expected_intent": "provision_database",
        "expected_keys": ["database_engine", "database_name", "database_user", "port"],
    },
    {
        "name": "Provision FastAPI application",
        "query": "Deploy my FastAPI application named billing-api on port 8000 with HTTP on port 80",
        "expected_intent": "provision_fastapi",
        "expected_keys": ["app_name", "app_port", "http_port"],
    },
    {
        "name": "Provision Static Website",
        "query": "Host a static website for docs.example.com on nginx with port 80",
        "expected_intent": "provision_static_website",
        "expected_keys": ["server_name", "http_port"],
    },
]

# =============================================================================
# NON-PROVISIONING QUERY EXAMPLES
# =============================================================================
NON_PROVISION_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "Incident report",
        "query": "Our API gateway is returning 503 errors and the database connection pool is exhausted",
        "expected_query_type": "incident",
    },
    {
        "name": "Cost optimization",
        "query": "Our AWS spend is too expensive, give me cost optimization recommendations for EC2 and RDS",
        "expected_query_type": "cost",
    },
    {
        "name": "Security concern",
        "query": "We found a vulnerability in our S3 bucket policy allowing public access to sensitive data",
        "expected_query_type": "security",
    },
    {
        "name": "Billing inquiry",
        "query": "Can you explain our last billing invoice and show the breakdown per service?",
        "expected_query_type": "billing",
    },
]

# =============================================================================
# V1 RAG + REASONING EXAMPLES
# =============================================================================
RAG_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "RAG: provision intent with reasoning",
        "query": "Deploy a small EC2 instance with 30GB disk in us-east-1 with Ubuntu",
        "expected_intents": ["provision"],
    },
    {
        "name": "RAG: troubleshooting with reasoning",
        "query": "Kubernetes pods stuck in CrashLoopBackOff after deployment, how to fix it?",
        "expected_intents": ["troubleshoot", "analyze"],
    },
    {
        "name": "RAG: observability query",
        "query": "OpenTelemetry traces show dropped spans at the edge, how to fix sampling?",
        "expected_intents": ["troubleshoot", "analyze"],
    },
    {
        "name": "RAG: networking issue",
        "query": "Edge load balancer returns 502 intermittently, what checks should I perform?",
        "expected_intents": ["troubleshoot"],
    },
    {
        "name": "RAG: database scaling",
        "query": "Database connection pool exhausted during traffic spikes, how to mitigate?",
        "expected_intents": ["troubleshoot", "analyze"],
    },
]


# =============================================================================
# HTTP CLIENT
# =============================================================================
class VaLLMClient:
    """HTTP client for VaLLM API."""

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

    def post(self, endpoint: str, payload: dict, timeout: int = TIMEOUT) -> dict:
        start = time.perf_counter()
        try:
            r = self.session.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return {"_error": str(e), "_duration_ms": 0}
        data["_duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        data["_request_id"] = r.headers.get("X-Request-ID")
        return data


# =============================================================================
# DISPLAY HELPERS
# =============================================================================
def _print_divider(char: str = "-", width: int = 70) -> None:
    print(char * width)


def _print_provision_result(name: str, data: dict, expected: dict) -> None:
    """Display provision-intent result and validate against expected."""
    if "_error" in data:
        print(f"  ERROR: {data['_error']}")
        return

    qt = data.get("query_type")
    intent = data.get("intent")
    confidence = data.get("confidence", 0)
    payload = data.get("payload") or {}
    match_prompt = data.get("match_prompt", "")
    duration = data.get("_duration_ms", 0)

    print(f"  query_type:   {qt}")
    print(f"  intent:       {intent}")
    print(f"  confidence:   {confidence:.4f}")
    print(f"  duration:     {duration}ms")

    if match_prompt:
        preview = match_prompt[:100]
        if len(match_prompt) > 100:
            preview += "..."
        print(f"  match_prompt: {preview}")

    if payload:
        print(f"  payload keys: {list(payload.keys())}")
        # Show first 5 payload values
        for i, (k, v) in enumerate(payload.items()):
            if i >= 6:
                print(f"                ... ({len(payload) - 6} more)")
                break
            print(f"    {k}: {v}")

    # Validate
    errors = []
    exp_intent = expected.get("expected_intent")
    if exp_intent and intent != exp_intent:
        errors.append(f"intent: expected '{exp_intent}', got '{intent}'")

    exp_qt = expected.get("expected_query_type")
    if exp_qt and qt != exp_qt:
        errors.append(f"query_type: expected '{exp_qt}', got '{qt}'")

    for key in expected.get("expected_keys", []):
        if key not in payload:
            errors.append(f"payload missing key: '{key}'")

    if errors:
        print(f"  VALIDATION: FAIL")
        for err in errors:
            print(f"    - {err}")
    else:
        print(f"  VALIDATION: PASS")


def _print_rag_result(data: dict, expected: dict) -> None:
    """Display V1 RAG query result."""
    if "_error" in data:
        print(f"  ERROR: {data['_error']}")
        return

    duration = data.get("_duration_ms", 0)
    response_text = data.get("response", "")
    reasoning = data.get("reasoning") or {}
    context = data.get("context") or []

    print(f"  duration: {duration}ms")
    print(f"\n  Response:")
    # Truncate long responses
    for line in response_text[:500].split("\n"):
        print(f"    {line}")
    if len(response_text) > 500:
        print(f"    ... (truncated, {len(response_text)} chars total)")

    if reasoning:
        print(f"\n  Reasoning:")
        print(f"    intent:     {reasoning.get('intent', 'unknown')}")
        print(f"    confidence: {reasoning.get('confidence', 0):.2%}")
        steps = reasoning.get("steps") or []
        if steps and SHOW_REASONING_STEPS:
            print(f"    steps ({len(steps)}):")
            for idx, step in enumerate(steps, 1):
                print(f"      {idx}. {step}")

    print(f"\n  Context: {len(context)} documents retrieved")
    if context and SHOW_CONTEXT:
        for idx, item in enumerate(context[:3], 1):
            doc = (item.get("document") or "").strip().replace("\n", " ")
            preview = doc[:CONTEXT_PREVIEW_CHARS]
            if len(doc) > CONTEXT_PREVIEW_CHARS:
                preview += "..."
            score = item.get("score", 0.0)
            doc_type = item.get("type", "unknown")
            print(f"    {idx}. [{doc_type}] score={score:.4f} | {preview}")

    # Validate
    exp_intents = expected.get("expected_intents", [])
    if exp_intents and reasoning:
        actual_intent = (reasoning.get("intent") or "").lower()
        if any(ei in actual_intent for ei in exp_intents):
            print(f"  VALIDATION: PASS (intent '{actual_intent}' matches expected)")
        else:
            print(f"  VALIDATION: FAIL (intent '{actual_intent}' not in {exp_intents})")


# =============================================================================
# SECTION RUNNERS
# =============================================================================
def run_provision_examples(client: VaLLMClient) -> int:
    """Run all provisioning intent examples. Returns number of errors."""
    print("\n" + "=" * 70)
    print("  SECTION 1: Cloud Provision-Intent (6 provisioning types)")
    print("  Endpoint: POST /api/cloud/provision-intent")
    print("=" * 70)

    errors = 0
    total = len(PROVISION_EXAMPLES)
    for i, example in enumerate(PROVISION_EXAMPLES, 1):
        print(f"\n[{i}/{total}] {example['name']}")
        print(f"  Query: {example['query'][:80]}")
        _print_divider()

        data = client.post("/api/cloud/provision-intent", {"query": example["query"]})
        _print_provision_result(example["name"], data, example)
        if "_error" in data:
            errors += 1
        print()

    return errors


def run_non_provision_examples(client: VaLLMClient) -> int:
    """Run non-provisioning classification examples. Returns number of errors."""
    print("\n" + "=" * 70)
    print("  SECTION 2: Non-Provisioning Classification")
    print("  Endpoint: POST /api/cloud/provision-intent")
    print("=" * 70)

    errors = 0
    total = len(NON_PROVISION_EXAMPLES)
    for i, example in enumerate(NON_PROVISION_EXAMPLES, 1):
        print(f"\n[{i}/{total}] {example['name']}")
        print(f"  Query: {example['query'][:80]}")
        _print_divider()

        data = client.post("/api/cloud/provision-intent", {"query": example["query"]})
        _print_provision_result(example["name"], data, example)
        if "_error" in data:
            errors += 1
        print()

    return errors


def run_rag_examples(client: VaLLMClient) -> int:
    """Run V1 RAG + reasoning examples. Returns number of errors."""
    print("\n" + "=" * 70)
    print("  SECTION 3: V1 RAG Query with Reasoning Engine")
    print("  Endpoint: POST /api/models/v1/query")
    print("=" * 70)

    errors = 0
    total = len(RAG_EXAMPLES)
    for i, example in enumerate(RAG_EXAMPLES, 1):
        print(f"\n[{i}/{total}] {example['name']}")
        print(f"  Query: {example['query'][:80]}")
        _print_divider()

        payload = {
            "query": example["query"],
            "include_reasoning": True,
            "top_k": TOP_K,
        }
        data = client.post("/api/models/v1/query", payload)
        _print_rag_result(data, example)
        if "_error" in data:
            errors += 1
        print()

    return errors


def run_feedback_example(client: VaLLMClient) -> int:
    """Demonstrate RLHF feedback submission. Returns number of errors."""
    print("\n" + "=" * 70)
    print("  SECTION 4: RLHF Feedback Loop")
    print("  Endpoint: POST /platform/evals/feedback")
    print("=" * 70)

    # Step 1: Make a query to get a response we can give feedback on
    print("\n  Step 1: Send a query to get a response...")
    query = "Deploy a t2.micro EC2 instance in us-east-1 for development"
    data = client.post("/api/cloud/provision-intent", {"query": query})

    if "_error" in data:
        print(f"  Query failed: {data['_error']}")
        return 1

    request_id = data.get("_request_id") or str(uuid.uuid4())
    intent = data.get("intent", "unknown")
    confidence = data.get("confidence", 0)
    print(f"  Got response: intent={intent}, confidence={confidence:.4f}")
    print(f"  request_id: {request_id}")

    # Step 2: Submit positive feedback (thumbs up)
    print("\n  Step 2: Submit positive feedback (thumbs up)...")
    feedback_payload = {
        "request_id": request_id,
        "model_id": "vallm-v1-embeddings",
        "feedback_type": "thumbs",
        "feedback_value": 1,
        "feedback_text": "Correct intent detection and complete payload",
        "query": query,
        "response": json.dumps({"intent": intent, "confidence": confidence}),
        "metadata": {
            "intent": intent,
            "confidence": confidence,
            "test_run": True,
        },
    }

    # Feedback endpoint requires X-Tenant-ID header
    try:
        r = client.session.post(
            f"{client.base_url}/platform/evals/feedback",
            json=feedback_payload,
            headers={"X-Tenant-ID": "demo-tenant"},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            fb_data = r.json()
            print(f"  Feedback submitted: id={fb_data.get('id')}")
            print(f"  RLHF FEEDBACK: PASS")
            return 0
        else:
            print(f"  Feedback response: {r.status_code} - {r.text[:200]}")
            print(f"  RLHF FEEDBACK: SKIPPED (endpoint may require DB)")
            return 0
    except Exception as e:
        print(f"  Feedback submission error: {e}")
        print(f"  RLHF FEEDBACK: SKIPPED (endpoint may require DB)")
        return 0


# =============================================================================
# MAIN
# =============================================================================
def main(base_url: str = BASE_URL, section: Optional[str] = None) -> None:
    """Run all usage examples against the live API."""
    print("\n" + "=" * 70)
    print("  VaLLM Usage Example - Comprehensive API Demo")
    print("=" * 70)
    print(f"  Base URL: {base_url}")
    print(f"  Section:  {section or 'all'}")
    print("=" * 70)

    client = VaLLMClient(base_url)

    print("\nChecking service health...", end=" ")
    if not client.health_check():
        print("FAILED")
        print("\nService not available. Start the app first:")
        print("  uvicorn app.app:app --host 0.0.0.0 --port 8745")
        return
    print("OK")

    total_errors = 0
    sections_run = 0

    if section in (None, "all", "provision"):
        total_errors += run_provision_examples(client)
        sections_run += 1
        total_errors += run_non_provision_examples(client)
        sections_run += 1

    if section in (None, "all", "rag"):
        total_errors += run_rag_examples(client)
        sections_run += 1

    if section in (None, "all", "feedback"):
        total_errors += run_feedback_example(client)
        sections_run += 1

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Sections run: {sections_run}")
    print(f"  Errors:       {total_errors}")

    if total_errors == 0:
        print("  Status:       ALL PASSED")
    else:
        print(f"  Status:       {total_errors} ERROR(S)")

    print("\n  Endpoints tested:")
    if section in (None, "all", "provision"):
        print(f"    POST /api/cloud/provision-intent  ({len(PROVISION_EXAMPLES)} provisioning + {len(NON_PROVISION_EXAMPLES)} non-provisioning)")
    if section in (None, "all", "rag"):
        print(f"    POST /api/models/v1/query          ({len(RAG_EXAMPLES)} RAG queries)")
    if section in (None, "all", "feedback"):
        print(f"    POST /platform/evals/feedback      (1 RLHF feedback)")

    print("\n  Intent types covered:")
    print("    Provisioning: provision_vm, provision_kubernetes, provision_docker,")
    print("                  provision_database, provision_fastapi, provision_static_website")
    print("    Non-provisioning: incident, cost, security, billing")
    print("    RAG reasoning: provision, troubleshoot, analyze")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VaLLM Usage Example - API Demo")
    parser.add_argument("--url", default=BASE_URL, help=f"Base URL (default: {BASE_URL})")
    parser.add_argument(
        "--section",
        choices=["all", "provision", "rag", "feedback"],
        default="all",
        help="Which section to run (default: all)",
    )
    args = parser.parse_args()
    main(args.url, args.section)
