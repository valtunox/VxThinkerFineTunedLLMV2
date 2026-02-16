"""
VaLLM ML Unit Tests - offline tests using mocks (no live server required).

Tests cover (provisioning only):
  - Health endpoint
  - V1 query endpoint (provision intent)
  - Terminal endpoint
  - Developer endpoint (Terraform code gen)
  - Cloud provision-intent endpoint (all 6 provisioning types)
  - Payload generation (_raw_to_golang_payload)
  - Non-provisioning classification (_classify_non_provisioning → "other")

Run:
    python -m pytest app/services/ai/ml/tests.py -v
    python -m unittest app.services.ai.ml.tests -v
"""

import unittest
import asyncio
import sys
import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

# Setup path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import app components
try:
    from app.services.ai.ml.routes import (
        QueryRequest,
        TerminalRequest,
        DeveloperRequest,
        query_endpoint,
        terminal_endpoint,
        developer_endpoint,
    )
    from app.services.ai.ml.cloud_routes import (
        _raw_to_golang_payload,
        _classify_non_provisioning,
        _is_deployment_result,
        ProvisionIntentRequest,
        provision_intent,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import app modules: {e}. "
        "Run from repo root: python -m pytest app/services/ai/ml/tests.py -v"
    ) from e

# Try importing the actual classes for mock specs
try:
    from app.services.ai.ml.reasoning import ReasoningEngine
    from app.services.ai.ml.embeddings import VectorStore
except ImportError:
    ReasoningEngine = None
    VectorStore = None

try:
    from app.app import health
except ImportError:
    health = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class DummyRequest:
    """Simulates a FastAPI Request with app.state."""

    def __init__(self, vector_store, reasoning_engine):
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                vector_store=vector_store,
                reasoning_engine=reasoning_engine,
            )
        )
        self.state = SimpleNamespace(request_id="test-request-id")


# ==========================================================================
# Test 1: Payload generation for all 6 provisioning intents
# ==========================================================================
class TestPayloadGeneration(unittest.TestCase):
    """Test _raw_to_golang_payload for each provisioning intent."""

    def test_provision_vm_payload(self):
        """provision_vm should produce instance_type, region, os, volume_size, cloud_provider."""
        raw = {
            "username": "joel",
            "workspace": "prod",
            "instance_name": "web-01",
            "instance_type": "t3.medium",
            "region": "us-west-2",
            "cloud_provider": "aws",
            "os": "ubuntu",
            "volume_size_gb": "50",
            "volume_type": "gp3",
            "environment": "production",
            "hostname": "web-01.prod.internal",
            "ssh_username": "ubuntu",
            "key_pair_name": "prod-key",
        }
        payload = _raw_to_golang_payload(raw, "provision_vm")
        self.assertEqual(payload["instance_type"], "t3.medium")
        self.assertEqual(payload["region"], "us-west-2")
        self.assertEqual(payload["cloud_provider"], "aws")
        self.assertEqual(payload["os"], "ubuntu")
        self.assertEqual(payload["volume_size"], 50)
        self.assertEqual(payload["volume_type"], "gp3")
        self.assertEqual(payload["environment"], "production")
        self.assertEqual(payload["username"], "joel")
        self.assertEqual(payload["workspace"], "prod")
        logger.info("PASS: provision_vm payload has all required fields")

    def test_provision_kubernetes_payload(self):
        """provision_kubernetes should produce cluster_name, node_count, node_type, kubernetes_version."""
        raw = {
            "username": "joel",
            "workspace": "prod",
            "cluster_name": "prod-eks",
            "node_count": "3",
            "node_type": "m5.large",
            "kubernetes_version": "1.29",
            "region": "us-east-1",
            "cloud_provider": "aws",
        }
        payload = _raw_to_golang_payload(raw, "provision_kubernetes")
        self.assertEqual(payload["cluster_name"], "prod-eks")
        self.assertEqual(payload["node_count"], 3)
        self.assertEqual(payload["node_type"], "m5.large")
        self.assertEqual(payload["kubernetes_version"], "1.29")
        self.assertEqual(payload["region"], "us-east-1")
        logger.info("PASS: provision_kubernetes payload correct")

    def test_provision_docker_payload(self):
        """provision_docker should produce docker_image, container_name, ports."""
        raw = {
            "username": "joel",
            "workspace": "dev",
            "docker_image": "nginx:latest",
            "docker_service": "nginx",
            "container_name": "web-nginx",
            "ports": "80:80",
        }
        payload = _raw_to_golang_payload(raw, "provision_docker")
        self.assertEqual(payload["docker_image"], "nginx:latest")
        self.assertEqual(payload["image"], "nginx:latest")
        self.assertEqual(payload["container_name"], "web-nginx")
        self.assertEqual(payload["ports"], "80:80")
        logger.info("PASS: provision_docker payload correct")

    def test_provision_database_payload(self):
        """provision_database should produce database_engine, database_name, port."""
        raw = {
            "username": "joel",
            "workspace": "prod",
            "hostname": "db.internal",
            "database_engine": "postgresql",
            "database_name": "analytics_db",
            "database_user": "admin",
            "postgres_version": "16",
            "port": "5432",
        }
        payload = _raw_to_golang_payload(raw, "provision_database")
        self.assertEqual(payload["database_engine"], "postgresql")
        self.assertEqual(payload["database_name"], "analytics_db")
        self.assertEqual(payload["database_user"], "admin")
        self.assertEqual(payload["postgres_version"], "16")
        self.assertEqual(payload["port"], 5432)
        logger.info("PASS: provision_database payload correct")

    def test_provision_fastapi_payload(self):
        """provision_fastapi should produce app_name, app_port, http_port."""
        raw = {
            "username": "joel",
            "workspace": "staging",
            "hostname": "api.staging.com",
            "app_name": "billing-api",
            "app_port": "8000",
            "http_port": "80",
            "ssh_username": "ubuntu",
            "key_pair_name": "staging-key",
        }
        payload = _raw_to_golang_payload(raw, "provision_fastapi")
        self.assertEqual(payload["app_name"], "billing-api")
        self.assertEqual(payload["app_port"], 8000)
        self.assertEqual(payload["http_port"], 80)
        self.assertEqual(payload["hostname"], "api.staging.com")
        logger.info("PASS: provision_fastapi payload correct")

    def test_provision_static_website_payload(self):
        """provision_static_website should produce server_name, http_port."""
        raw = {
            "username": "joel",
            "workspace": "prod",
            "hostname": "docs.example.com",
            "server_name": "docs.example.com",
            "http_port": "80",
            "ssh_username": "ubuntu",
            "key_pair_name": "web-key",
        }
        payload = _raw_to_golang_payload(raw, "provision_static_website")
        self.assertEqual(payload["server_name"], "docs.example.com")
        self.assertEqual(payload["http_port"], 80)
        self.assertEqual(payload["hostname"], "docs.example.com")
        logger.info("PASS: provision_static_website payload correct")


# ==========================================================================
# Test 2: Non-provisioning classification (provisioning-only API: always "other")
# ==========================================================================
class TestNonProvisioningClassification(unittest.TestCase):
    """Test _classify_non_provisioning returns 'other' for all non-provisioning queries."""

    def test_all_return_other(self):
        self.assertEqual(_classify_non_provisioning("Our API is down with 503 error"), "other")
        self.assertEqual(_classify_non_provisioning("AWS spend is too expensive"), "other")
        self.assertEqual(_classify_non_provisioning("Show me the billing for last month"), "other")
        self.assertEqual(_classify_non_provisioning("What time is it"), "other")
        self.assertEqual(_classify_non_provisioning("Deploy EC2 instance"), "other")  # classification only when no match


# ==========================================================================
# Test 3: Deployment result detection
# ==========================================================================
class TestDeploymentResultDetection(unittest.TestCase):
    """Test _is_deployment_result filter logic."""

    def test_valid_deployment_result(self):
        meta = {"raw": {"intent": "provision_vm", "prompt": "Deploy EC2"}}
        self.assertTrue(_is_deployment_result(meta))

    def test_missing_intent(self):
        meta = {"raw": {"prompt": "Deploy EC2"}}
        self.assertFalse(_is_deployment_result(meta))

    def test_empty_intent(self):
        meta = {"raw": {"intent": "", "prompt": "Deploy EC2"}}
        self.assertFalse(_is_deployment_result(meta))

    def test_no_raw(self):
        meta = {"type": "incident"}
        self.assertFalse(_is_deployment_result(meta))

    def test_empty_meta(self):
        self.assertFalse(_is_deployment_result({}))


# ==========================================================================
# Test 4: V1 Query endpoint with mocked reasoning engine
# ==========================================================================
class TestV1QueryEndpoint(unittest.TestCase):
    """Test the V1 /query endpoint with mocked dependencies."""

    @classmethod
    def setUpClass(cls):
        cls.mock_vector_store = MagicMock()
        cls.mock_reasoning_engine = MagicMock()
        cls.mock_vector_store.search = AsyncMock(return_value=[
            {
                "document": "Deploy t2.micro in us-east-1 with Ubuntu 22.04",
                "metadata": {"type": "deployment", "raw": {"intent": "provision_vm"}},
                "score": 0.88,
            },
            {
                "document": "Incident: VM quota exceeded in us-west-2",
                "metadata": {"type": "incident", "raw": {"severity": "high"}},
                "score": 0.72,
            },
        ])
        cls.request = DummyRequest(cls.mock_vector_store, cls.mock_reasoning_engine)

    def test_provision_query_returns_response_and_reasoning(self):
        """V1 query with provision intent should return response, reasoning, and context."""
        self.mock_reasoning_engine.reason = AsyncMock(return_value={
            "intent": "provision",
            "steps": [
                {"type": "analyze", "content": "Detected provision intent", "confidence": 0.95, "metadata": {}},
                {"type": "search", "content": "Found 2 deployment templates", "confidence": 0.88, "metadata": {}},
            ],
            "final_answer": "Deploy a t2.micro instance in us-east-1 with Ubuntu 22.04 and 30GB gp3 volume.",
            "confidence": 0.92,
            "context_used": "2 deployment templates",
        })
        payload = QueryRequest(
            query="Deploy a small EC2 instance with 30GB disk in us-east-1",
            include_reasoning=True,
            top_k=5,
        )
        data = run_async(query_endpoint(payload, self.request))
        self.assertIn("response", data)
        self.assertIn("reasoning", data)
        self.assertIn("context", data)
        self.assertEqual(data["reasoning"]["intent"], "provision")
        self.assertGreaterEqual(data["reasoning"]["confidence"], 0.5)
        self.assertEqual(len(data["context"]), 2)
        logger.info("PASS: V1 provision query returned correct structure")

# ==========================================================================
# Test 5: Terminal endpoint
# ==========================================================================
class TestTerminalEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_vector_store = MagicMock()
        cls.mock_reasoning_engine = MagicMock()
        cls.mock_vector_store.search = AsyncMock(return_value=[
            {
                "document": "kubectl get pods returns pod status across namespaces",
                "metadata": {"type": "incident", "raw_data": {"title": "K8s troubleshooting"}},
                "score": 0.80,
            },
        ])
        cls.mock_reasoning_engine.reason = AsyncMock(return_value={
            "intent": "analyze",
            "steps": [],
            "final_answer": "This command lists all pods in all namespaces with their status.",
            "confidence": 0.95,
        })
        cls.request = DummyRequest(cls.mock_vector_store, cls.mock_reasoning_engine)

    def test_terminal_kubectl_command(self):
        """Terminal endpoint should analyze kubectl commands and return incidents + recommendations."""
        payload = TerminalRequest(
            command="kubectl get pods --all-namespaces",
            include_explanation=True,
        )
        data = run_async(terminal_endpoint(payload, self.request))
        self.assertIn("response", data)
        self.assertIn("incidents", data)
        self.assertIn("recommendations", data)
        logger.info("PASS: Terminal endpoint returned analysis with incidents and recommendations")


# ==========================================================================
# Test 6: Developer endpoint (Terraform code gen)
# ==========================================================================
class TestDeveloperEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_vector_store = MagicMock()
        cls.mock_reasoning_engine = MagicMock()
        cls.mock_vector_store.search = AsyncMock(return_value=[])
        cls.mock_reasoning_engine.reason = AsyncMock(return_value={
            "intent": "provision",
            "steps": [],
            "final_answer": "Here is the Terraform code for a production VPC.",
            "confidence": 0.90,
        })
        cls.request = DummyRequest(cls.mock_vector_store, cls.mock_reasoning_engine)

    def test_terraform_vpc_code_gen(self):
        """Developer endpoint should return code_examples with Terraform for VPC."""
        payload = DeveloperRequest(
            query="Generate Terraform for a production VPC with Flow Logs and KMS encryption",
            include_code=True,
        )
        data = run_async(developer_endpoint(payload, self.request))
        self.assertIn("code_examples", data)
        self.assertEqual(data["model"], "vallm-developer-v1")
        # The Terraform example should contain VPC-related resources
        code = data.get("code_examples", "")
        self.assertIn("aws_vpc", code)
        self.assertIn("flow_log", code.lower().replace("_", " ").replace("-", " ") or code)
        logger.info("PASS: Developer endpoint returned Terraform code with VPC resources")


# ==========================================================================
# Test 7: Cloud provision-intent endpoint (mocked vector store)
# ==========================================================================
class TestCloudProvisionIntentEndpoint(unittest.TestCase):
    """Test the /api/cloud/provision-intent endpoint with mocked search."""

    def _make_request(self, vector_store):
        req = MagicMock()
        req.app = SimpleNamespace(state=SimpleNamespace(vector_store=vector_store))
        return req

    def test_provision_vm_intent_detected(self):
        """When vector search returns a deployment match with provision_vm, endpoint should return it."""
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(return_value=[
            {
                "document": "Deploy t3.medium EC2 in us-west-2",
                "metadata": {
                    "raw": {
                        "intent": "provision_vm",
                        "prompt": "Deploy t3.medium EC2 in us-west-2",
                        "username": "joel",
                        "workspace": "prod",
                        "instance_type": "t3.medium",
                        "region": "us-west-2",
                        "cloud_provider": "aws",
                        "os": "ubuntu",
                        "volume_size_gb": "50",
                    }
                },
                "score": 0.85,
            },
        ])
        req = self._make_request(mock_vs)
        body = ProvisionIntentRequest(query="Deploy a t3.medium EC2 in us-west-2")
        data = run_async(provision_intent(body, req))
        self.assertEqual(data["query_type"], "provisioning")
        self.assertEqual(data["intent"], "provision_vm")
        self.assertGreaterEqual(data["confidence"], 0.2)
        self.assertIn("instance_type", data["payload"])
        self.assertEqual(data["payload"]["instance_type"], "t3.medium")
        logger.info("PASS: provision-intent detected provision_vm with correct payload")

    def test_non_provisioning_returns_other(self):
        """When no deployment matches found, return query_type other (provisioning-only)."""
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(return_value=[
            {
                "document": "Database connection pool exhausted",
                "metadata": {"type": "other"},
                "score": 0.70,
            },
        ])
        req = self._make_request(mock_vs)
        body = ProvisionIntentRequest(query="Our database has error 503 and is down")
        data = run_async(provision_intent(body, req))
        self.assertEqual(data["query_type"], "other")
        self.assertIsNone(data["intent"])
        self.assertIsNone(data["payload"])
        logger.info("PASS: provision-intent returned other when no deployment match")

    def test_low_confidence_falls_back(self):
        """When deployment match has low confidence (<0.2), classify as non-provisioning."""
        mock_vs = MagicMock()
        mock_vs.search = AsyncMock(return_value=[
            {
                "document": "Some deployment",
                "metadata": {"raw": {"intent": "provision_vm", "prompt": "deploy vm"}},
                "score": 0.10,
            },
        ])
        req = self._make_request(mock_vs)
        body = ProvisionIntentRequest(query="What is the cost of running t3.medium?")
        data = run_async(provision_intent(body, req))
        self.assertNotEqual(data["query_type"], "provisioning")
        self.assertIsNone(data["intent"])
        logger.info("PASS: Low confidence correctly fell back to non-provisioning")

    def test_empty_query(self):
        """Empty query should return query_type=other."""
        mock_vs = MagicMock()
        req = self._make_request(mock_vs)
        body = ProvisionIntentRequest(query="")
        data = run_async(provision_intent(body, req))
        self.assertEqual(data["query_type"], "other")
        self.assertIsNone(data["intent"])
        logger.info("PASS: Empty query returned query_type=other")


# ==========================================================================
# Test 8: Health endpoint
# ==========================================================================
class TestHealthEndpoint(unittest.TestCase):

    @unittest.skipIf(health is None, "Could not import health endpoint")
    def test_health_returns_healthy(self):
        result = run_async(health())
        self.assertEqual(result, {"status": "healthy"})
        logger.info("PASS: Health endpoint returned healthy")


# ==========================================================================
# Test 9: Payload defaults and edge cases
# ==========================================================================
class TestPayloadDefaults(unittest.TestCase):
    """Test that _raw_to_golang_payload handles missing/nan values with sensible defaults."""

    def test_vm_defaults_for_missing_fields(self):
        """Missing fields should get sensible defaults."""
        raw = {"username": "joel", "workspace": "dev"}
        payload = _raw_to_golang_payload(raw, "provision_vm")
        self.assertEqual(payload["instance_type"], "t2.micro")
        self.assertEqual(payload["region"], "us-east-1")
        self.assertEqual(payload["cloud_provider"], "aws")
        self.assertEqual(payload["os"], "ubuntu")
        self.assertEqual(payload["volume_size"], 30)
        self.assertEqual(payload["volume_type"], "gp3")
        self.assertEqual(payload["environment"], "dev")
        logger.info("PASS: VM payload defaults applied correctly")

    def test_nan_values_treated_as_empty(self):
        """NaN string values from CSV should be treated as empty."""
        raw = {
            "username": "joel",
            "workspace": "prod",
            "instance_type": "nan",
            "region": "NaN",
            "os": "NAN",
        }
        payload = _raw_to_golang_payload(raw, "provision_vm")
        self.assertEqual(payload["instance_type"], "t2.micro")  # default
        self.assertEqual(payload["region"], "us-east-1")  # default
        self.assertEqual(payload["os"], "ubuntu")  # default
        logger.info("PASS: NaN values correctly treated as empty, defaults applied")

    def test_kubernetes_defaults(self):
        """Kubernetes defaults should be node_count=2, node_type=t3.medium."""
        raw = {"username": "joel", "workspace": "dev"}
        payload = _raw_to_golang_payload(raw, "provision_kubernetes")
        self.assertEqual(payload["node_count"], 2)
        self.assertEqual(payload["node_type"], "t3.medium")
        self.assertEqual(payload["region"], "us-east-1")
        self.assertEqual(payload["cloud_provider"], "aws")
        logger.info("PASS: Kubernetes defaults applied correctly")

    def test_database_defaults(self):
        """Database defaults should be engine=postgres, port=5432."""
        raw = {"username": "joel", "workspace": "dev"}
        payload = _raw_to_golang_payload(raw, "provision_database")
        self.assertEqual(payload["database_engine"], "postgres")
        self.assertEqual(payload["port"], 5432)
        logger.info("PASS: Database defaults applied correctly")


if __name__ == "__main__":
    unittest.main()
