"""
VaLLM Interactive Chat & RAG Tests
====================================

Author: Joel Otepa Wembo
https://joelwembo.com

DESCRIPTION
===========
Multi-turn conversational test suite with three modes:

    MODE 1 - Automated RAG Conversations (scored, needs server)
    ============================================================
    Runs pre-defined multi-turn queries through the RAG pipeline
    (search FAISS -> build context -> generate with LLM).  Scores
    each response for relevance, coherence, and correctness.

    MODE 2 - Live Interactive Chat (terminal REPL, needs server)
    ============================================================
    Open a chat session and talk to VaLLM freely.  Uses the full
    RAG pipeline: search + generate.  Maintains conversation history.

    MODE 3 - Offline Interactive Chat (no server, uses embedding service)
    =====================================================================
    Loads the FAISS index directly and answers from indexed knowledge.
    No server or API keys needed - pure local inference.

    MODE 4 - Direct LLM Generation Tests (no server, loads model)
    ==============================================================
    Loads the fine-tuned model from disk and tests text generation
    quality with cloud/DevOps prompts.

USAGE
=====
    # Start the app first (for Modes 1 & 2):
    python -m app.app   (port 8747)

    # Run automated RAG tests (needs server)
    python -m app.tests.tests_interactive_chat

    # Run with pytest (scored tests only)
    pytest app/tests/tests_interactive_chat.py -v -s

    # Start live interactive chat (needs server)
    python -m app.tests.tests_interactive_chat --interactive

    # Offline chat (no server needed, uses FAISS directly)
    python -m app.tests.tests_interactive_chat --offline

    # Direct LLM generation tests (no server)
    python -m app.tests.tests_interactive_chat --generate

    # Custom server URL
    python -m app.tests.tests_interactive_chat --url http://localhost:8747
"""

import atexit
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — project root and app dir
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP_DIR = _PROJECT_ROOT / "app"
for _p in [str(_PROJECT_ROOT), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import requests

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

BASE_URL = os.environ.get("VALLM_TEST_URL", "http://localhost:8747")


# ============================================================================
# SCORECARD
# ============================================================================

_SCORECARD: Dict[str, Dict[str, Any]] = {}
_TEST_COUNTER = 0


def _record_score(test_name: str, score: int, details: str = ""):
    global _TEST_COUNTER
    _TEST_COUNTER += 1
    score = max(1, min(10, score))
    bar = "#" * score + "." * (10 - score)
    label = (
        "PERFECT" if score == 10 else
        "EXCELLENT" if score >= 8 else
        "GOOD" if score >= 6 else
        "FAIR" if score >= 4 else
        "POOR"
    )
    _SCORECARD[f"Test {_TEST_COUNTER}"] = {
        "name": test_name, "score": score, "label": label, "details": details,
    }
    print(f"\n  {'=' * 65}")
    print(f"  [Test {_TEST_COUNTER}] {test_name}")
    print(f"  Score: {score}/10 [{bar}] {label}")
    if details:
        print(f"  Details: {details}")
    print(f"  {'=' * 65}")


def _print_final_scorecard():
    if not _SCORECARD:
        return
    total = sum(v["score"] for v in _SCORECARD.values())
    count = len(_SCORECARD)
    avg = total / count if count else 0
    print("\n\n" + "=" * 70)
    print("  VALLM INTERACTIVE CHAT - FINAL SCORECARD")
    print("=" * 70)
    for key, val in _SCORECARD.items():
        bar = "#" * val["score"] + "." * (10 - val["score"])
        print(f"  {key:>8} | {val['score']:>2}/10 [{bar}] {val['label']:<10} | {val['name']}")
    print("-" * 70)
    print(f"  {'TOTAL':>8} | {total}/{count * 10}  Average: {avg:.1f}/10")
    overall = (
        "EXCELLENT" if avg >= 8 else "GOOD" if avg >= 6 else
        "NEEDS IMPROVEMENT" if avg >= 4 else "CRITICAL ISSUES"
    )
    print(f"  Overall Assessment: {overall}")
    print("=" * 70)


atexit.register(_print_final_scorecard)


# ============================================================================
# RAG CHAT CLIENT - uses actual /search + /generate endpoints
# ============================================================================

class VaLLMChatClient:
    """
    Stateful RAG chat client that maintains conversation history.

    Pipeline:
        1. Search FAISS via POST /search -> get relevant documents
        2. Build augmented prompt with context + history + user question
        3. Generate answer via POST /generate -> return LLM response
    """

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.session_id = str(uuid.uuid4())
        self.history: List[Dict[str, str]] = []
        self.turn = 0

    def health_check(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def search(self, query: str, top_k: int = 5, timeout: float = 30.0) -> List[Dict]:
        """Search FAISS index via /search endpoint."""
        try:
            r = self.session.post(f"{self.base_url}/search", json={
                "query": query,
                "top_k": top_k,
            }, timeout=timeout)
            if r.status_code == 200:
                return r.json().get("results", [])
            return []
        except Exception:
            return []

    def generate(self, prompt: str, max_new_tokens: int = 200,
                 temperature: float = 0.7, timeout: float = 60.0) -> Dict[str, Any]:
        """Generate text via /generate endpoint."""
        try:
            r = self.session.post(f"{self.base_url}/generate", json={
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.9,
            }, timeout=timeout)
            data = r.json()
            return {
                "status_code": r.status_code,
                "response": data.get("response", data.get("text", "")),
                "model_loaded": data.get("model_loaded", False),
                "device": data.get("device", "unknown"),
            }
        except Exception as e:
            return {"status_code": 0, "response": f"Error: {e}", "model_loaded": False}

    def _build_rag_prompt(self, user_message: str, context_docs: List[Dict]) -> str:
        """Build an augmented prompt with retrieved context and conversation history."""
        parts = []

        # System instruction
        parts.append(
            "You are VaLLM, an AI assistant specializing in cloud infrastructure, "
            "DevOps, SRE, networking, and IT operations. Use the provided context "
            "to answer the user's question accurately and concisely."
        )

        # Retrieved context
        if context_docs:
            parts.append("\n--- Context from knowledge base ---")
            for i, doc in enumerate(context_docs[:5], 1):
                text = doc.get("text", "")[:500]
                score = doc.get("score", 0)
                parts.append(f"[{i}] (score: {score:.3f}) {text}")
            parts.append("--- End context ---\n")

        # Recent conversation history
        if self.history:
            parts.append("Recent conversation:")
            for entry in self.history[-4:]:
                role = entry["role"]
                content = entry["content"][:200]
                parts.append(f"  {role}: {content}")
            parts.append("")

        # Current question
        parts.append(f"User: {user_message}")
        parts.append("\nAssistant:")

        return "\n".join(parts)

    def chat(self, message: str, mode: str = "rag", timeout: float = 60.0) -> Dict[str, Any]:
        """
        Send a message and get a response.

        Modes:
            "rag"      - Search + Generate (full RAG pipeline)
            "search"   - Search only (return relevant documents)
            "generate" - Generate only (raw LLM, no retrieval)
        """
        self.turn += 1
        self.history.append({"role": "User", "content": message})
        start = time.perf_counter()

        if mode == "search":
            # Search-only mode
            results = self.search(message, top_k=5, timeout=timeout)
            response_text = ""
            if results:
                response_text = "\n\n".join(
                    f"[{i+1}] (score: {r.get('score', 0):.3f}) {r.get('text', '')[:300]}"
                    for i, r in enumerate(results[:5])
                )
            else:
                response_text = "No relevant documents found."

            result = {
                "mode": "search",
                "status_code": 200 if results else 404,
                "response": response_text,
                "search_results": results,
                "result_count": len(results),
            }

        elif mode == "generate":
            # Direct generation (no retrieval)
            context_prompt = self._build_rag_prompt(message, [])
            gen_result = self.generate(context_prompt, timeout=timeout)
            result = {
                "mode": "generate",
                **gen_result,
            }

        else:
            # Full RAG: search -> augment -> generate
            search_results = self.search(message, top_k=5, timeout=timeout)
            context_prompt = self._build_rag_prompt(message, search_results)
            gen_result = self.generate(context_prompt, timeout=timeout)
            result = {
                "mode": "rag",
                "search_results": search_results,
                "context_docs": len(search_results),
                **gen_result,
            }

        elapsed_ms = (time.perf_counter() - start) * 1000
        result["elapsed_ms"] = elapsed_ms
        result["turn"] = self.turn

        response_text = result.get("response", "")
        self.history.append({"role": "Assistant", "content": response_text[:500]})

        return result

    def reset(self):
        """Reset conversation state."""
        self.history.clear()
        self.turn = 0
        self.session_id = str(uuid.uuid4())


# ============================================================================
# CONVERSATION DEFINITIONS - multi-turn scored RAG dialogues
# ============================================================================

CONVERSATIONS: List[Dict[str, Any]] = [
    {
        "id": "CONV-1",
        "name": "Kubernetes Q&A (RAG)",
        "description": "Multi-turn Kubernetes questions answered via search + generate",
        "turns": [
            {
                "user": "What is Kubernetes and why do companies use it?",
                "mode": "rag",
                "check_keywords": ["kubernetes", "container", "orchestr"],
            },
            {
                "user": "How do I scale a deployment in Kubernetes?",
                "mode": "rag",
                "check_keywords": ["scale", "replica", "hpa"],
            },
            {
                "user": "What is the difference between a Deployment and a StatefulSet?",
                "mode": "rag",
                "check_keywords": ["stateful", "deployment"],
            },
        ],
    },
    {
        "id": "CONV-2",
        "name": "Terraform Knowledge (RAG)",
        "description": "Infrastructure as Code questions through RAG",
        "turns": [
            {
                "user": "How do I initialize a Terraform project?",
                "mode": "rag",
                "check_keywords": ["terraform", "init"],
            },
            {
                "user": "What is Terraform state and why is it important?",
                "mode": "rag",
                "check_keywords": ["state", "terraform"],
            },
            {
                "user": "How do I manage secrets in Terraform?",
                "mode": "rag",
                "check_keywords": ["secret"],
            },
        ],
    },
    {
        "id": "CONV-3",
        "name": "SRE Troubleshooting (RAG)",
        "description": "Incident troubleshooting dialogue",
        "turns": [
            {
                "user": "My pods keep crashing with OOM errors. What should I check?",
                "mode": "rag",
                "check_keywords": ["memory", "resource", "limit"],
            },
            {
                "user": "How do I set up error budgets for an SLO?",
                "mode": "rag",
                "check_keywords": ["error", "budget", "slo"],
            },
            {
                "user": "What is the circuit breaker pattern and when should I use it?",
                "mode": "rag",
                "check_keywords": ["circuit", "breaker", "fail"],
            },
        ],
    },
    {
        "id": "CONV-4",
        "name": "Cloud Billing (RAG)",
        "description": "Cost optimization questions",
        "turns": [
            {
                "user": "How is EC2 billed on AWS?",
                "mode": "rag",
                "check_keywords": ["ec2", "billing", "hour"],
            },
            {
                "user": "What are Savings Plans vs Reserved Instances?",
                "mode": "rag",
                "check_keywords": ["reserved", "saving"],
            },
        ],
    },
    {
        "id": "CONV-5",
        "name": "Search-Only Mode",
        "description": "Test pure FAISS search without generation",
        "turns": [
            {
                "user": "How do I configure a VPC on AWS?",
                "mode": "search",
                "check_keywords": ["vpc", "subnet", "cidr"],
                "min_results": 1,
            },
            {
                "user": "What is a service mesh?",
                "mode": "search",
                "check_keywords": ["mesh", "istio", "service"],
                "min_results": 1,
            },
        ],
    },
    {
        "id": "CONV-6",
        "name": "Direct LLM Generation",
        "description": "Test /generate endpoint directly (no RAG)",
        "turns": [
            {
                "user": "Explain how Docker containers work in simple terms.",
                "mode": "generate",
                "min_word_count": 5,
            },
            {
                "user": "What is CI/CD and why is it important?",
                "mode": "generate",
                "min_word_count": 5,
            },
        ],
    },
    {
        "id": "CONV-7",
        "name": "Networking Deep Dive (RAG)",
        "description": "Multi-turn networking questions",
        "turns": [
            {
                "user": "What is the difference between a security group and a NACL?",
                "mode": "rag",
                "check_keywords": ["security", "group", "nacl"],
            },
            {
                "user": "How do I set up VPC peering?",
                "mode": "rag",
                "check_keywords": ["peering", "vpc"],
            },
            {
                "user": "What is a CDN and how does it work?",
                "mode": "rag",
                "check_keywords": ["cdn", "content", "edge"],
            },
        ],
    },
    {
        "id": "CONV-8",
        "name": "Multi-Domain Cross-Search",
        "description": "Questions spanning multiple knowledge domains",
        "turns": [
            {
                "user": "How do I monitor Kubernetes clusters with Prometheus?",
                "mode": "rag",
                "check_keywords": ["prometheus", "monitor", "kubernetes"],
            },
            {
                "user": "What are the four golden signals in SRE?",
                "mode": "rag",
                "check_keywords": ["latency", "traffic", "error", "saturation"],
            },
        ],
    },
]


# ============================================================================
# LLM GENERATION TEST PROMPTS (no server needed)
# ============================================================================

GENERATION_PROMPTS = [
    {
        "id": "G1",
        "name": "Cloud Architecture",
        "prompt": "Explain a three-tier architecture on AWS with VPC, load balancer, EC2, and RDS.",
        "max_tokens": 150,
        "keywords": ["vpc", "load", "ec2", "rds", "tier"],
    },
    {
        "id": "G2",
        "name": "K8s Troubleshooting",
        "prompt": "A Kubernetes pod is in CrashLoopBackOff. What are the common causes and how to debug?",
        "max_tokens": 150,
        "keywords": ["pod", "crash", "log", "debug", "container"],
    },
    {
        "id": "G3",
        "name": "Terraform Practices",
        "prompt": "What are best practices for organizing Terraform code in a large project?",
        "max_tokens": 150,
        "keywords": ["module", "state", "terraform", "variable"],
    },
    {
        "id": "G4",
        "name": "Docker Security",
        "prompt": "How to secure a Docker container in production? List the key security measures.",
        "max_tokens": 150,
        "keywords": ["docker", "security", "image", "root"],
    },
    {
        "id": "G5",
        "name": "CI/CD Pipeline",
        "prompt": "Design a CI/CD pipeline for a FastAPI application using GitHub Actions and AWS ECS.",
        "max_tokens": 200,
        "keywords": ["github", "action", "deploy", "test", "pipeline"],
    },
    {
        "id": "G6",
        "name": "Cost Optimization",
        "prompt": "What are the top 5 ways to reduce AWS cloud costs for a mid-size company?",
        "max_tokens": 150,
        "keywords": ["cost", "reserved", "spot", "right-siz"],
    },
    {
        "id": "G7",
        "name": "Incident Response",
        "prompt": "An RDS database shows high CPU and slow queries. What is the incident response process?",
        "max_tokens": 150,
        "keywords": ["cpu", "query", "rds", "database", "performance"],
    },
    {
        "id": "G8",
        "name": "SRE Error Budgets",
        "prompt": "Explain SRE error budgets. How do you calculate them and what happens when the budget is exhausted?",
        "max_tokens": 180,
        "keywords": ["error", "budget", "slo", "reliability"],
    },
]


# ============================================================================
# SCORING HELPERS
# ============================================================================

def score_turn(turn_def: dict, result: dict) -> int:
    """Score a single conversation turn (1-10)."""
    score = 1

    status = result.get("status_code", 0)
    if status != 200 and status != 0:
        return 2

    response = result.get("response", "")
    if not response or len(response.strip()) < 5:
        return 3

    score = 5

    # Speed bonus
    if result.get("elapsed_ms", 99999) < 5000:
        score += 1
    if result.get("elapsed_ms", 99999) < 2000:
        score += 1

    # Keyword match
    keywords = turn_def.get("check_keywords", [])
    if keywords:
        response_lower = response.lower()
        matched = [kw for kw in keywords if kw in response_lower]
        if len(matched) >= len(keywords):
            score += 2
        elif matched:
            score += 1

    # Search results count
    min_results = turn_def.get("min_results", 0)
    if min_results:
        actual = result.get("result_count", 0)
        if actual >= min_results:
            score += 1

    # Word count
    min_words = turn_def.get("min_word_count", 0)
    if min_words:
        word_count = len(response.split())
        if word_count >= min_words:
            score += 1

    # Context docs found (RAG quality)
    if result.get("context_docs", 0) > 0:
        score += 1

    return min(score, 10)


def score_generation_quality(text: str, prompt_def: dict) -> Dict[str, Any]:
    """Score LLM generation output quality (0-100 scale)."""
    if not text or len(text.strip()) < 5:
        return {"score": 0, "pct": 0, "grade": "VERY POOR", "details": "empty output"}

    score = 0
    details = []
    text_stripped = text.strip()
    word_count = len(text_stripped.split())

    # Non-empty (15 pts)
    score += 10
    if len(text_stripped) > 20:
        score += 5

    # Length (15 pts)
    if word_count >= 30:
        score += 15
    elif word_count >= 10:
        score += 10
    elif word_count >= 5:
        score += 5

    # Keyword relevance (25 pts)
    keywords = prompt_def.get("keywords", [])
    if keywords:
        text_lower = text_stripped.lower()
        matched = [kw for kw in keywords if kw in text_lower]
        kw_score = round(len(matched) / len(keywords) * 25)
        score += kw_score
        if len(matched) < len(keywords) * 0.4:
            details.append(f"low keyword match ({len(matched)}/{len(keywords)})")

    # No error markers (10 pts)
    error_markers = ["error", "exception", "traceback", "failed"]
    if not any(m in text_stripped.lower() for m in error_markers):
        score += 10

    # Coherence (15 pts)
    sentences = [s.strip() for s in text_stripped.split(".") if len(s.strip()) > 3]
    if len(sentences) >= 3:
        score += 15
    elif len(sentences) >= 2:
        score += 10
    elif len(sentences) >= 1:
        score += 5

    # Not just repeating the prompt (10 pts)
    prompt_text = prompt_def.get("prompt", "")
    if text_stripped != prompt_text and not text_stripped.startswith(prompt_text):
        score += 10
    else:
        details.append("output is just the prompt repeated")

    pct = round(min(score, 100), 1)
    grade = (
        "EXCELLENT" if pct >= 85 else "GOOD" if pct >= 70 else
        "FAIR" if pct >= 50 else "POOR" if pct >= 30 else "VERY POOR"
    )
    return {
        "score": score, "pct": pct, "grade": grade, "word_count": word_count,
        "details": "; ".join(details) if details else "Good generation",
    }


# ============================================================================
# PYTEST TESTS - automated RAG conversation scoring
# ============================================================================

if HAS_PYTEST:

    @pytest.fixture(scope="module")
    def chat_client():
        client = VaLLMChatClient(BASE_URL)
        if not client.health_check():
            pytest.skip(f"VaLLM server not available at {BASE_URL}")
        return client

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    class TestRAGConversation_Kubernetes:
        """Multi-turn Kubernetes Q&A via RAG pipeline."""

        def test_what_is_k8s(self, chat_client):
            chat_client.reset()
            conv = CONVERSATIONS[0]
            turn = conv["turns"][0]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score(f"{conv['name']} - What is K8s", s, f"context_docs={result.get('context_docs', 0)}")
            print(f"  Response: {result.get('response', '')[:200]}...")

        def test_scale_deployment(self, chat_client):
            turn = CONVERSATIONS[0]["turns"][1]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("K8s - Scale deployment", s, turn["user"][:50])

        def test_deployment_vs_statefulset(self, chat_client):
            turn = CONVERSATIONS[0]["turns"][2]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("K8s - Deployment vs StatefulSet", s, turn["user"][:50])

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    class TestRAGConversation_Terraform:
        """Terraform IaC questions via RAG."""

        def test_terraform_init(self, chat_client):
            chat_client.reset()
            turn = CONVERSATIONS[1]["turns"][0]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Terraform - Init project", s, turn["user"][:50])

        def test_terraform_state(self, chat_client):
            turn = CONVERSATIONS[1]["turns"][1]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Terraform - State management", s, turn["user"][:50])

        def test_terraform_secrets(self, chat_client):
            turn = CONVERSATIONS[1]["turns"][2]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Terraform - Secrets", s, turn["user"][:50])

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    class TestRAGConversation_SRE:
        """SRE and reliability questions via RAG."""

        def test_oom_troubleshooting(self, chat_client):
            chat_client.reset()
            turn = CONVERSATIONS[2]["turns"][0]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("SRE - OOM troubleshooting", s, turn["user"][:50])

        def test_error_budgets(self, chat_client):
            turn = CONVERSATIONS[2]["turns"][1]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("SRE - Error budgets", s, turn["user"][:50])

        def test_circuit_breaker(self, chat_client):
            turn = CONVERSATIONS[2]["turns"][2]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("SRE - Circuit breaker", s, turn["user"][:50])

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    class TestRAGConversation_Billing:
        """Cloud billing questions via RAG."""

        def test_ec2_billing(self, chat_client):
            chat_client.reset()
            turn = CONVERSATIONS[3]["turns"][0]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Billing - EC2 pricing", s, turn["user"][:50])

        def test_savings_plans(self, chat_client):
            turn = CONVERSATIONS[3]["turns"][1]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Billing - Savings Plans", s, turn["user"][:50])

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    class TestSearchOnly:
        """Pure FAISS search without LLM generation."""

        def test_search_vpc(self, chat_client):
            chat_client.reset()
            turn = CONVERSATIONS[4]["turns"][0]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Search - VPC config", s, f"results={result.get('result_count', 0)}")

        def test_search_service_mesh(self, chat_client):
            turn = CONVERSATIONS[4]["turns"][1]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Search - Service mesh", s, f"results={result.get('result_count', 0)}")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    class TestDirectGeneration:
        """Direct /generate endpoint (no retrieval)."""

        def test_generate_docker(self, chat_client):
            chat_client.reset()
            turn = CONVERSATIONS[5]["turns"][0]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Generate - Docker basics", s, f"model_loaded={result.get('model_loaded')}")

        def test_generate_cicd(self, chat_client):
            turn = CONVERSATIONS[5]["turns"][1]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Generate - CI/CD", s, turn["user"][:50])

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    class TestRAGConversation_Networking:
        """Networking questions via RAG."""

        def test_sg_vs_nacl(self, chat_client):
            chat_client.reset()
            turn = CONVERSATIONS[6]["turns"][0]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Network - SG vs NACL", s, turn["user"][:50])

        def test_vpc_peering(self, chat_client):
            turn = CONVERSATIONS[6]["turns"][1]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Network - VPC peering", s, turn["user"][:50])

        def test_cdn(self, chat_client):
            turn = CONVERSATIONS[6]["turns"][2]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Network - CDN", s, turn["user"][:50])

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    class TestRAGConversation_CrossDomain:
        """Cross-domain queries via RAG."""

        def test_k8s_prometheus(self, chat_client):
            chat_client.reset()
            turn = CONVERSATIONS[7]["turns"][0]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Cross - K8s + Prometheus", s, f"context_docs={result.get('context_docs', 0)}")

        def test_golden_signals(self, chat_client):
            turn = CONVERSATIONS[7]["turns"][1]
            result = chat_client.chat(turn["user"], turn["mode"])
            s = score_turn(turn, result)
            _record_score("Cross - Golden signals", s, turn["user"][:50])


# ============================================================================
# MODE 1 - AUTOMATED RAG CONVERSATIONS (standalone runner)
# ============================================================================

def run_automated_conversations(base_url: str = BASE_URL):
    """Run all pre-defined RAG conversations and produce a scorecard."""
    print("\n" + "=" * 78)
    print("  VaLLM Interactive Chat Tests - Automated RAG Conversations")
    print("=" * 78)
    print(f"  Server:        {base_url}")
    print(f"  Conversations: {len(CONVERSATIONS)}")
    print(f"  Total turns:   {sum(len(c['turns']) for c in CONVERSATIONS)}")
    print(f"  Pipeline:      search (FAISS) -> augment -> generate (LLM)")
    print("=" * 78)

    client = VaLLMChatClient(base_url)

    print("\nChecking service health...", end=" ", flush=True)
    if not client.health_check():
        # Try common alternatives
        for alt in [base_url.replace("localhost", "127.0.0.1"),
                     base_url.rsplit(":", 1)[0] + ":8000",
                     "http://127.0.0.1:8000"]:
            alt_client = VaLLMChatClient(alt)
            if alt_client.health_check():
                client = alt_client
                print(f"FOUND at {alt}")
                break
        else:
            print("FAILED")
            print(f"\nService not available at {base_url}.")
            print("Options:")
            print("  1. Start the app:  python -m app.app")
            print("  2. Use offline:    python -m app.tests.tests_interactive_chat --offline")
            return
    else:
        print("OK")
    print()

    total_passed = 0
    total_turns = 0

    for conv in CONVERSATIONS:
        client.reset()
        print(f"\n{'=' * 78}")
        print(f"  CONVERSATION: {conv['name']}")
        print(f"  {conv['description']}")
        print(f"{'=' * 78}")

        for i, turn in enumerate(conv["turns"], 1):
            total_turns += 1
            mode = turn.get("mode", "rag")
            print(f"\n  [{i}/{len(conv['turns'])}] User: {turn['user']}")
            print(f"  Mode: {mode}")

            result = client.chat(turn["user"], mode)
            response = result.get("response", "")
            elapsed = result.get("elapsed_ms", 0)

            print(f"  Time: {elapsed:.0f}ms | Context docs: {result.get('context_docs', '-')}")

            if response and len(response.strip()) > 5:
                preview = response[:250].replace("\n", " ")
                print(f"  Assistant: {preview}...")
                total_passed += 1

                score = score_turn(turn, result)
                _record_score(f"{conv['name']} - Turn {i}", score, turn["user"][:50])
            else:
                print(f"  [WARN] Empty or short response")
                _record_score(f"{conv['name']} - Turn {i}", 3, "Empty response")

    print(f"\n{'=' * 78}")
    print(f"  RESULTS: {total_passed}/{total_turns} turns successful")
    print(f"  Pass Rate: {total_passed / total_turns * 100:.0f}%" if total_turns else "  No turns")
    print(f"{'=' * 78}")


# ============================================================================
# MODE 2 - LIVE INTERACTIVE CHAT (terminal REPL, needs server)
# ============================================================================

def interactive_chat(base_url: str = BASE_URL, initial_query: str = ""):
    """Live interactive chat with VaLLM via RAG pipeline."""
    print("\n" + "=" * 78)
    print("  VaLLM Interactive Chat (RAG Pipeline)")
    print("=" * 78)
    print("  Talk to VaLLM like ChatGPT.  Each query goes through:")
    print("    1. FAISS search -> retrieve relevant knowledge")
    print("    2. Build augmented prompt with context")
    print("    3. Generate answer with fine-tuned LLM")
    print()
    print("  Modes (prefix your message):")
    print("    /search    Search only (show relevant documents)")
    print("    /gen       Generate only (raw LLM, no retrieval)")
    print("    (default)  Full RAG (search + generate)")
    print()
    print("  Commands:")
    print("    /reset     Clear conversation history")
    print("    /history   Show conversation history")
    print("    /health    Check service health")
    print("    /stats     Show search/generation stats")
    print("    /quit      Exit")
    print("=" * 78)

    client = VaLLMChatClient(base_url)

    # Try the given URL first, then common alternatives
    print(f"\n  Connecting to {base_url}...", end=" ", flush=True)
    connected = False
    if client.health_check():
        connected = True
        print("OK")
    else:
        # Try common alternative URLs
        alt_urls = []
        if "localhost" in base_url:
            alt_urls.append(base_url.replace("localhost", "127.0.0.1"))
            alt_urls.append(base_url.replace("localhost", "host.docker.internal"))
        elif "127.0.0.1" in base_url:
            alt_urls.append(base_url.replace("127.0.0.1", "localhost"))
        # Try port 8000 (Docker default)
        alt_urls.append(base_url.rsplit(":", 1)[0] + ":8000")
        alt_urls.append("http://127.0.0.1:8000")

        for alt in alt_urls:
            alt_client = VaLLMChatClient(alt)
            if alt_client.health_check():
                client = alt_client
                connected = True
                print(f"FOUND at {alt}")
                break

    if not connected:
        print("FAILED")
        print(f"\n  Server not reachable at {base_url} (or common alternatives).")
        print("  Switching to OFFLINE mode (local FAISS search, no LLM generation).\n")
        offline_chat()
        return
    print(f"  Server: {client.base_url}")

    search_count = 0
    generate_count = 0

    if initial_query:
        print(f"\n  [{client.turn + 1}] You: {initial_query}")
        result = client.chat(initial_query, "rag")
        _display_chat_result(result)
        search_count += 1
        generate_count += 1

    while True:
        try:
            print()
            prompt = "  You > " if client.turn > 0 else "  Ask me anything > "
            user_input = input(prompt).strip()
            if not user_input:
                continue

            lower = user_input.lower()

            # Commands
            if lower in ("/quit", "/exit", "quit", "exit", "q"):
                print(f"\n  Session ended. {client.turn} messages exchanged. Goodbye!")
                break
            if lower == "/reset":
                client.reset()
                print("  Conversation history cleared.")
                continue
            if lower == "/history":
                if not client.history:
                    print("  No history yet.")
                else:
                    for entry in client.history:
                        role = entry["role"]
                        content = entry["content"][:150]
                        print(f"    {role}: {content}")
                continue
            if lower == "/health":
                status = "OK" if client.health_check() else "Unreachable"
                print(f"  Health: {status}")
                continue
            if lower == "/stats":
                print(f"  Turns: {client.turn} | Searches: {search_count} | Generations: {generate_count}")
                print(f"  History: {len(client.history)} entries | Session: {client.session_id[:8]}...")
                continue

            # Determine mode from prefix
            mode = "rag"
            if user_input.startswith("/search "):
                user_input = user_input[8:]
                mode = "search"
            elif user_input.startswith("/gen "):
                user_input = user_input[5:]
                mode = "generate"

            print(f"\n  [{client.turn + 1}] You: {user_input}")
            result = client.chat(user_input, mode)
            _display_chat_result(result)

            if mode in ("rag", "search"):
                search_count += 1
            if mode in ("rag", "generate"):
                generate_count += 1

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  Session ended. {client.turn} messages exchanged. Goodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}")


def _display_chat_result(result: dict):
    """Pretty-print a chat response."""
    mode = result.get("mode", "?")
    elapsed = result.get("elapsed_ms", 0)
    context = result.get("context_docs", 0)

    info_parts = [f"mode={mode}", f"{elapsed:.0f}ms"]
    if context:
        info_parts.append(f"context_docs={context}")
    if result.get("model_loaded") is not None:
        info_parts.append(f"model={'loaded' if result['model_loaded'] else 'not loaded'}")

    print(f"  [{' | '.join(info_parts)}]")

    response = result.get("response", "")
    if not response:
        print("  (no response)")
        return

    print(f"\n  Assistant:")
    for line in response[:1500].split("\n"):
        print(f"    {line}")
    if len(response) > 1500:
        print(f"    ... ({len(response)} chars total)")


# ============================================================================
# MODE 3 - OFFLINE INTERACTIVE CHAT (no server, uses embedding service)
# ============================================================================

def offline_chat():
    """
    Offline interactive chat using FAISS index directly.
    No server needed. Loads embedding model and searches local knowledge.
    """
    print("\n" + "=" * 78)
    print("  VaLLM Offline Chat (Local FAISS Search)")
    print("=" * 78)
    print("  Answers from indexed knowledge base (279K+ cloud/DevOps Q&A).")
    print("  No server or API keys needed.")
    print()
    print("  Commands:")
    print("    /quit      Exit")
    print("    /stats     Show index stats")
    print("=" * 78)

    import asyncio
    try:
        from services.ai.ml.embedding import embedding_service
    except ImportError:
        from app.services.ai.ml.embedding import embedding_service

    async def _init():
        print("\n  Loading embedding model and FAISS index...", end=" ", flush=True)
        start = time.perf_counter()
        ok = await embedding_service.initialize()
        elapsed = time.perf_counter() - start
        if not ok:
            print("FAILED")
            print("  Run precompute first: python -m app.services.ai.ml.precompute")
            return False
        stats = await embedding_service.get_faiss_stats()
        print(f"OK ({elapsed:.1f}s)")
        print(f"  Vectors: {stats.get('total_vectors', 0):,} | Dim: {stats.get('dimension', 0)}")
        return True

    async def _search(query: str, top_k: int = 5):
        return await embedding_service.search_faiss(query, top_k=top_k, use_reranker=True)

    async def _stats():
        return await embedding_service.get_faiss_stats()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if not loop.run_until_complete(_init()):
        return

    print("\n  Ready! Ask anything about cloud, DevOps, SRE, networking.\n")

    turn = 0
    while True:
        try:
            prompt = "  You > " if turn > 0 else "  Ask me anything > "
            user_input = input(prompt).strip()
            if not user_input:
                continue

            lower = user_input.lower()
            if lower in ("/quit", "/exit", "quit", "exit", "q"):
                print(f"\n  Session ended after {turn} queries. Goodbye!")
                break
            if lower == "/stats":
                stats = loop.run_until_complete(_stats())
                print(f"  Vectors: {stats.get('total_vectors', 0):,}")
                for t, c in stats.get('by_type', {}).items():
                    print(f"    {t}: {c:,}")
                continue

            turn += 1
            start = time.perf_counter()
            results = loop.run_until_complete(_search(user_input, top_k=5))
            elapsed_ms = (time.perf_counter() - start) * 1000

            print(f"\n  [{turn}] Results ({elapsed_ms:.0f}ms, {len(results)} found):")
            if results:
                for i, r in enumerate(results, 1):
                    score = r.get("rerank_score", r.get("score", 0))
                    doc = r.get("document", "")[:400]
                    meta = r.get("metadata", {})
                    source = meta.get("source", meta.get("type", ""))
                    print(f"\n  [{i}] Score: {score:.4f}" + (f" | Source: {source}" if source else ""))
                    for line in doc.split("\n")[:5]:
                        print(f"      {line.strip()}")
                    if len(doc) > 300:
                        print(f"      ...")
            else:
                print("  No results found.")
            print()

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  Session ended after {turn} queries. Goodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}")

    loop.close()


# ============================================================================
# MODE 4 - DIRECT LLM GENERATION TESTS (no server needed)
# ============================================================================

def run_generation_tests():
    """Test the fine-tuned LLM model directly (no server)."""
    print("\n" + "=" * 78)
    print("  VaLLM LLM Generation Tests (direct model, no server)")
    print("=" * 78)

    MODEL_DIR = _APP_DIR / "data" / "models" / "model"

    if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").exists():
        print(f"\n  Model not found at {MODEL_DIR}")
        print("  Train first: python -m app.services.ai.ml.train --num-train-epochs 1")
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("\n  Missing dependencies: pip install transformers torch")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Model dir: {MODEL_DIR}")
    print(f"  Device:    {device}")
    print(f"  Prompts:   {len(GENERATION_PROMPTS)}")

    print("\n  Loading model...", end=" ")
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR)).to(device)
    load_ms = (time.perf_counter() - start) * 1000
    print(f"OK ({load_ms:.0f}ms)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print("=" * 78)

    all_scores = []
    for prompt_def in GENERATION_PROMPTS:
        print(f"\n  [{prompt_def['id']}] {prompt_def['name']}")
        print(f"  Prompt: {prompt_def['prompt'][:80]}...")

        start = time.perf_counter()
        inputs = tokenizer(prompt_def["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=prompt_def["max_tokens"],
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_ms = (time.perf_counter() - start) * 1000

        if generated.startswith(prompt_def["prompt"]):
            generated = generated[len(prompt_def["prompt"]):].strip()

        tokens_gen = len(outputs[0]) - len(inputs["input_ids"][0])

        print(f"  Time: {gen_ms:.0f}ms | Tokens: {tokens_gen}")
        print(f"  Output:")
        for line in generated[:400].split("\n"):
            print(f"    {line}")

        score_info = score_generation_quality(generated, prompt_def)
        all_scores.append({"id": prompt_def["id"], "name": prompt_def["name"], **score_info})

        bar = "#" * (score_info["pct"] // 10) + "." * (10 - score_info["pct"] // 10)
        print(f"  Score: {score_info['pct']:.0f}% [{score_info['grade']}] [{bar}]")
        print(f"  {score_info['details']}")

    # Scorecard
    print(f"\n{'=' * 78}")
    print(f"  LLM GENERATION - SCORECARD")
    print(f"{'=' * 78}")
    print(f"  {'ID':<5} {'Test':<35} {'Score':>6}  {'Grade':<12}")
    print(f"  {'-' * 5} {'-' * 35} {'-' * 6}  {'-' * 12}")
    for s in all_scores:
        print(f"  {s['id']:<5} {s['name'][:34]:<35} {s['pct']:>5.0f}%  {s['grade']:<12}")
    avg = sum(s["pct"] for s in all_scores) / len(all_scores) if all_scores else 0
    grade = "EXCELLENT" if avg >= 85 else "GOOD" if avg >= 70 else "FAIR" if avg >= 50 else "POOR"
    print(f"  {'-' * 5} {'-' * 35} {'-' * 6}  {'-' * 12}")
    print(f"  {'AVG':<5} {'OVERALL':<35} {avg:>5.0f}%  {grade:<12}")
    print(f"{'=' * 78}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VaLLM Interactive Chat & RAG Tests",
        epilog=(
            "Examples:\n"
            "  python -m app.tests.tests_interactive_chat                              # automated RAG tests\n"
            "  python -m app.tests.tests_interactive_chat --interactive                 # live chat (needs server)\n"
            '  python -m app.tests.tests_interactive_chat --interactive "deploy EC2"    # chat with initial query\n'
            "  python -m app.tests.tests_interactive_chat --offline                     # offline chat (no server)\n"
            "  python -m app.tests.tests_interactive_chat --generate                    # direct LLM tests\n"
            "  python -m app.tests.tests_interactive_chat --url http://localhost:8747   # custom URL\n"
            "\n"
            "  pytest app/tests/tests_interactive_chat.py -v -s                        # pytest scored tests\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--interactive", nargs="?", const="", default=None,
                        metavar="QUERY",
                        help="Start live interactive chat via server (optionally with initial query)")
    parser.add_argument("--offline", action="store_true",
                        help="Offline interactive chat using local FAISS (no server needed)")
    parser.add_argument("--generate", action="store_true",
                        help="Run direct LLM generation tests (no server needed)")
    parser.add_argument("--url", default=BASE_URL,
                        help=f"Base URL (default: {BASE_URL})")
    args = parser.parse_args()

    if args.offline:
        offline_chat()
    elif args.interactive is not None:
        interactive_chat(args.url, args.interactive)
    elif args.generate:
        run_generation_tests()
    else:
        run_automated_conversations(args.url)
