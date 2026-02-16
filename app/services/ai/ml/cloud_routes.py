"""
Cloud provisioning intent API for VaLLM.

Returns structured intent + Golang-ready payload for provisioning requests only.
Non-provisioning queries return query_type "other". Used by InfinityAI cloud agent.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .entity_extraction import extract_entities_from_query

router = APIRouter(prefix="/api/cloud", tags=["cloud"])
logger = logging.getLogger("vallm.cloud")


class ProvisionIntentRequest(BaseModel):
    query: str


def _is_deployment_result(meta: Dict[str, Any]) -> bool:
    """True if this search result is from cloud_deployments (has intent in raw)."""
    raw = meta.get("raw") or {}
    return isinstance(raw, dict) and "intent" in raw and raw.get("intent")


def _raw_to_golang_payload(raw: Dict[str, Any], intent: str) -> Dict[str, Any]:
    """
    Map cloud_deployments CSV row (metadata.raw) to payload shape expected by
    Golang/InfinityAI provisionner_services (username, workspace, instance_type, etc.).
    """
    def v(key: str, default: str = "") -> str:
        val = raw.get(key)
        if val is None or (isinstance(val, str) and val.strip().lower() in ("", "nan")):
            return default
        return str(val).strip()

    def vi(key: str, default: int = 0) -> int:
        val = raw.get(key)
        if val is None or val == "":
            return default
        try:
            return int(float(str(val)))
        except (ValueError, TypeError):
            return default

    payload: Dict[str, Any] = {
        "username": v("username"),
        "workspace": v("workspace"),
        "workspace_id": v("workspace"),
        "user_id": v("username"),
    }

    if intent == "provision_vm":
        payload["instance_name"] = v("instance_name") or v("hostname")
        payload["resource_name"] = payload["instance_name"]
        payload["instance_type"] = v("instance_type", "t2.micro")
        payload["region"] = v("region", "us-east-1")
        payload["cloud_provider"] = v("cloud_provider", "aws")
        payload["os"] = v("os", "ubuntu")
        payload["volume_size"] = vi("volume_size_gb", 30)
        payload["volume_type"] = v("volume_type", "gp3")
        payload["environment"] = v("environment", "dev")
        payload["hostname"] = v("hostname")
        payload["ssh_username"] = v("ssh_username", "ubuntu")
        payload["key_pair_name"] = v("key_pair_name")

    elif intent == "provision_kubernetes":
        payload["cluster_name"] = v("cluster_name")
        payload["node_count"] = vi("node_count", 2)
        payload["node_type"] = v("node_type", "t3.medium")
        payload["kubernetes_version"] = v("kubernetes_version")
        payload["region"] = v("region", "us-east-1")
        payload["cloud_provider"] = v("cloud_provider", "aws")

    elif intent == "provision_docker":
        payload["docker_image"] = v("docker_image")
        payload["image"] = payload["docker_image"]
        payload["container_name"] = v("container_name") or v("docker_service")
        payload["docker_service"] = v("docker_service")
        payload["ports"] = v("ports")

    elif intent == "provision_fastapi":
        payload["hostname"] = v("hostname")
        payload["app_name"] = v("app_name")
        payload["app_port"] = vi("app_port", 8000)
        payload["http_port"] = vi("http_port", 8080)
        payload["ssh_username"] = v("ssh_username", "ubuntu")
        payload["key_pair_name"] = v("key_pair_name")

    elif intent == "provision_static_website":
        payload["hostname"] = v("hostname")
        payload["server_name"] = v("server_name")
        payload["http_port"] = vi("http_port", 80)
        payload["ssh_username"] = v("ssh_username", "ubuntu")
        payload["key_pair_name"] = v("key_pair_name")

    elif intent == "provision_database":
        payload["hostname"] = v("hostname")
        payload["database_engine"] = v("database_engine", "postgres")
        payload["database_name"] = v("database_name")
        payload["database_user"] = v("database_user")
        payload["postgres_version"] = v("postgres_version")
        payload["port"] = vi("port", 5432)
        payload["ssh_username"] = v("ssh_username", "ubuntu")
        payload["key_pair_name"] = v("key_pair_name")

    # Common optional fields
    for key in ("region", "cloud_provider", "hostname", "ssh_username", "key_pair_name"):
        if key not in payload and v(key):
            payload[key] = v(key)

    return payload


def _classify_non_provisioning(query: str) -> str:
    """When no provisioning match, return 'other' only (provisioning-focused API)."""
    return "other"


@router.post("/provision-intent")
async def provision_intent(
    request: ProvisionIntentRequest,
    req: Request,
):
    """
    Resolve user message to provisioning intent + Golang-ready payload.
    When no deployment match: returns query_type "other". Provisioning-only.
    """
    vector_store = getattr(req.app.state, "vector_store", None)
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    query = (request.query or "").strip()
    if not query:
        return {
            "query_type": "other",
            "intent": None,
            "payload": None,
            "confidence": 0.0,
            "match_prompt": None,
        }

    # Search; prefer deployment-type results if precompute tags them (filter_type="deployment")
    try:
        search_results = await vector_store.search(
            query=query,
            top_k=20,
            filter_type=None,
        )
    except Exception as e:
        logger.exception("provision-intent search failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Keep only rows from cloud_deployments (raw has "intent")
    deployment_results: List[Dict[str, Any]] = [
        r for r in search_results
        if _is_deployment_result(r.get("metadata") or {})
    ]

    if not deployment_results:
        query_type = _classify_non_provisioning(query)
        return {
            "query_type": query_type,
            "intent": None,
            "payload": None,
            "confidence": 0.0,
            "match_prompt": None,
        }

    best = deployment_results[0]
    meta = best.get("metadata") or {}
    raw = meta.get("raw") or {}
    intent = raw.get("intent", "")
    score = best.get("score", 0.0)
    match_prompt = raw.get("prompt", "")

    # Require minimum confidence (e.g. score > 0.3 for cosine-like scores; adjust if your metric differs)
    confidence = float(score)
    if confidence < 0.2:
        query_type = _classify_non_provisioning(query)
        return {
            "query_type": query_type,
            "intent": None,
            "payload": None,
            "confidence": confidence,
            "match_prompt": match_prompt,
        }

    payload = _raw_to_golang_payload(raw, intent)
    
    # Extract entities from user query and override matched values
    extracted_entities = extract_entities_from_query(query)
    if extracted_entities:
        logger.info(f"Overriding matched payload with extracted entities: {extracted_entities}")
        payload.update(extracted_entities)
    
    return {
        "query_type": "provisioning",
        "intent": intent,
        "payload": payload,
        "confidence": confidence,
        "match_prompt": match_prompt,
    }
