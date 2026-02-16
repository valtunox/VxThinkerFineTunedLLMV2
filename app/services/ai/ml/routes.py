"""
API Routes for VaLLM
All API versions (v1, v2, v3) consolidated in a single module.

V1 endpoints: /query, /developer, /terminal
V2 endpoints: /query, /extract, /upload, /status
V3 endpoints: /query
"""

import io
import re
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import Counter
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, Body, File, UploadFile, Form
from pydantic import BaseModel
import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import IsolationForest
import xgboost as xgb

# Try to import spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Try to import PDF extraction
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Try to import image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try to import OCR for text extraction from images
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

router = APIRouter()
router_v2 = APIRouter()
router_v3 = APIRouter()
logger = logging.getLogger("vallm")


def _summarize_context(results: list, max_items: int = 3, preview_chars: int = 180) -> str:
    if not results:
        return "no_results"
    parts = []
    for r in results[:max_items]:
        doc = (r.get("document") or "").replace("\n", " ").strip()
        if len(doc) > preview_chars:
            doc = f"{doc[:preview_chars]}..."
        meta = r.get("metadata") or {}
        doc_type = meta.get("type", "unknown") if isinstance(meta, dict) else "unknown"
        score = r.get("score", 0.0)
        parts.append(f"[{doc_type}] score={score:.4f} doc='{doc}'")
    return " | ".join(parts)


def _build_terraform_example(query: str) -> str:
    """Return a full Terraform example based on the query intent."""
    q = (query or "").lower()
    wants_vpc = "vpc" in q or "subnet" in q or "flow log" in q
    wants_private_dns = "private dns" in q or "private dns namespace" in q
    wants_kms = "kms" in q or "encryption" in q

    if wants_vpc:
        dns_namespace_block = ""
        if wants_private_dns:
            dns_namespace_block = """
resource "aws_service_discovery_private_dns_namespace" "main" {
  name = "svc.prod.internal"
  vpc  = aws_vpc.main.id
  description = "Private DNS namespace for internal services"
}
""".strip()

        kms_blocks = ""
        log_kms_id = ""
        if wants_kms:
            kms_blocks = """
resource "aws_kms_key" "logs" {
  description             = "KMS key for VPC flow logs"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}
""".strip()
            log_kms_id = "  kms_key_id = aws_kms_key.logs.arn\n"

        return f"""
provider "aws" {{
  region = "us-east-1"
}}

{kms_blocks}

resource "aws_vpc" "main" {{
  cidr_block           = "10.50.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {{
    Name        = "prod-vpc"
    Environment = "Production"
    Compliance  = "SOC2"
  }}
}}

resource "aws_subnet" "private_a" {{
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.50.1.0/24"
  availability_zone = "us-east-1a"
  map_public_ip_on_launch = false
}}

resource "aws_cloudwatch_log_group" "vpc_flow" {{
  name = "/vpc/flow-logs"
{log_kms_id}  retention_in_days = 30
}}

resource "aws_iam_role" "vpc_flow" {{
  name = "vpc-flow-logs-role"
  assume_role_policy = jsonencode({{
    Version = "2012-10-17"
    Statement = [{{
      Effect = "Allow"
      Principal = {{ Service = "vpc-flow-logs.amazonaws.com" }}
      Action = "sts:AssumeRole"
    }}]
  }})
}}

resource "aws_iam_role_policy" "vpc_flow" {{
  name = "vpc-flow-logs-policy"
  role = aws_iam_role.vpc_flow.id
  policy = jsonencode({{
    Version = "2012-10-17"
    Statement = [{{
      Effect = "Allow"
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams"
      ]
      Resource = "*"
    }}]
  }})
}}

resource "aws_flow_log" "vpc" {{
  vpc_id               = aws_vpc.main.id
  log_destination_type = "cloud-watch-logs"
  log_destination      = aws_cloudwatch_log_group.vpc_flow.arn
  iam_role_arn         = aws_iam_role.vpc_flow.arn
  traffic_type         = "ALL"
}}

{dns_namespace_block}
""".strip()

    return """
provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "example" {
  bucket = "example-app-bucket"

  tags = {
    Environment = "Production"
    Owner       = "platform-team"
  }
}
""".strip()


class QueryRequest(BaseModel):
    """Query endpoint request model"""
    query: str
    top_k: Optional[int] = 5
    filter_type: Optional[str] = None
    include_reasoning: Optional[bool] = True


class DeveloperRequest(BaseModel):
    """Developer endpoint request model"""
    query: str
    context: Optional[Dict[str, Any]] = None
    include_code: Optional[bool] = True
    include_reasoning: Optional[bool] = True


class TerminalRequest(BaseModel):
    """Terminal endpoint request model"""
    command: str
    context: Optional[Dict[str, Any]] = None
    include_explanation: Optional[bool] = True


class V3QueryRequest(BaseModel):
    """V3 query endpoint request model with cloud/devops incident patterns"""
    query: str
    top_k: Optional[int] = 10
    include_reasoning: Optional[bool] = True
    focus: Optional[str] = "cloud_devops"


CLOUD_DEVOPS_KEYWORDS = {
    "aws", "azure", "gcp", "kubernetes", "eks", "aks", "gke",
    "vpc", "iam", "ec2", "s3", "rds", "cloud", "container",
    "docker", "k8s", "cicd", "ci", "cd", "terraform", "ansible",
    "devops", "network", "security", "vpn", "dns", "loadbalancer",
}


def _normalize_incident_meta(meta: Dict[str, Any]) -> Dict[str, str]:
    raw = meta.get("raw") or meta.get("raw_data") or {}
    return {
        "incident_id": str(raw.get("incident_id", "")),
        "severity": str(raw.get("severity", "")).lower(),
        "category": str(raw.get("category", "")).lower(),
        "service": str(raw.get("service", "")).lower(),
        "error_code": str(raw.get("error_code", "")).lower(),
        "tags": str(raw.get("tags", "")).lower(),
        "timestamp": str(raw.get("timestamp", "")),
    }


def _is_cloud_devops_incident(incident: Dict[str, str]) -> bool:
    haystack = " ".join(
        [
            incident.get("category", ""),
            incident.get("service", ""),
            incident.get("tags", ""),
            incident.get("error_code", ""),
        ]
    )
    return any(keyword in haystack for keyword in CLOUD_DEVOPS_KEYWORDS)


def _severity_to_score(severity: str) -> float:
    mapping = {
        "critical": 1.0,
        "high": 0.75,
        "medium": 0.5,
        "low": 0.25,
    }
    return mapping.get(severity, 0.0)


def _compute_query_similarity(
    vector_store,
    query: str,
    incidents: List[Dict[str, Any]]
) -> List[float]:
    if not incidents:
        return []
    if not getattr(vector_store, "model", None):
        return [0.0 for _ in incidents]

    documents = [item.get("document", "") for item in incidents]
    embeddings = vector_store.model.encode(documents, convert_to_numpy=True)
    query_embedding = vector_store.model.encode([query], convert_to_numpy=True)[0]

    doc_t = torch.tensor(embeddings, dtype=torch.float32)
    query_t = torch.tensor(query_embedding, dtype=torch.float32)
    doc_norm = torch.nn.functional.normalize(doc_t, dim=1)
    query_norm = torch.nn.functional.normalize(query_t, dim=0)
    similarities = torch.matmul(doc_norm, query_norm).cpu().numpy().tolist()
    return [float(s) for s in similarities]


def analyze_unusual_incident_patterns(incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = []
    for item in incidents:
        meta = _normalize_incident_meta(item.get("metadata", {}))
        normalized.append(meta)

    if not normalized:
        return {
            "unusual_incidents": [],
            "metrics": {
                "total_incidents": 0,
                "total_candidates": 0,
                "unusual_count": 0,
                "unusual_rate": 0.0,
                "unique_categories": 0,
                "unique_services": 0,
                "unique_error_codes": 0,
            },
            "top_patterns": [],
        }

    category_counts = Counter(i["category"] for i in normalized if i["category"])
    service_counts = Counter(i["service"] for i in normalized if i["service"])
    error_counts = Counter(i["error_code"] for i in normalized if i["error_code"])

    unusual = []
    for inc in normalized:
        signals = []
        if inc["severity"] in {"critical", "high"}:
            signals.append("high_severity")
        if inc["category"] and category_counts[inc["category"]] == 1:
            signals.append("rare_category")
        if inc["service"] and service_counts[inc["service"]] == 1:
            signals.append("rare_service")
        if inc["error_code"] and error_counts[inc["error_code"]] == 1:
            signals.append("rare_error_code")
        if _is_cloud_devops_incident(inc):
            signals.append("cloud_devops_signal")

        if signals:
            unusual.append(
                {
                    "incident_id": inc.get("incident_id"),
                    "severity": inc.get("severity"),
                    "category": inc.get("category"),
                    "service": inc.get("service"),
                    "error_code": inc.get("error_code"),
                    "timestamp": inc.get("timestamp"),
                    "signals": signals,
                }
            )

    metrics = {
        "total_incidents": len(normalized),
        "total_candidates": len(normalized),
        "unusual_count": len(unusual),
        "unusual_rate": round(len(unusual) / max(len(normalized), 1), 3),
        "unique_categories": len(category_counts),
        "unique_services": len(service_counts),
        "unique_error_codes": len(error_counts),
    }

    top_patterns = [
        {
            "pattern": "top_categories",
            "values": category_counts.most_common(3),
        },
        {
            "pattern": "top_services",
            "values": service_counts.most_common(3),
        },
        {
            "pattern": "top_error_codes",
            "values": error_counts.most_common(3),
        },
    ]

    return {
        "unusual_incidents": unusual,
        "metrics": metrics,
        "top_patterns": top_patterns,
    }


def analyze_unusual_incident_patterns_v3(
    vector_store,
    query: str,
    incidents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    normalized = []
    for item in incidents:
        meta = _normalize_incident_meta(item.get("metadata", {}))
        normalized.append(meta)

    if not normalized:
        return {
            "unusual_incidents": [],
            "metrics": {
                "total_incidents": 0,
                "total_candidates": 0,
                "unusual_count": 0,
                "unusual_rate": 0.0,
                "unique_categories": 0,
                "unique_services": 0,
                "unique_error_codes": 0,
            },
            "top_patterns": [],
            "success_metrics": {
                "avg_anomaly_score": 0.0,
                "avg_xgb_residual": 0.0,
                "avg_query_similarity": 0.0,
            },
        }

    category_counts = Counter(i["category"] for i in normalized if i["category"])
    service_counts = Counter(i["service"] for i in normalized if i["service"])
    error_counts = Counter(i["error_code"] for i in normalized if i["error_code"])

    query_similarities = _compute_query_similarity(vector_store, query, incidents)

    categorical_dicts = []
    numeric_features = []
    severity_scores = []
    for idx, inc in enumerate(normalized):
        categorical_dicts.append(
            {
                "category": inc.get("category", ""),
                "service": inc.get("service", ""),
                "error_code": inc.get("error_code", ""),
            }
        )
        severity_score = _severity_to_score(inc.get("severity", ""))
        severity_scores.append(severity_score)
        numeric_features.append(
            [
                severity_score,
                float(category_counts.get(inc.get("category", ""), 0)),
                float(service_counts.get(inc.get("service", ""), 0)),
                float(error_counts.get(inc.get("error_code", ""), 0)),
                float(query_similarities[idx]) if idx < len(query_similarities) else 0.0,
            ]
        )

    vectorizer = DictVectorizer(sparse=False)
    cat_matrix = vectorizer.fit_transform(categorical_dicts)
    feature_matrix = np.hstack([np.array(numeric_features, dtype=np.float32), cat_matrix])

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
    )
    iso_scores = -iso_forest.fit_predict(feature_matrix)
    anomaly_scores = iso_forest.decision_function(feature_matrix)

    xgb_model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    xgb_model.fit(feature_matrix, np.array(severity_scores, dtype=np.float32))
    xgb_pred = xgb_model.predict(feature_matrix)
    xgb_residuals = np.abs(xgb_pred - np.array(severity_scores, dtype=np.float32))

    unusual = []
    for idx, inc in enumerate(normalized):
        signals = []
        if inc["severity"] in {"critical", "high"}:
            signals.append("high_severity")
        if inc["category"] and category_counts[inc["category"]] == 1:
            signals.append("rare_category")
        if inc["service"] and service_counts[inc["service"]] == 1:
            signals.append("rare_service")
        if inc["error_code"] and error_counts[inc["error_code"]] == 1:
            signals.append("rare_error_code")
        if _is_cloud_devops_incident(inc):
            signals.append("cloud_devops_signal")
        if iso_scores[idx] == 1:
            signals.append("sklearn_isolation_forest")
        if xgb_residuals[idx] > np.percentile(xgb_residuals, 75):
            signals.append("xgboost_residual")
        if query_similarities[idx] > 0.35:
            signals.append("pytorch_query_similarity")

        if signals:
            unusual.append(
                {
                    "incident_id": inc.get("incident_id"),
                    "severity": inc.get("severity"),
                    "category": inc.get("category"),
                    "service": inc.get("service"),
                    "error_code": inc.get("error_code"),
                    "timestamp": inc.get("timestamp"),
                    "signals": signals,
                    "anomaly_score": float(anomaly_scores[idx]),
                    "xgb_residual": float(xgb_residuals[idx]),
                    "query_similarity": float(query_similarities[idx]) if idx < len(query_similarities) else 0.0,
                }
            )

    metrics = {
        "total_incidents": len(normalized),
        "total_candidates": len(normalized),
        "unusual_count": len(unusual),
        "unusual_rate": round(len(unusual) / max(len(normalized), 1), 3),
        "unique_categories": len(category_counts),
        "unique_services": len(service_counts),
        "unique_error_codes": len(error_counts),
    }

    top_patterns = [
        {
            "pattern": "top_categories",
            "values": category_counts.most_common(3),
        },
        {
            "pattern": "top_services",
            "values": service_counts.most_common(3),
        },
        {
            "pattern": "top_error_codes",
            "values": error_counts.most_common(3),
        },
    ]

    success_metrics = {
        "avg_anomaly_score": float(np.mean(anomaly_scores)) if len(anomaly_scores) else 0.0,
        "avg_xgb_residual": float(np.mean(xgb_residuals)) if len(xgb_residuals) else 0.0,
        "avg_query_similarity": float(np.mean(query_similarities)) if len(query_similarities) else 0.0,
    }

    return {
        "unusual_incidents": unusual,
        "metrics": metrics,
        "top_patterns": top_patterns,
        "success_metrics": success_metrics,
    }


@router.post("/query")
async def query_endpoint(
    request: QueryRequest,
    req: Request
):
    """
    Query endpoint - General purpose query interface
    Similar to OpenAI/Anthropic chat completion
    """
    try:
        request_id = getattr(req.state, "request_id", "unknown")
        logger.info(
            "V1 query start | query='%s' top_k=%s filter_type=%s include_reasoning=%s",
            request.query,
            request.top_k,
            request.filter_type,
            request.include_reasoning,
            extra={"request_id": request_id, "use_color": True},
        )
        vector_store = req.app.state.vector_store
        reasoning_engine = req.app.state.reasoning_engine
        
        if not vector_store or not reasoning_engine:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Perform reasoning if requested
        if request.include_reasoning:
            reason_start = time.perf_counter()
            reasoning_result = await reasoning_engine.reason(
                query=request.query,
                max_steps=5
            )
            reason_ms = (time.perf_counter() - reason_start) * 1000
            logger.info(
                "V1 reasoning complete | intent=%s confidence=%.2f steps=%s duration_ms=%.2f",
                reasoning_result.get("intent"),
                reasoning_result.get("confidence", 0.0),
                len(reasoning_result.get("steps") or []),
                reason_ms,
                extra={"request_id": request_id, "use_color": True},
            )
            
            # Get additional context from vector store
            search_start = time.perf_counter()
            search_results = await vector_store.search(
                query=request.query,
                top_k=request.top_k,
                filter_type=request.filter_type
            )
            search_ms = (time.perf_counter() - search_start) * 1000
            logger.info(
                "V1 search complete | results=%s duration_ms=%.2f",
                len(search_results),
                search_ms,
                extra={"request_id": request_id, "use_color": True},
            )
            logger.info(
                "V1 search preview | %s",
                _summarize_context(search_results, max_items=2, preview_chars=120),
                extra={"request_id": request_id, "use_color": True},
            )
            logger.debug(
                "V1 search preview (full) | %s",
                _summarize_context(search_results),
                extra={"request_id": request_id},
            )
            
            return {
                "response": reasoning_result['final_answer'],
                "reasoning": {
                    "intent": reasoning_result['intent'],
                    "steps": reasoning_result['steps'],
                    "confidence": reasoning_result['confidence']
                },
                "context": [
                    {
                        "document": r.get('document', ''),
                        "type": r.get('metadata', {}).get('type', 'unknown') if isinstance(r.get('metadata'), dict) else 'unknown',
                        "score": r.get('score', 0.0)
                    }
                    for r in search_results
                ],
                "model": "vallm-v1",
                "usage": {
                    "tokens": len(request.query.split()),
                    "reasoning_steps": len(reasoning_result['steps'])
                }
            }
        else:
            # Simple search without reasoning
            search_start = time.perf_counter()
            search_results = await vector_store.search(
                query=request.query,
                top_k=request.top_k,
                filter_type=request.filter_type
            )
            search_ms = (time.perf_counter() - search_start) * 1000
            logger.info(
                "V1 search complete (no reasoning) | results=%s duration_ms=%.2f",
                len(search_results),
                search_ms,
                extra={"request_id": request_id, "use_color": True},
            )
            logger.info(
                "V1 search preview | %s",
                _summarize_context(search_results, max_items=2, preview_chars=120),
                extra={"request_id": request_id, "use_color": True},
            )
            logger.debug(
                "V1 search preview (full) | %s",
                _summarize_context(search_results),
                extra={"request_id": request_id},
            )

            # Simple response from top result
            if search_results:
                top_result = search_results[0]
                return {
                    "response": top_result.get('document', 'No content available'),
                    "context": [
                        {
                            "document": r.get('document', ''),
                            "type": r.get('metadata', {}).get('type', 'unknown') if isinstance(r.get('metadata'), dict) else 'unknown',
                            "score": r.get('score', 0.0)
                        }
                        for r in search_results
                    ],
                    "model": "vallm-v1",
                    "usage": {
                        "tokens": len(request.query.split())
                    }
                }
            else:
                return {
                    "response": "No relevant information found in the knowledge base.",
                    "context": [],
                    "model": "vallm-v1",
                    "usage": {
                        "tokens": len(request.query.split())
                    }
                }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        request_id = getattr(req.state, "request_id", "unknown")
        tb_str = traceback.format_exc()
        logger.error(f"V1 query failed [request_id={request_id}]\n{tb_str}")
        detail = f"Error processing query: {type(e).__name__}: {e or repr(e)}"
        raise HTTPException(status_code=500, detail=detail)


@router.post("/developer")
async def developer_endpoint(
    request: DeveloperRequest,
    req: Request
):
    """
    Developer endpoint - For code generation and developer assistance
    Focuses on infrastructure as code, configurations, and automation
    """
    try:
        vector_store = req.app.state.vector_store
        reasoning_engine = req.app.state.reasoning_engine
        
        if not vector_store or not reasoning_engine:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Perform reasoning with developer context
        reasoning_result = await reasoning_engine.reason(
            query=request.query,
            context=json.dumps(request.context) if request.context else None,
            max_steps=6
        )
        
        # Search for relevant configurations and code examples
        config_results = await vector_store.search(
            query=request.query,
            top_k=10,
            filter_type='configuration'
        )
        
        resource_results = await vector_store.search(
            query=request.query,
            top_k=5,
            filter_type='resource'
        )
        
        # Build developer-focused response
        response_parts = [reasoning_result['final_answer']]
        terraform_example = _build_terraform_example(request.query) if request.include_code else ""
        
        if request.include_code:
            response_parts.append("\n\n**Terraform (full example):**")
            response_parts.append(f"\n```terraform\n{terraform_example}\n```")

        if request.include_code and config_results:
            response_parts.append("\n\n**Relevant Configurations:**")
            for i, result in enumerate(config_results[:3], 1):
                response_parts.append(f"\n{i}. {result['document']}")
        
        if resource_results:
            response_parts.append("\n\n**Related Resources:**")
            for i, result in enumerate(resource_results[:3], 1):
                response_parts.append(f"\n{i}. {result['document']}")
        
        return {
            "response": "\n".join(response_parts),
            "reasoning": {
                "intent": reasoning_result['intent'],
                "steps": reasoning_result['steps'] if request.include_reasoning else None,
                "confidence": reasoning_result['confidence']
            },
            "code_examples": (
                [{
                    "type": "terraform",
                    "config": terraform_example,
                    "score": 1.0,
                }]
                + [
                    {
                        "type": r['metadata'].get('type'),
                        "config": r['document'],
                        "score": r['score']
                    }
                    for r in config_results[:5]
                ]
            ) if request.include_code else None,
            "model": "vallm-developer-v1",
            "usage": {
                "tokens": len(request.query.split()),
                "reasoning_steps": len(reasoning_result['steps'])
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing developer query: {str(e)}")


@router.post("/terminal")
async def terminal_endpoint(
    request: TerminalRequest,
    req: Request
):
    """
    Terminal endpoint - For command-line operations and terminal assistance
    Focuses on operational commands, troubleshooting, and system management
    """
    try:
        vector_store = req.app.state.vector_store
        reasoning_engine = req.app.state.reasoning_engine
        
        if not vector_store or not reasoning_engine:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Search for relevant incidents and troubleshooting info
        incident_results = await vector_store.search(
            query=request.command,
            top_k=10,
            filter_type='incident'
        )
        
        recommendation_results = await vector_store.search(
            query=request.command,
            top_k=5,
            filter_type='recommendation'
        )
        
        # Perform reasoning
        reasoning_result = await reasoning_engine.reason(
            query=request.command,
            context=json.dumps(request.context) if request.context else None,
            max_steps=5
        )
        
        # Build terminal-focused response
        response_parts = []
        
        if request.include_explanation:
            response_parts.append(f"**Command Analysis:**\n{reasoning_result['final_answer']}")
        
        if incident_results:
            response_parts.append("\n\n**Similar Incidents:**")
            for i, result in enumerate(incident_results[:3], 1):
                incident_data = result['metadata'].get('raw_data', {})
                title = incident_data.get('title', 'Unknown')
                resolution = incident_data.get('resolution_time_minutes', 'N/A')
                response_parts.append(f"\n{i}. {title} (Resolution: {resolution} min)")
        
        if recommendation_results:
            response_parts.append("\n\n**Recommendations:**")
            for i, result in enumerate(recommendation_results[:3], 1):
                rec_data = result['metadata'].get('raw_data', {})
                rec_type = rec_data.get('recommendation_type', 'general')
                priority = rec_data.get('priority', 'medium')
                response_parts.append(f"\n{i}. [{priority.upper()}] {rec_type}: {result['document'][:200]}...")
        
        return {
            "response": "\n".join(response_parts),
            "command": request.command,
            "reasoning": {
                "intent": reasoning_result['intent'],
                "steps": reasoning_result['steps'] if request.include_explanation else None,
                "confidence": reasoning_result['confidence']
            },
            "incidents": [
                {
                    "title": r['metadata'].get('raw_data', {}).get('title', 'Unknown'),
                    "severity": r['metadata'].get('raw_data', {}).get('severity', 'unknown'),
                    "score": r['score']
                }
                for r in incident_results[:5]
            ],
            "recommendations": [
                {
                    "type": r['metadata'].get('raw_data', {}).get('recommendation_type', 'general'),
                    "priority": r['metadata'].get('raw_data', {}).get('priority', 'medium'),
                    "score": r['score']
                }
                for r in recommendation_results[:5]
            ],
            "model": "vallm-terminal-v1",
            "usage": {
                "tokens": len(request.command.split()),
                "reasoning_steps": len(reasoning_result['steps'])
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing terminal command: {str(e)}")


@router_v3.post("/query")
async def v3_query_endpoint(
    request: V3QueryRequest,
    req: Request
):
    """
    V3 Query endpoint - Cloud/DevOps incident pattern analysis with success metrics.

    Analyzes incident patterns using local data with embeddings, sklearn IsolationForest,
    XGBoost residuals, and PyTorch query similarity.
    """
    try:
        request_id = getattr(req.state, "request_id", "unknown")
        logger.info(
            "V3 incident analysis start | query='%s' focus='%s'",
            request.query,
            request.focus,
            extra={"request_id": request_id, "use_color": True},
        )

        vector_store = req.app.state.vector_store
        reasoning_engine = req.app.state.reasoning_engine

        if not vector_store or not reasoning_engine:
            raise HTTPException(status_code=503, detail="Service not initialized")

        # Step 1: Local incident search
        local_start = time.perf_counter()
        search_results = await vector_store.search(
            query=request.query,
            top_k=request.top_k,
            filter_type="incident"
        )

        if request.focus == "cloud_devops":
            filtered = [
                r for r in search_results
                if _is_cloud_devops_incident(_normalize_incident_meta(r.get("metadata", {})))
            ]
        else:
            filtered = search_results

        local_ms = (time.perf_counter() - local_start) * 1000
        logger.info(
            "V3 local search complete | incidents=%s duration_ms=%.2f",
            len(filtered),
            local_ms,
            extra={"request_id": request_id, "use_color": True},
        )

        # Step 2: Analyze incident patterns
        analysis_start = time.perf_counter()
        analysis = analyze_unusual_incident_patterns_v3(vector_store, request.query, filtered)
        analysis_ms = (time.perf_counter() - analysis_start) * 1000

        # Step 3: Reasoning
        if request.include_reasoning:
            reasoning_start = time.perf_counter()

            context_summary = {
                "unusual_count": analysis["metrics"]["unusual_count"],
                "top_patterns": analysis["top_patterns"],
            }

            reasoning_result = await reasoning_engine.reason(
                query=request.query,
                context=json.dumps(context_summary),
                max_steps=5
            )
            reasoning_ms = (time.perf_counter() - reasoning_start) * 1000

            response_text = reasoning_result["final_answer"]

            reasoning_payload = {
                "intent": reasoning_result["intent"],
                "steps": reasoning_result["steps"],
                "confidence": reasoning_result["confidence"],
            }

            logger.info(
                "V3 reasoning complete | intent=%s confidence=%.2f duration_ms=%.2f",
                reasoning_result.get("intent"),
                reasoning_result.get("confidence", 0.0),
                reasoning_ms,
                extra={"request_id": request_id, "use_color": True},
            )
        else:
            response_text = "Incident pattern analysis completed."
            reasoning_payload = None
            reasoning_ms = 0

        return {
            "response": response_text,
            "query": request.query,
            "focus": request.focus,
            "unusual_incidents": analysis["unusual_incidents"],
            "metrics": analysis["metrics"],
            "top_patterns": analysis["top_patterns"],
            "success_metrics": analysis["success_metrics"],
            "reasoning": reasoning_payload,
            "model": "vallm-v3-enhanced",
            "signals_used": ["embeddings", "vector_store", "sklearn", "xgboost", "pytorch"],
            "performance": {
                "local_search_ms": round(local_ms, 2),
                "analysis_ms": round(analysis_ms, 2),
                "reasoning_ms": round(reasoning_ms, 2) if request.include_reasoning else 0,
                "total_ms": round(local_ms + analysis_ms + reasoning_ms, 2)
            }
        }

    except Exception as e:
        import traceback
        request_id = getattr(req.state, "request_id", "unknown")
        tb_str = traceback.format_exc()
        logger.error(f"V3 query failed [request_id={request_id}]\n{tb_str}")
        raise HTTPException(status_code=500, detail=f"Error processing v3 query: {str(e)}")


# =============================================================================
# V2 ROUTES - Advanced NLP with spaCy and Document Extraction
# =============================================================================

# Cloud/DevOps entity patterns for V2 NLP extraction
CLOUD_PATTERNS = {
    "aws_services": [
        "ec2", "s3", "lambda", "rds", "eks", "ecs", "fargate", "cloudwatch",
        "cloudfront", "route53", "vpc", "iam", "dynamodb", "sqs", "sns",
        "api gateway", "elasticache", "aurora", "redshift", "kinesis",
        "athena", "glue", "emr", "sagemaker", "ecr", "secrets manager"
    ],
    "azure_services": [
        "aks", "azure functions", "cosmos db", "blob storage", "azure sql",
        "azure monitor", "azure devops", "azure ad", "key vault", "app service"
    ],
    "gcp_services": [
        "gke", "cloud run", "bigquery", "cloud storage", "cloud functions",
        "pub/sub", "cloud sql", "firestore", "vertex ai", "gcr"
    ],
    "kubernetes": [
        "pod", "deployment", "service", "ingress", "configmap", "secret",
        "namespace", "node", "cluster", "helm", "kubectl", "statefulset",
        "daemonset", "replicaset", "pvc", "pv", "hpa", "vpa"
    ],
    "docker": [
        "container", "image", "dockerfile", "docker-compose", "registry",
        "volume", "network", "swarm", "docker hub", "buildx"
    ],
    "devops_tools": [
        "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
        "argocd", "flux", "prometheus", "grafana", "datadog", "splunk",
        "elasticsearch", "fluentd", "jaeger", "istio", "envoy", "nginx"
    ],
    "web_frameworks": [
        "fastapi", "flask", "django", "express", "spring boot", "rails",
        "uvicorn", "gunicorn", "next.js", "nest.js", "gin", "fiber"
    ],
    "security_issues": [
        "stack trace", "debug mode", "xss", "csrf", "sql injection", "ssrf",
        "authentication", "authorization", "jwt", "oauth", "cors", "secrets",
        "credentials", "password", "api key", "token", "vulnerability", "cve"
    ],
    "error_types": [
        "timeout", "connection refused", "out of memory", "oom", "crash",
        "failed", "error", "exception", "denied", "unauthorized", "forbidden",
        "rate limit", "throttle", "latency", "slow", "high cpu", "high memory"
    ],
    "regions": [
        "us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1",
        "ap-northeast-1", "sa-east-1", "ca-central-1", "ap-south-1"
    ]
}


# V2 Pydantic models

class QueryV2Request(BaseModel):
    """V2 Query with NLP enhancement"""
    query: str
    extract_entities: bool = True
    include_recommendations: bool = True
    top_k: int = 5


class EntityExtractionRequest(BaseModel):
    """Extract cloud entities from text"""
    text: str
    entity_types: Optional[List[str]] = None  # Filter specific types


class ExtractedEntity(BaseModel):
    """Extracted entity"""
    text: str
    label: str
    category: str
    confidence: float


class DocumentAnalysis(BaseModel):
    """Document analysis result"""
    filename: str
    content_preview: str
    word_count: int
    entities: List[ExtractedEntity]
    cloud_services: List[str]
    recommendations: List[str]


# V2 NLP entity extractor

class CloudEntityExtractor:
    """Extract cloud/DevOps entities using pattern matching and spaCy"""

    def __init__(self):
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not downloaded
                pass

    def extract_pattern_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns"""
        entities = []
        text_lower = text.lower()

        for category, patterns in CLOUD_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    entities.append(ExtractedEntity(
                        text=pattern,
                        label=pattern.upper().replace(" ", "_"),
                        category=category,
                        confidence=0.9
                    ))

        # Extract IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        for match in re.finditer(ip_pattern, text):
            entities.append(ExtractedEntity(
                text=match.group(),
                label="IP_ADDRESS",
                category="network",
                confidence=0.95
            ))

        # Extract ports
        port_pattern = r':(\d{2,5})\b'
        for match in re.finditer(port_pattern, text):
            entities.append(ExtractedEntity(
                text=f":{match.group(1)}",
                label="PORT",
                category="network",
                confidence=0.85
            ))

        # Extract instance IDs (AWS style)
        instance_pattern = r'\bi-[a-f0-9]{8,17}\b'
        for match in re.finditer(instance_pattern, text, re.IGNORECASE):
            entities.append(ExtractedEntity(
                text=match.group(),
                label="EC2_INSTANCE",
                category="aws_services",
                confidence=0.95
            ))

        # Extract ARNs
        arn_pattern = r'arn:aws:[a-z0-9-]+:[a-z0-9-]*:\d*:[a-zA-Z0-9-_/:]+'
        for match in re.finditer(arn_pattern, text):
            entities.append(ExtractedEntity(
                text=match.group(),
                label="AWS_ARN",
                category="aws_services",
                confidence=0.98
            ))

        return entities

    def extract_spacy_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NLP"""
        entities = []

        if self.nlp is None:
            return entities

        doc = self.nlp(text)

        for ent in doc.ents:
            # Map spaCy labels to our categories
            category = "general"
            if ent.label_ in ["ORG", "PRODUCT"]:
                category = "organization"
            elif ent.label_ in ["GPE", "LOC"]:
                category = "location"
            elif ent.label_ in ["DATE", "TIME"]:
                category = "temporal"
            elif ent.label_ == "CARDINAL":
                category = "numeric"

            entities.append(ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                category=category,
                confidence=0.8
            ))

        return entities

    def extract_all(self, text: str) -> List[ExtractedEntity]:
        """Extract all entities from text"""
        entities = []

        # Pattern-based extraction (always works)
        entities.extend(self.extract_pattern_entities(text))

        # spaCy extraction (if available)
        entities.extend(self.extract_spacy_entities(text))

        # Remove duplicates
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e.text.lower(), e.category)
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        return unique_entities


# Global extractor instance
entity_extractor = CloudEntityExtractor()


# V2 document processing helpers

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        return "[PDF extraction not available - install PyPDF2]"

    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text_parts = []
        for page in pdf_reader.pages:
            text_parts.append(page.extract_text())
        return "\n".join(text_parts)
    except Exception as e:
        return f"[Error extracting PDF: {e}]"


def extract_text_from_image(file_content: bytes) -> tuple:
    """
    Extract text from image using OCR and analyze image properties.
    Returns: (extracted_text, image_info)
    """
    image_info = {
        "width": 0,
        "height": 0,
        "format": None,
        "mode": None,
        "has_text": False
    }

    if not PIL_AVAILABLE:
        return "[Image processing not available - install Pillow: pip install Pillow]", image_info

    try:
        img = Image.open(io.BytesIO(file_content))
        image_info = {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "has_text": False
        }

        # Try OCR if available
        if OCR_AVAILABLE:
            try:
                text = pytesseract.image_to_string(img)
                if text.strip():
                    image_info["has_text"] = True
                    return text, image_info
            except Exception:
                pass

        # If no OCR, try to describe what we can detect
        description = f"[Image: {img.width}x{img.height} {img.format or 'unknown'} format]"

        # Analyze image for common diagram patterns
        if img.mode == "RGB" or img.mode == "RGBA":
            description += "\n[Tip: Install pytesseract for OCR text extraction from diagrams]"

        return description, image_info

    except Exception as e:
        return f"[Error processing image: {e}]", image_info


def analyze_architecture_diagram(text: str, image_info: dict) -> dict:
    """Analyze extracted text for architecture/infrastructure patterns"""
    analysis = {
        "diagram_type": "unknown",
        "detected_services": [],
        "detected_connections": [],
        "infrastructure_patterns": []
    }

    text_lower = text.lower()

    # Detect diagram type
    if any(x in text_lower for x in ["aws", "amazon", "ec2", "s3", "lambda"]):
        analysis["diagram_type"] = "AWS Architecture"
    elif any(x in text_lower for x in ["azure", "microsoft", "aks", "blob"]):
        analysis["diagram_type"] = "Azure Architecture"
    elif any(x in text_lower for x in ["gcp", "google cloud", "gke", "bigquery"]):
        analysis["diagram_type"] = "GCP Architecture"
    elif any(x in text_lower for x in ["kubernetes", "k8s", "pod", "deployment", "service"]):
        analysis["diagram_type"] = "Kubernetes Architecture"
    elif any(x in text_lower for x in ["docker", "container", "dockerfile"]):
        analysis["diagram_type"] = "Container Architecture"
    elif any(x in text_lower for x in ["vpc", "subnet", "firewall", "load balancer"]):
        analysis["diagram_type"] = "Network Architecture"
    elif any(x in text_lower for x in ["ci/cd", "pipeline", "jenkins", "github actions"]):
        analysis["diagram_type"] = "CI/CD Pipeline"

    # Detect services from text
    for category, services in CLOUD_PATTERNS.items():
        for service in services:
            if service.lower() in text_lower:
                analysis["detected_services"].append({
                    "name": service,
                    "category": category
                })

    # Detect connection patterns
    connection_patterns = [
        (r"->|-->", "directional flow"),
        (r"<->|<-->", "bidirectional"),
        (r"api|rest|grpc|http", "API connection"),
        (r"database|db|rds|sql", "database connection"),
        (r"queue|sqs|kafka|pubsub", "message queue"),
        (r"cache|redis|memcached", "caching layer"),
    ]

    for pattern, conn_type in connection_patterns:
        if re.search(pattern, text_lower):
            analysis["detected_connections"].append(conn_type)

    # Detect infrastructure patterns
    if "load balancer" in text_lower or "alb" in text_lower or "nlb" in text_lower:
        analysis["infrastructure_patterns"].append("Load Balanced")
    if "auto scaling" in text_lower or "hpa" in text_lower:
        analysis["infrastructure_patterns"].append("Auto-Scaling")
    if "multi-az" in text_lower or "high availability" in text_lower:
        analysis["infrastructure_patterns"].append("High Availability")
    if "vpc" in text_lower and "subnet" in text_lower:
        analysis["infrastructure_patterns"].append("VPC Network Isolation")
    if "cdn" in text_lower or "cloudfront" in text_lower:
        analysis["infrastructure_patterns"].append("CDN Distribution")

    return analysis


def extract_text_from_file(filename: str, content: bytes) -> tuple:
    """
    Extract text from various file types.
    Returns: (text, file_info)
    """
    ext = Path(filename).suffix.lower()
    file_info = {"type": ext, "is_image": False, "image_info": None}

    # Image files
    if ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]:
        file_info["is_image"] = True
        text, image_info = extract_text_from_image(content)
        file_info["image_info"] = image_info
        return text, file_info

    # PDF files
    if ext == ".pdf":
        return extract_text_from_pdf(content), file_info

    # Text-based files
    if ext in [".txt", ".md", ".yaml", ".yml", ".json", ".tf", ".py", ".sh", ".csv",
               ".xml", ".html", ".hcl", ".toml", ".ini", ".conf", ".log"]:
        return content.decode("utf-8", errors="ignore"), file_info

    return f"[Unsupported file type: {ext}]", file_info


def generate_recommendations_from_entities(entities: List[ExtractedEntity]) -> List[str]:
    """Generate recommendations based on extracted entities"""
    recommendations = []

    categories = set(e.category for e in entities)
    labels = set(e.label.lower() for e in entities)

    # Security recommendations
    if "error_types" in categories:
        if any(x in labels for x in ["timeout", "connection_refused"]):
            recommendations.append("Consider implementing circuit breakers and retry logic for network calls")
        if any(x in labels for x in ["oom", "out_of_memory"]):
            recommendations.append("Review memory limits and implement proper resource requests in Kubernetes")
        if any(x in labels for x in ["unauthorized", "forbidden", "denied"]):
            recommendations.append("Audit IAM policies and RBAC configurations for proper access control")

    # Kubernetes recommendations
    if "kubernetes" in categories:
        recommendations.append("Ensure HPA/VPA is configured for auto-scaling workloads")
        recommendations.append("Implement pod disruption budgets for high availability")

    # Docker recommendations
    if "docker" in categories:
        recommendations.append("Use multi-stage builds to reduce image size")
        recommendations.append("Enable image vulnerability scanning in your container registry")

    # AWS recommendations
    if "aws_services" in categories:
        recommendations.append("Enable CloudWatch alarms for critical metrics")
        recommendations.append("Consider using AWS Cost Explorer to identify optimization opportunities")

    # DevOps tools
    if "devops_tools" in categories:
        if "prometheus" in labels or "grafana" in labels:
            recommendations.append("Define SLOs and set up alerting based on error budgets")
        if "terraform" in labels:
            recommendations.append("Use Terraform workspaces for environment separation")

    # Web frameworks
    if "web_frameworks" in categories:
        if "fastapi" in labels or "uvicorn" in labels:
            recommendations.append("Set debug=False in production and configure proper exception handlers")
            recommendations.append("Use structured logging with correlation IDs for tracing")
        if "django" in labels or "flask" in labels:
            recommendations.append("Ensure DEBUG=False and configure proper security middleware")

    # Security issues
    if "security_issues" in categories:
        if any(x in labels for x in ["stack_trace", "debug_mode"]):
            recommendations.append("Disable debug mode in production to prevent information disclosure")
        if any(x in labels for x in ["xss", "csrf"]):
            recommendations.append("Implement proper security headers and CSRF protection")
        if any(x in labels for x in ["credentials", "password", "api_key", "secrets"]):
            recommendations.append("Use a secrets manager (AWS Secrets Manager, HashiCorp Vault) for sensitive data")

    return recommendations[:5]  # Limit to 5


# V2 API endpoints

@router_v2.post("/query")
async def query_v2(
    request: QueryV2Request,
    req: Request
):
    """
    V2 Query endpoint with NLP entity extraction

    Features:
    - Extracts cloud/DevOps entities from query
    - Provides context-aware recommendations
    - Enhanced search with entity filtering
    """
    try:
        vector_store = req.app.state.vector_store
        reasoning_engine = req.app.state.reasoning_engine

        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")

        # Extract entities from query
        entities = []
        if request.extract_entities:
            entities = entity_extractor.extract_all(request.query)

        # Perform vector search
        search_results = await vector_store.search(
            query=request.query,
            top_k=request.top_k
        )

        # Perform reasoning
        reasoning_result = None
        if reasoning_engine:
            reasoning_result = await reasoning_engine.reason(
                query=request.query,
                max_steps=5
            )

        # Generate recommendations based on entities
        recommendations = []
        if request.include_recommendations:
            recommendations = generate_recommendations_from_entities(entities)

        # Build a more useful response that includes the actual recommendation content
        response_text = ""
        if search_results:
            top_result = search_results[0]
            doc_text = top_result.get('document', '')
            score = top_result.get('score', 0)

            response_text = f"Found matching recommendation (score: {score:.2f}):\n\n"
            response_text += doc_text

            if len(search_results) > 1:
                response_text += f"\n\nAlso found {len(search_results) - 1} related recommendations."
        else:
            response_text = reasoning_result['final_answer'] if reasoning_result else "No matching recommendations found."

        return {
            "query": request.query,
            "response": response_text,
            "entities": [e.dict() for e in entities],
            "entity_summary": {
                "total": len(entities),
                "by_category": {
                    cat: len([e for e in entities if e.category == cat])
                    for cat in set(e.category for e in entities)
                }
            },
            "context": [
                {
                    "document": r['document'],
                    "score": r['score'],
                    "metadata": r.get('metadata', {})
                }
                for r in search_results
            ],
            "recommendations": recommendations,
            "nlp_info": {
                "spacy_available": SPACY_AVAILABLE,
                "spacy_model": "en_core_web_sm" if entity_extractor.nlp else None
            },
            "model": "vallm-v2-nlp",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router_v2.post("/extract")
async def extract_entities(request: EntityExtractionRequest):
    """
    Extract cloud/DevOps entities from text

    Recognizes:
    - AWS/Azure/GCP services
    - Kubernetes resources
    - Docker components
    - DevOps tools
    - Error types
    - IP addresses, ports, instance IDs
    """
    try:
        entities = entity_extractor.extract_all(request.text)

        # Filter by types if specified
        if request.entity_types:
            entities = [e for e in entities if e.category in request.entity_types]

        return {
            "text_preview": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            "entities": [e.dict() for e in entities],
            "summary": {
                "total_entities": len(entities),
                "categories": list(set(e.category for e in entities)),
                "cloud_services": [e.text for e in entities if "services" in e.category],
                "errors_detected": [e.text for e in entities if e.category == "error_types"]
            },
            "nlp_info": {
                "spacy_available": SPACY_AVAILABLE,
                "pattern_matching": True
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting entities: {str(e)}")


@router_v2.post("/upload")
async def upload_document(
    req: Request,
    file: UploadFile = File(...),
    analyze: bool = Form(default=True)
):
    """
    Upload and analyze cloud/DevOps documents and images

    Supported formats:
    - Images (.png, .jpg, .jpeg, .gif, .bmp, .webp) - Architecture diagrams
    - PDF (.pdf) - Documentation
    - Text (.txt, .md) - Readme, notes
    - Config files (.yaml, .yml, .json, .tf, .hcl) - Infrastructure as Code
    - Scripts (.py, .sh) - Automation scripts
    - Logs (.log, .csv) - Log files
    """
    try:
        # Read file content
        content = await file.read()

        # Extract text and file info
        text, file_info = extract_text_from_file(file.filename, content)

        if not analyze:
            return {
                "filename": file.filename,
                "size_bytes": len(content),
                "file_info": file_info,
                "extracted_text": text[:2000] + "..." if len(text) > 2000 else text
            }

        # For images, do architecture analysis
        diagram_analysis = None
        if file_info.get("is_image"):
            diagram_analysis = analyze_architecture_diagram(text, file_info.get("image_info", {}))

        # Extract entities
        entities = entity_extractor.extract_all(text)

        # Get cloud services mentioned
        cloud_services = list(set(
            e.text for e in entities
            if "services" in e.category
        ))

        # Add services from diagram analysis
        if diagram_analysis:
            for svc in diagram_analysis.get("detected_services", []):
                if svc["name"] not in cloud_services:
                    cloud_services.append(svc["name"])

        # Generate recommendations
        recommendations = generate_recommendations_from_entities(entities)

        # Add diagram-specific recommendations
        if diagram_analysis:
            if diagram_analysis.get("diagram_type") == "Kubernetes Architecture":
                recommendations.append("Consider implementing NetworkPolicies for pod-to-pod security")
            if "High Availability" not in diagram_analysis.get("infrastructure_patterns", []):
                recommendations.append("Consider adding multi-AZ deployment for high availability")
            if "CDN Distribution" not in diagram_analysis.get("infrastructure_patterns", []):
                recommendations.append("Consider using a CDN for static content delivery")

        # Search for related content in vector store
        vector_store = req.app.state.vector_store
        related_docs = []
        if vector_store and cloud_services:
            search_query = " ".join(cloud_services[:5])
            search_results = await vector_store.search(query=search_query, top_k=3)
            related_docs = [r['document'][:200] for r in search_results]

        response = {
            "filename": file.filename,
            "file_type": Path(file.filename).suffix,
            "size_bytes": len(content),
            "is_image": file_info.get("is_image", False),
            "word_count": len(text.split()),
            "content_preview": text[:500] + "..." if len(text) > 500 else text,
            "analysis": {
                "entities": [e.dict() for e in entities[:20]],
                "entity_count": len(entities),
                "cloud_services": cloud_services,
                "error_indicators": [e.text for e in entities if e.category == "error_types"]
            },
            "recommendations": recommendations[:5],
            "related_documents": related_docs,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add image-specific info
        if file_info.get("is_image"):
            response["image_analysis"] = {
                "dimensions": f"{file_info['image_info']['width']}x{file_info['image_info']['height']}" if file_info.get("image_info") else "unknown",
                "format": file_info['image_info'].get('format') if file_info.get("image_info") else "unknown",
                "ocr_available": OCR_AVAILABLE,
                "text_detected": file_info['image_info'].get('has_text', False) if file_info.get("image_info") else False,
                "diagram_analysis": diagram_analysis
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router_v2.get("/status")
async def nlp_status():
    """Check NLP system status and capabilities"""
    return {
        "status": "online",
        "capabilities": {
            "spacy_nlp": {
                "available": SPACY_AVAILABLE,
                "model_loaded": entity_extractor.nlp is not None,
                "install": "pip install spacy && python -m spacy download en_core_web_sm"
            },
            "pdf_extraction": {
                "available": PDF_AVAILABLE,
                "install": "pip install PyPDF2"
            },
            "image_processing": {
                "available": PIL_AVAILABLE,
                "install": "pip install Pillow"
            },
            "ocr_text_extraction": {
                "available": OCR_AVAILABLE,
                "install": "pip install pytesseract (requires Tesseract OCR installed)"
            },
            "pattern_matching": {
                "available": True,
                "install": "Built-in (no installation needed)"
            }
        },
        "supported_file_types": {
            "images": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"],
            "documents": [".pdf", ".txt", ".md"],
            "config": [".yaml", ".yml", ".json", ".tf", ".hcl", ".toml", ".ini"],
            "code": [".py", ".sh", ".js", ".ts"],
            "logs": [".log", ".csv"]
        },
        "entity_categories": list(CLOUD_PATTERNS.keys()),
        "entity_patterns_count": {
            cat: len(patterns) for cat, patterns in CLOUD_PATTERNS.items()
        },
        "total_patterns": sum(len(p) for p in CLOUD_PATTERNS.values())
    }

