"""
Industry Registry — per-industry configuration for the multi-industry engine.

Each industry defines system prompts, dataset sources, evaluation tasks,
embedding text templates, supported file types, and domain keywords.
"""

from enum import Enum
from typing import Any, Dict


class Industry(str, Enum):
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    CLOUD = "cloud"
    AUTOMATION = "automation"
    CUSTOMER_SERVICE = "customer_service"


INDUSTRY_REGISTRY: Dict[Industry, Dict[str, Any]] = {
    Industry.HEALTHCARE: {
        "system_prompt": (
            "You are an AI assistant for healthcare operations and clinical decision support. "
            "Analyze the following medical record and provide insights.\n\n"
        ),
        "embedding_text_template": (
            "Healthcare record — {fields}\n\nClinical summary:"
        ),
        "dataset_sources": [
            "medmcqa",
            "pubmedqa",
            "medical_meadow",
            "bigbio/med_qa",
        ],
        "evaluation_tasks": ["medqa", "pubmedqa"],
        "supported_file_types": [".csv", ".json", ".txt", ".pdf", ".hl7", ".fhir"],
        "domain_keywords": {
            "patient", "diagnosis", "treatment", "medication", "clinical",
            "ehr", "fhir", "hl7", "icd", "cpt", "ndc", "snomed", "loinc",
            "vitals", "lab", "radiology", "pharmacy", "nursing", "discharge",
        },
        "json_category_prompts": {
            "records": "Clinical record. Review the following patient data:",
            "data": "Healthcare data:",
            "items": "Medical item:",
            "entries": "Clinical entry:",
        },
        "txt_context_label": "healthcare knowledge",
    },
    Industry.FINANCE: {
        "system_prompt": (
            "You are an AI assistant for financial analysis and risk management. "
            "Analyze the following financial record and provide insights.\n\n"
        ),
        "embedding_text_template": (
            "Financial record — {fields}\n\nAnalysis:"
        ),
        "dataset_sources": [
            "FinGPT/fingpt-sentiment-train",
            "financial_phrasebank",
            "AdaptLLM/finance-tasks",
        ],
        "evaluation_tasks": ["finbench", "flare"],
        "supported_file_types": [".csv", ".json", ".txt", ".pdf", ".xlsx"],
        "domain_keywords": {
            "portfolio", "equity", "bond", "derivative", "risk", "hedge",
            "compliance", "aml", "kyc", "sec", "finra", "gaap", "ifrs",
            "trading", "settlement", "margin", "volatility", "yield",
        },
        "json_category_prompts": {
            "transactions": "Financial transaction. Analyze the following:",
            "records": "Financial record:",
            "data": "Financial data:",
            "items": "Financial item:",
        },
        "txt_context_label": "financial knowledge",
    },
    Industry.CLOUD: {
        "system_prompt": (
            "You are an AI assistant for cloud provisioning and deployment. "
            "Analyze the following telemetry record and provide insights.\n\n"
        ),
        "embedding_text_template": (
            "Cloud operations record — {fields}\n\nAnswer:"
        ),
        "dataset_sources": [
            "cloud_deployments.csv",
            "deployments.json",
        ],
        "evaluation_tasks": ["cloud_ops_bench"],
        "supported_file_types": [".csv", ".json", ".txt", ".pdf"],
        "domain_keywords": {
            "aws", "azure", "gcp", "kubernetes", "eks", "aks", "gke",
            "vpc", "iam", "ec2", "s3", "rds", "cloud", "container",
            "docker", "k8s", "cicd", "terraform", "ansible", "devops",
        },
        "json_category_prompts": {
            "use_cases": "Provisioning use case. Deploy or create the following:",
            "deployments": "Deployment configuration. Provision the following:",
            "data": "Cloud provisioning data:",
            "items": "Provisioning item:",
            "records": "Deployment record:",
        },
        "txt_context_label": "cloud operations and provisioning",
    },
    Industry.AUTOMATION: {
        "system_prompt": (
            "You are an AI assistant for business process automation and workflow optimization. "
            "Analyze the following process record and provide insights.\n\n"
        ),
        "embedding_text_template": (
            "Automation record — {fields}\n\nOptimization summary:"
        ),
        "dataset_sources": [
            "robocorp/rpa-datasets",
        ],
        "evaluation_tasks": ["automation_bench"],
        "supported_file_types": [".csv", ".json", ".txt", ".pdf", ".bpmn"],
        "domain_keywords": {
            "rpa", "workflow", "process", "bot", "automation", "trigger",
            "schedule", "orchestration", "task", "pipeline", "etl",
            "integration", "api", "webhook", "batch", "queue",
        },
        "json_category_prompts": {
            "workflows": "Automation workflow. Execute the following:",
            "processes": "Business process configuration:",
            "data": "Automation data:",
            "items": "Automation item:",
            "records": "Process record:",
        },
        "txt_context_label": "business process automation",
    },
    Industry.CUSTOMER_SERVICE: {
        "system_prompt": (
            "You are an AI assistant for customer service and support operations. "
            "Analyze the following support record and provide insights.\n\n"
        ),
        "embedding_text_template": (
            "Customer service record — {fields}\n\nResolution summary:"
        ),
        "dataset_sources": [
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            "MohitGoel/Customer_Support_Dataset",
        ],
        "evaluation_tasks": ["customer_service_bench"],
        "supported_file_types": [".csv", ".json", ".txt", ".pdf"],
        "domain_keywords": {
            "ticket", "support", "customer", "complaint", "resolution",
            "sla", "escalation", "satisfaction", "nps", "csat", "faq",
            "helpdesk", "agent", "response", "feedback", "churn",
        },
        "json_category_prompts": {
            "tickets": "Support ticket. Resolve the following:",
            "interactions": "Customer interaction record:",
            "data": "Customer service data:",
            "items": "Support item:",
            "records": "Service record:",
        },
        "txt_context_label": "customer service",
    },
}


def get_industry_config(industry: Industry) -> Dict[str, Any]:
    """Return the full configuration dict for the given industry."""
    if industry not in INDUSTRY_REGISTRY:
        raise ValueError(f"Unknown industry: {industry}. Valid: {list(Industry)}")
    return INDUSTRY_REGISTRY[industry]
