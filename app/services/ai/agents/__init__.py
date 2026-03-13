"""
VaLLM Specialist Model - AI Agents
LangGraph orchestrated AI agents for document analysis and business intelligence.
"""

from .base_agent import BaseAgent

# Conditionally import agents to avoid breaking if optional dependencies are missing
try:
    from .base_agent import DocumentAnalysisAgent
    DOCUMENT_ANALYSIS_AVAILABLE = True
except (ImportError, Exception):
    DocumentAnalysisAgent = None
    DOCUMENT_ANALYSIS_AVAILABLE = False

try:
    from .base_agent import VerificationAgent
    VERIFICATION_AVAILABLE = True
except (ImportError, Exception):
    VerificationAgent = None
    VERIFICATION_AVAILABLE = False

try:
    from .base_agent import FinancialAnalysisAgent
    FINANCIAL_ANALYSIS_AVAILABLE = True
except (ImportError, Exception):
    FinancialAnalysisAgent = None
    FINANCIAL_ANALYSIS_AVAILABLE = False

try:
    from .base_agent import RecommendationAgent
    RECOMMENDATION_AVAILABLE = True
except (ImportError, Exception):
    RecommendationAgent = None
    RECOMMENDATION_AVAILABLE = False

# Conditionally import WorkflowOrchestrator only if langgraph is available
try:
    from .base_agent import WorkflowOrchestrator
    WORKFLOW_ORCHESTRATOR_AVAILABLE = True
except (ImportError, Exception):
    WorkflowOrchestrator = None
    WORKFLOW_ORCHESTRATOR_AVAILABLE = False

__all__ = ["BaseAgent"]

if DOCUMENT_ANALYSIS_AVAILABLE:
    __all__.append("DocumentAnalysisAgent")
if VERIFICATION_AVAILABLE:
    __all__.append("VerificationAgent")
if FINANCIAL_ANALYSIS_AVAILABLE:
    __all__.append("FinancialAnalysisAgent")
if RECOMMENDATION_AVAILABLE:
    __all__.append("RecommendationAgent")
if WORKFLOW_ORCHESTRATOR_AVAILABLE:
    __all__.append("WorkflowOrchestrator")
