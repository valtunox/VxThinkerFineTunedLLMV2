"""
VaLLM Specialist Model - AI Models Package.

Author: Joel Otepa Wembo
https://joelwembo.com

Exposes custom ML models for document analysis: document parser, document matching,
entity scoring, verification scoring, financial prediction (app/data/models).
"""

# Conditionally import models to avoid breaking if optional dependencies are missing
try:
    from .base_model import BaseModel
    BASE_MODEL_AVAILABLE = True
except (ImportError, Exception) as e:
    BaseModel = None
    BASE_MODEL_AVAILABLE = False
    import logging
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"BaseModel not available: {e}")

import logging as _logging
_logger = _logging.getLogger(__name__)

try:
    from .document_parser import DocumentParserModel
    DOCUMENT_PARSER_AVAILABLE = True
except (ImportError, Exception) as e:
    DocumentParserModel = None
    DOCUMENT_PARSER_AVAILABLE = False
    _logger.warning(f"DocumentParserModel not available: {e}")

try:
    from .document_matching import DocumentMatchingModel
    DOCUMENT_MATCHING_AVAILABLE = True
except (ImportError, Exception) as e:
    DocumentMatchingModel = None
    DOCUMENT_MATCHING_AVAILABLE = False
    _logger.warning(f"DocumentMatchingModel not available: {e}")

try:
    from .entity_scoring import EntityScoringModel
    ENTITY_SCORING_AVAILABLE = True
except (ImportError, Exception) as e:
    EntityScoringModel = None
    ENTITY_SCORING_AVAILABLE = False
    _logger.warning(f"EntityScoringModel not available: {e}")

try:
    from .verification_scoring import VerificationScoringModel
    VERIFICATION_SCORING_AVAILABLE = True
except (ImportError, Exception) as e:
    VerificationScoringModel = None
    VERIFICATION_SCORING_AVAILABLE = False
    _logger.warning(f"VerificationScoringModel not available: {e}")

try:
    from .financial_prediction import FinancialPredictionModel
    FINANCIAL_PREDICTION_AVAILABLE = True
except (ImportError, Exception) as e:
    FinancialPredictionModel = None
    FINANCIAL_PREDICTION_AVAILABLE = False
    _logger.warning(f"FinancialPredictionModel not available: {e}")

__all__ = []

if BASE_MODEL_AVAILABLE:
    __all__.append("BaseModel")
if DOCUMENT_PARSER_AVAILABLE:
    __all__.append("DocumentParserModel")
if DOCUMENT_MATCHING_AVAILABLE:
    __all__.append("DocumentMatchingModel")
if ENTITY_SCORING_AVAILABLE:
    __all__.append("EntityScoringModel")
if VERIFICATION_SCORING_AVAILABLE:
    __all__.append("VerificationScoringModel")
if FINANCIAL_PREDICTION_AVAILABLE:
    __all__.append("FinancialPredictionModel")
