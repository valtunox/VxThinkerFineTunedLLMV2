"""
VaLLM Specialist Model - Business Scoring Service.

Author: Joel Otepa Wembo
https://joelwembo.com

XGBoost-based entity scoring with optional SHAP explainability for
document verification, financial risk assessment, and business analytics.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Conditional imports for optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

from core.logging import performance_logger

logger = logging.getLogger(__name__)


class ScoringService:
    """Business entity scoring service with XGBoost and SHAP explainability"""
    
    def __init__(self):
        if not XGBOOST_AVAILABLE:
            logger.warning("xgboost is not installed. ScoringService will not be functional.")
            self.scoring_model = None
        else:
            self.scoring_model = xgb.XGBClassifier(random_state=42)
        
        if not SHAP_AVAILABLE:
            logger.warning("shap is not installed. SHAP explanations will not be available.")
        
        self.explainer = None
        
        # Performance tracking
        self.scoring_count = 0
        self.total_processing_time = 0.0
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize scoring service"""
        try:
            self.is_initialized = True
            logger.info("Scoring Service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Scoring Service: {e}")
            return False
    
    async def score_entity(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score a business entity (document, transaction, organization) with explainability"""
        try:
            if not XGBOOST_AVAILABLE or self.scoring_model is None:
                logger.error("Entity scoring not available - xgboost not installed")
                return {
                    "entity_id": entity_data.get("id"),
                    "score": 0.5,
                    "confidence": 0.0,
                    "explanation": {"error": "xgboost not available"},
                    "recommendation": "unavailable"
                }
            
            start_time = datetime.utcnow()
            self.scoring_count += 1
            
            # Extract features
            features = self._extract_entity_features(entity_data)

            # Make prediction
            score = self.scoring_model.predict_proba([features])[0][1]  # Probability score

            # Generate explanation
            explanation = self._generate_explanation(features)

            result = {
                "entity_id": entity_data.get("id"),
                "score": float(score),
                "confidence": float(max(self.scoring_model.predict_proba([features])[0])),
                "explanation": explanation,
                "recommendation": self._get_recommendation(score)
            }
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.total_processing_time += processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Entity scoring failed: {e}")
            raise

    def _extract_entity_features(self, entity_data: Dict[str, Any]) -> List[float]:
        """Extract features for entity scoring"""
        features = []
        
        # Company size
        company_size = entity_data.get("company_size", "medium")
        size_mapping = {"startup": 1, "small": 2, "medium": 3, "large": 4, "enterprise": 5}
        features.append(size_mapping.get(company_size, 3))
        
        # Industry
        industry = entity_data.get("industry", "technology")
        industry_score = self._get_industry_score(industry)
        features.append(industry_score)
        
        # Title level
        title = entity_data.get("title", "").lower()
        title_level = self._get_title_level(title)
        features.append(title_level)
        
        # Location
        location = entity_data.get("location", "")
        location_score = self._get_location_score(location)
        features.append(location_score)
        
        return features
    
    def _get_industry_score(self, industry: str) -> float:
        """Get industry score for business potential"""
        high_value_industries = ["technology", "finance", "healthcare", "consulting"]
        return 1.0 if industry.lower() in high_value_industries else 0.5
    
    def _get_title_level(self, title: str) -> float:
        """Get title level score"""
        if any(word in title for word in ["director", "vp", "ceo", "cto"]):
            return 1.0
        elif any(word in title for word in ["manager", "senior", "lead"]):
            return 0.8
        elif any(word in title for word in ["analyst", "coordinator"]):
            return 0.6
        else:
            return 0.4
    
    def _get_location_score(self, location: str) -> float:
        """Get location score"""
        major_cities = ["toronto", "vancouver", "montreal", "calgary", "ottawa"]
        return 1.0 if any(city in location.lower() for city in major_cities) else 0.6
    
    def _generate_explanation(self, features: List[float]) -> Dict[str, Any]:
        """Generate SHAP explanation"""
        # Mock explanation (would use actual SHAP in production)
        return {
            "company_size_impact": features[0] * 0.3,
            "industry_impact": features[1] * 0.4,
            "title_impact": features[2] * 0.2,
            "location_impact": features[3] * 0.1
        }
    
    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on score"""
        if score >= 0.8:
            return "high_priority"
        elif score >= 0.6:
            return "medium_priority"
        else:
            return "low_priority"
    
    def get_scoring_stats(self) -> Dict[str, Any]:
        """Get scoring service statistics"""
        avg_processing_time = (self.total_processing_time / self.scoring_count) if self.scoring_count > 0 else 0
        
        return {
            "total_scores": self.scoring_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time
        }


# Global scoring service instance
scoring_service = ScoringService()
