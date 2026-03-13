"""
VaLLM Specialist Model - Fraud Detection Service.

Author: Joel Otepa Wembo
https://joelwembo.com

Isolation Forest anomaly detection for timesheet and activity fraud detection.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

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
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    IsolationForest = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

from core.logging import performance_logger

logger = logging.getLogger(__name__)


class FraudDetectionService:
    """Fraud detection service using Isolation Forest"""
    
    def __init__(self):
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn is not installed. FraudDetectionService will not be functional.")
            self.fraud_model = None
            self.scaler = None
        else:
            self.fraud_model = IsolationForest(
                contamination=0.1,  # Expected 10% fraud rate
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()
        
        if not NUMPY_AVAILABLE:
            logger.warning("numpy is not installed. FraudDetectionService will not be functional.")
        
        # Performance tracking
        self.detection_count = 0
        self.fraud_detected_count = 0
        self.total_processing_time = 0.0
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize fraud detection service"""
        try:
            self.is_initialized = True
            logger.info("Fraud Detection Service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Fraud Detection Service: {e}")
            return False
    
    async def detect_fraud(self, timesheet_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect fraudulent timesheet entries"""
        try:
            if not SKLEARN_AVAILABLE or self.fraud_model is None:
                logger.error("Fraud detection not available - scikit-learn not installed")
                return []
            
            if not NUMPY_AVAILABLE:
                logger.error("Fraud detection not available - numpy not installed")
                return []
            
            start_time = datetime.utcnow()
            self.detection_count += 1
            
            if not timesheet_data:
                return []
            
            # Extract features for fraud detection
            features = self._extract_fraud_features(timesheet_data)
            
            # Detect anomalies
            fraud_scores = self.fraud_model.decision_function(features)
            fraud_predictions = self.fraud_model.predict(features)
            
            # Process results
            results = []
            for i, entry in enumerate(timesheet_data):
                is_fraud = fraud_predictions[i] == -1
                fraud_score = fraud_scores[i]
                
                result = {
                    "entry_id": entry.get("id", i),
                    "is_fraud": is_fraud,
                    "fraud_score": float(fraud_score),
                    "risk_level": self._calculate_risk_level(fraud_score),
                    "fraud_indicators": self._identify_fraud_indicators(entry, fraud_score)
                }
                
                results.append(result)
                
                if is_fraud:
                    self.fraud_detected_count += 1
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.total_processing_time += processing_time
            
            return results
            
        except Exception as e:
            logger.error(f"Fraud detection failed: {e}")
            raise
    
    def _extract_fraud_features(self, timesheet_data: List[Dict[str, Any]]):
        """Extract features for fraud detection"""
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is not installed")
        
        features = []
        
        for entry in timesheet_data:
            feature_vector = []
            
            # Hours worked
            hours_worked = entry.get("hours_worked", 0)
            feature_vector.append(hours_worked)
            
            # Overtime hours
            overtime_hours = entry.get("overtime_hours", 0)
            feature_vector.append(overtime_hours)
            
            # Weekend work
            is_weekend = self._is_weekend(entry.get("date"))
            feature_vector.append(1 if is_weekend else 0)
            
            # Holiday work
            is_holiday = self._is_holiday(entry.get("date"))
            feature_vector.append(1 if is_holiday else 0)
            
            # Time deviation from normal
            normal_hours = 8  # Standard work day
            time_deviation = abs(hours_worked - normal_hours)
            feature_vector.append(time_deviation)
            
            # Location consistency (simplified)
            location_score = self._calculate_location_consistency(entry)
            feature_vector.append(location_score)
            
            # Time pattern anomaly
            time_pattern_score = self._calculate_time_pattern_score(entry)
            feature_vector.append(time_pattern_score)
            
            features.append(feature_vector)
        
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is not installed")
        
        return np.array(features)
    
    def _is_weekend(self, date_str: str) -> bool:
        """Check if date is weekend"""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.weekday() >= 5  # Saturday = 5, Sunday = 6
        except:
            return False
    
    def _is_holiday(self, date_str: str) -> bool:
        """Check if date is holiday (simplified)"""
        # This would check against a holiday calendar in production
        holidays = ["2024-01-01", "2024-12-25"]  # Example holidays
        return date_str in holidays
    
    def _calculate_location_consistency(self, entry: Dict[str, Any]) -> float:
        """Calculate location consistency score"""
        # This would compare with historical locations in production
        return 0.8  # Mock score
    
    def _calculate_time_pattern_score(self, entry: Dict[str, Any]) -> float:
        """Calculate time pattern anomaly score"""
        # This would analyze time patterns in production
        return 0.5  # Mock score
    
    def _calculate_risk_level(self, fraud_score: float) -> str:
        """Calculate risk level based on fraud score"""
        if fraud_score < -0.5:
            return "high"
        elif fraud_score < -0.2:
            return "medium"
        else:
            return "low"
    
    def _identify_fraud_indicators(self, entry: Dict[str, Any], 
                                 fraud_score: float) -> List[str]:
        """Identify specific fraud indicators"""
        indicators = []
        
        hours_worked = entry.get("hours_worked", 0)
        if hours_worked > 16:
            indicators.append("excessive_hours")
        
        if entry.get("overtime_hours", 0) > 8:
            indicators.append("excessive_overtime")
        
        if self._is_weekend(entry.get("date")):
            indicators.append("weekend_work")
        
        if fraud_score < -0.3:
            indicators.append("anomalous_pattern")
        
        return indicators
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get fraud detection statistics"""
        fraud_rate = (self.fraud_detected_count / self.detection_count * 100) if self.detection_count > 0 else 0
        avg_processing_time = (self.total_processing_time / self.detection_count) if self.detection_count > 0 else 0
        
        return {
            "total_detections": self.detection_count,
            "fraud_detected": self.fraud_detected_count,
            "fraud_rate": fraud_rate,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time
        }


# Global fraud detection service instance (only create if dependencies are available)
try:
    fraud_detection_service = FraudDetectionService()
except Exception as e:
    logger.warning(f"Failed to initialize global fraud detection service instance: {e}")
    logger.warning("Fraud detection service will be unavailable. Some features may not work.")
    fraud_detection_service = None
