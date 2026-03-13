"""
VaLLM Specialist Model - Base Model Class.

Author: Joel Otepa Wembo
https://joelwembo.com

Abstract base class for all AI models: common paths (app/data/models),
metrics, save/load, and training interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Conditionally import sklearn to avoid breaking if it's not installed
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    accuracy_score = None
    precision_score = None
    recall_score = None
    f1_score = None
    SKLEARN_METRICS_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

try:
    from core.settings import get_settings
    from core.logging import ai_logger, performance_logger
except ImportError:
    from app.core.settings import get_settings
    from app.core.logging import ai_logger, performance_logger

settings = get_settings()
logger = logging.getLogger(__name__)


class ModelMetrics:
    """Model performance metrics"""
    
    def __init__(self):
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.confusion_matrix = None
        self.classification_report = None
        self.training_time = 0.0
        self.inference_time = 0.0
        self.model_size_mb = 0.0
        self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "model_size_mb": self.model_size_mb,
            "last_updated": self.last_updated.isoformat()
        }


class BaseModel(ABC):
    """Abstract base class for all AI models"""
    
    def __init__(self, model_id: str, model_name: str, model_type: str, version: str = "1.0"):
        self.model_id = model_id
        self.model_name = model_name
        self.model_type = model_type
        self.version = version
        
        # Model state
        self.is_initialized = False
        self.is_trained = False
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        
        # Performance tracking
        self.metrics = ModelMetrics()
        self.training_history = []
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # Model paths (always under app/data/models; base_model.py is in app/services/ai/ml/models/)
        app_root = Path(__file__).resolve().parents[4]
        self.model_dir = app_root / "data" / "models" / model_id
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.model_dir / f"{model_name}_v{version}.joblib"
        self.preprocessor_path = self.model_dir / f"{model_name}_preprocessor_v{version}.joblib"
        self.metrics_path = self.model_dir / f"{model_name}_metrics_v{version}.json"
        self.config_path = self.model_dir / f"{model_name}_config_v{version}.json"
        
        logger.info(f"Initialized {self.__class__.__name__}: {model_name} v{version}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the model"""
        pass
    
    @abstractmethod
    async def train(self, training_data: Dict[str, Any], 
                   validation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train the model"""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Any) -> Dict[str, Any]:
        """Make predictions"""
        pass
    
    @abstractmethod
    async def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model performance"""
        pass
    
    async def load_model(self) -> bool:
        """Load pre-trained model"""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                return False
            
            # Load model
            self.model = joblib.load(self.model_path)
            
            # Load preprocessor if exists
            if self.preprocessor_path.exists():
                self.preprocessor = joblib.load(self.preprocessor_path)
            
            # Load metrics if exists
            if self.metrics_path.exists():
                with open(self.metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    self.metrics.accuracy = metrics_data.get("accuracy", 0.0)
                    self.metrics.precision = metrics_data.get("precision", 0.0)
                    self.metrics.recall = metrics_data.get("recall", 0.0)
                    self.metrics.f1_score = metrics_data.get("f1_score", 0.0)
            
            # Load config if exists
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.feature_columns = config.get("feature_columns")
            
            self.is_trained = True
            logger.info(f"Loaded model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            return False
    
    async def save_model(self) -> bool:
        """Save trained model"""
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            # Save model
            joblib.dump(self.model, self.model_path)
            
            # Save preprocessor if exists
            if self.preprocessor is not None:
                joblib.dump(self.preprocessor, self.preprocessor_path)
            
            # Save metrics
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics.to_dict(), f, indent=2)
            
            # Save config
            config = {
                "model_id": self.model_id,
                "model_name": self.model_name,
                "model_type": self.model_type,
                "version": self.version,
                "feature_columns": self.feature_columns,
                "is_trained": self.is_trained,
                "last_saved": datetime.utcnow().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {self.model_name}: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "is_initialized": self.is_initialized,
            "is_trained": self.is_trained,
            "model_path": str(self.model_path),
            "metrics": self.metrics.to_dict(),
            "inference_count": self.inference_count,
            "average_inference_time": self.total_inference_time / max(self.inference_count, 1)
        }
    
    def get_metrics(self) -> ModelMetrics:
        """Get model metrics"""
        return self.metrics
    
    def update_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      training_time: float = 0.0) -> None:
        """Update model metrics"""
        try:
            if SKLEARN_METRICS_AVAILABLE:
                self.metrics.accuracy = accuracy_score(y_true, y_pred)
                self.metrics.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                self.metrics.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                self.metrics.f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                logger.warning("sklearn metrics not available - skipping metric calculation")
                # Calculate basic metrics manually if sklearn is not available
                correct = sum(y_true == y_pred)
                self.metrics.accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0
                self.metrics.precision = 0.0
                self.metrics.recall = 0.0
                self.metrics.f1_score = 0.0
            self.metrics.training_time = training_time
            self.metrics.last_updated = datetime.utcnow()
            
            # Calculate model size
            if self.model_path.exists():
                self.metrics.model_size_mb = self.model_path.stat().st_size / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def log_inference(self, inference_time: float) -> None:
        """Log inference performance"""
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.metrics.inference_time = inference_time
        
        # Log to performance logger
        performance_logger.log_ai_model_inference(
            model_name=self.model_name,
            input_tokens=0,  # Would be calculated based on input
            output_tokens=0,  # Would be calculated based on output
            duration_ms=inference_time * 1000
        )
    
    def preprocess_input(self, input_data: Any) -> Any:
        """Preprocess input data"""
        if self.preprocessor is None:
            return input_data
        
        try:
            return self.preprocessor.transform(input_data)
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return input_data
    
    def postprocess_output(self, output_data: Any) -> Any:
        """Postprocess output data"""
        # Override in subclasses if needed
        return output_data
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        # Override in subclasses for specific validation
        return input_data is not None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        if self.model is None:
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                if self.feature_columns:
                    return dict(zip(self.feature_columns, self.model.feature_importances_))
                else:
                    return dict(enumerate(self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                if self.feature_columns:
                    return dict(zip(self.feature_columns, self.model.coef_[0]))
                else:
                    return dict(enumerate(self.model.coef_[0]))
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
        
        return None
    
    def get_model_explanation(self, input_data: Any) -> Optional[Dict[str, Any]]:
        """Get model explanation for input"""
        # This would integrate with SHAP or LIME for explainability
        explanation = {
            "feature_importance": self.get_feature_importance(),
            "prediction_confidence": 0.8,  # Would be calculated
            "key_factors": ["factor1", "factor2"],  # Would be extracted
            "explanation_type": "feature_importance"
        }
        
        return explanation
    
    async def health_check(self) -> Dict[str, Any]:
        """Check model health"""
        return {
            "model_id": self.model_id,
            "status": "healthy" if self.is_initialized and self.is_trained else "unhealthy",
            "is_initialized": self.is_initialized,
            "is_trained": self.is_trained,
            "model_exists": self.model_path.exists(),
            "metrics": self.metrics.to_dict(),
            "last_inference": self.inference_count,
            "average_inference_time": self.total_inference_time / max(self.inference_count, 1)
        }
    
    def reset_model(self) -> None:
        """Reset model to initial state"""
        self.model = None
        self.preprocessor = None
        self.is_trained = False
        self.metrics = ModelMetrics()
        self.inference_count = 0
        self.total_inference_time = 0.0
        logger.info(f"Reset model: {self.model_name}")
    
    def __str__(self) -> str:
        """String representation of model"""
        return f"{self.model_name} (v{self.version}) - {self.model_type}"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"{self.__class__.__name__}(model_id='{self.model_id}', "
                f"model_name='{self.model_name}', model_type='{self.model_type}', "
                f"version='{self.version}', is_trained={self.is_trained})")
