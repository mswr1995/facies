"""
Hybrid CNN+XGBoost inference for hierarchical grain classification.

Uses CNN for Stages 1 & 2, optionally XGBoost for Stage 3.
"""
import torch
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, Optional, Tuple

from .hierarchical_model import HierarchicalGrainClassifier


class HybridGrainClassifier:
    """
    Hybrid classifier combining CNN and XGBoost.
    
    - Stage 1 (CNN): Peloid vs Non-peloid
    - Stage 2 (CNN): Ooid-like vs Intraclast
    - Stage 3 (XGBoost): Broken vs Whole Ooid
    
    Args:
        cnn_model: Trained hierarchical CNN model
        xgboost_model: Trained XGBoost model for Stage 3 (optional)
        device: Device to run CNN on
        thresholds: Decision thresholds (T1, T2, T3)
    """
    
    def __init__(
        self,
        cnn_model: HierarchicalGrainClassifier,
        xgboost_model: Optional[xgb.Booster] = None,
        device: str = 'cpu',
        thresholds: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ):
        self.cnn_model = cnn_model.to(device)
        self.cnn_model.eval()
        self.xgboost_model = xgboost_model
        self.device = device
        self.t1, self.t2, self.t3 = thresholds
        
        # Freeze backbone for embedding extraction
        self.cnn_model.freeze_backbone()
    
    @classmethod
    def from_checkpoints(
        cls,
        cnn_checkpoint_path: str,
        xgboost_model_path: Optional[str] = None,
        device: str = 'cpu',
        thresholds: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ):
        """
        Load hybrid model from checkpoint files.
        
        Args:
            cnn_checkpoint_path: Path to CNN checkpoint
            xgboost_model_path: Path to XGBoost model (optional)
            device: Device for CNN
            thresholds: Decision thresholds
            
        Returns:
            HybridGrainClassifier instance
        """
        # Load CNN
        checkpoint = torch.load(cnn_checkpoint_path, map_location=device)
        cnn_model = HierarchicalGrainClassifier(pretrained=False)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load XGBoost if provided
        xgboost_model = None
        if xgboost_model_path and Path(xgboost_model_path).exists():
            xgboost_model = xgb.Booster()
            xgboost_model.load_model(xgboost_model_path)
        
        return cls(cnn_model, xgboost_model, device, thresholds)
    
    def extract_cnn_embedding(self, image: torch.Tensor) -> np.ndarray:
        """
        Extract 512-D embedding from CNN backbone.
        
        Args:
            image: Input image tensor (1, 3, H, W) or (3, H, W)
            
        Returns:
            512-D embedding vector
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            features = self.cnn_model.backbone(image)
            features = torch.flatten(features, 1)
        
        return features.cpu().numpy()[0]
    
    def predict_stage1(self, image: torch.Tensor) -> Tuple[float, int]:
        """
        Stage 1: Peloid vs Non-peloid.
        
        Args:
            image: Input image tensor
            
        Returns:
            (probability, prediction) where prediction is 1=peloid, 0=non-peloid
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            logits = self.cnn_model(image)
            prob = torch.sigmoid(logits['stage1']).item()
        
        prediction = 1 if prob > self.t1 else 0
        return prob, prediction
    
    def predict_stage2(self, image: torch.Tensor) -> Tuple[float, int]:
        """
        Stage 2: Ooid-like vs Intraclast (for non-peloids).
        
        Args:
            image: Input image tensor
            
        Returns:
            (probability, prediction) where prediction is 1=ooid-like, 0=intraclast
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            logits = self.cnn_model(image)
            prob = torch.sigmoid(logits['stage2']).item()
        
        prediction = 1 if prob > self.t2 else 0
        return prob, prediction
    
    def predict_stage3_cnn(self, image: torch.Tensor) -> Tuple[float, int]:
        """
        Stage 3 (CNN version): Broken vs Whole Ooid.
        
        Args:
            image: Input image tensor
            
        Returns:
            (probability, prediction) where prediction is 1=whole, 0=broken
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            logits = self.cnn_model(image)
            prob = torch.sigmoid(logits['stage3']).item()
        
        prediction = 1 if prob > self.t3 else 0
        return prob, prediction
    
    def predict_stage3_xgboost(
        self,
        embedding: np.ndarray,
        geometric_features: Optional[np.ndarray] = None
    ) -> Tuple[float, int]:
        """
        Stage 3 (XGBoost version): Broken vs Whole Ooid.
        
        Args:
            embedding: 512-D CNN embedding
            geometric_features: Optional geometric features to append
            
        Returns:
            (probability_broken, prediction) where prediction is 1=whole, 0=broken
        """
        if self.xgboost_model is None:
            raise ValueError("XGBoost model not loaded")
        
        # Combine features
        if geometric_features is not None:
            features = np.concatenate([embedding, geometric_features])
        else:
            features = embedding
        
        # Predict
        dmatrix = xgb.DMatrix(features.reshape(1, -1))
        prob_broken = self.xgboost_model.predict(dmatrix)[0]
        
        # XGBoost predicts probability of broken (class 1)
        # If prob_broken >= threshold, classify as broken (0), else whole (1)
        prediction = 0 if prob_broken >= self.t3 else 1
        return prob_broken, prediction
    
    def predict(
        self,
        image: torch.Tensor,
        geometric_features: Optional[np.ndarray] = None,
        use_xgboost_stage3: bool = None
    ) -> Dict:
        """
        Full hierarchical prediction.
        
        Args:
            image: Input image tensor
            geometric_features: Optional geometric features for XGBoost
            use_xgboost_stage3: Whether to use XGBoost for Stage 3
                               (defaults to True if XGBoost model is available)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Determine Stage 3 method
        if use_xgboost_stage3 is None:
            use_xgboost_stage3 = self.xgboost_model is not None
        
        # Stage 1: Peloid vs Non-peloid
        prob_peloid, pred_stage1 = self.predict_stage1(image)
        
        result = {
            'stage1_prob': prob_peloid,
            'stage1_pred': pred_stage1,
            'final_class': None,
            'final_class_name': None
        }
        
        if pred_stage1 == 1:
            # Peloid
            result['final_class'] = 0
            result['final_class_name'] = 'Peloid'
            return result
        
        # Stage 2: Ooid-like vs Intraclast
        prob_ooid_like, pred_stage2 = self.predict_stage2(image)
        result['stage2_prob'] = prob_ooid_like
        result['stage2_pred'] = pred_stage2
        
        if pred_stage2 == 0:
            # Intraclast
            result['final_class'] = 3
            result['final_class_name'] = 'Intraclast'
            return result
        
        # Stage 3: Broken vs Whole Ooid
        if use_xgboost_stage3:
            # XGBoost version
            embedding = self.extract_cnn_embedding(image)
            prob_broken, pred_stage3 = self.predict_stage3_xgboost(
                embedding, geometric_features
            )
            result['stage3_method'] = 'xgboost'
            result['stage3_prob'] = prob_broken  # Store probability of broken
        else:
            # CNN version
            prob_whole, pred_stage3 = self.predict_stage3_cnn(image)
            result['stage3_method'] = 'cnn'
            result['stage3_prob'] = prob_whole  # Store probability of whole
        result['stage3_pred'] = pred_stage3
        
        if pred_stage3 == 1:
            # Whole Ooid
            result['final_class'] = 1
            result['final_class_name'] = 'Ooid'
        else:
            # Broken Ooid
            result['final_class'] = 2
            result['final_class_name'] = 'Broken ooid'
        
        return result
    
    def predict_batch(
        self,
        images: torch.Tensor,
        geometric_features_list: Optional[list] = None,
        use_xgboost_stage3: bool = None
    ) -> list:
        """
        Predict for a batch of images.
        
        Args:
            images: Batch of images (B, 3, H, W)
            geometric_features_list: List of geometric features per sample
            use_xgboost_stage3: Whether to use XGBoost for Stage 3
            
        Returns:
            List of prediction dictionaries
        """
        batch_size = images.size(0)
        results = []
        
        for i in range(batch_size):
            geom_feats = None
            if geometric_features_list is not None:
                geom_feats = geometric_features_list[i]
            
            result = self.predict(
                images[i],
                geometric_features=geom_feats,
                use_xgboost_stage3=use_xgboost_stage3
            )
            results.append(result)
        
        return results
