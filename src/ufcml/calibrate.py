"""
Probability calibration for UFC ML predictor.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelBinarizer


def fit_calibrator(probs_valid: np.ndarray, 
                   y_valid: np.ndarray, 
                   method: str = "isotonic") -> Union[IsotonicRegression, CalibratedClassifierCV]:
    """
    Fit a probability calibrator using validation set predictions.
    
    Args:
        probs_valid: Raw probabilities from model (shape: n_samples,)
        y_valid: True labels (shape: n_samples,)
        method: Calibration method ("isotonic" or "sigmoid")
        
    Returns:
        Fitted calibrator object
        
    Example:
        >>> calibrator = fit_calibrator(y_pred_proba, y_valid, method="isotonic")
        >>> calibrated_probs = apply_calibrator(y_pred_proba, calibrator)
    """
    if method == "isotonic":
        # Isotonic regression for non-parametric calibration
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(probs_valid, y_valid)
        return calibrator
        
    elif method == "sigmoid":
        # Platt scaling for parametric calibration
        calibrator = CalibratedClassifierCV(
            cv="prefit", 
            method="sigmoid"
        )
        # Create dummy classifier for sigmoid calibration
        from sklearn.base import BaseEstimator, ClassifierMixin
        class DummyClassifier(BaseEstimator, ClassifierMixin):
            def __init__(self):
                self.classes_ = np.array([0, 1])
            def predict_proba(self, X):
                return np.column_stack([1-X, X])
        
        dummy = DummyClassifier()
        calibrator.fit(dummy, probs_valid, y_valid)
        return calibrator
        
    else:
        raise ValueError(f"Unknown calibration method: {method}. Use 'isotonic' or 'sigmoid'")


def apply_calibrator(probs: np.ndarray, 
                     calibrator: Union[IsotonicRegression, CalibratedClassifierCV]) -> np.ndarray:
    """
    Apply fitted calibrator to new probabilities.
    
    Args:
        probs: Raw probabilities to calibrate (shape: n_samples,)
        calibrator: Fitted calibrator from fit_calibrator()
        
    Returns:
        Calibrated probabilities (shape: n_samples,)
        
    Example:
        >>> calibrated_probs = apply_calibrator(y_pred_proba, calibrator)
    """
    if isinstance(calibrator, IsotonicRegression):
        # Isotonic regression calibration
        calibrated = calibrator.predict(probs)
        # Ensure probabilities are in [0, 1]
        calibrated = np.clip(calibrated, 0, 1)
        return calibrated
        
    elif hasattr(calibrator, 'predict_proba'):
        # CalibratedClassifierCV calibration
        # Reshape for 2D input if needed
        if probs.ndim == 1:
            probs_2d = np.column_stack([1-probs, probs])
        else:
            probs_2d = probs
            
        calibrated_2d = calibrator.predict_proba(probs_2d)
        # Return probability of positive class
        return calibrated_2d[:, 1]
        
    else:
        raise ValueError("Invalid calibrator object")


def fit_ensemble_calibrator(probs_list: list, 
                           y_valid: np.ndarray, 
                           method: str = "isotonic") -> list:
    """
    Fit calibrators for multiple models/folds.
    
    Args:
        probs_list: List of probability arrays from different models
        y_valid: True labels for validation set
        method: Calibration method
        
    Returns:
        List of fitted calibrators
    """
    calibrators = []
    for probs in probs_list:
        calibrator = fit_calibrator(probs, y_valid, method)
        calibrators.append(calibrator)
    return calibrators


def evaluate_calibration(y_true: np.ndarray, 
                        probs: np.ndarray, 
                        n_bins: int = 10) -> dict:
    """
    Evaluate calibration quality using reliability metrics.
    
    Args:
        y_true: True labels
        probs: Predicted probabilities
        n_bins: Number of bins for reliability analysis
        
    Returns:
        Dictionary with calibration metrics
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate reliability metrics
    reliability_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_probs = probs[mask]
            bin_true = y_true[mask]
            bin_mean_prob = bin_probs.mean()
            bin_true_rate = bin_true.mean()
            bin_count = mask.sum()
            
            reliability_data.append({
                'bin': i,
                'mean_prob': bin_mean_prob,
                'true_rate': bin_true_rate,
                'count': bin_count,
                'calibration_error': abs(bin_mean_prob - bin_true_rate)
            })
    
    reliability_df = pd.DataFrame(reliability_data)
    
    # Calculate overall calibration metrics
    if len(reliability_df) > 0:
        ece = (reliability_df['calibration_error'] * reliability_df['count']).sum() / len(y_true)
        mce = reliability_df['calibration_error'].max()
    else:
        ece = mce = np.nan
    
    return {
        'ece': ece,  # Expected Calibration Error
        'mce': mce,  # Maximum Calibration Error
        'reliability_data': reliability_df
    }
