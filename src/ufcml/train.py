"""
Model training pipeline for UFC ML predictor.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import xgboost as xgb
import joblib
from pathlib import Path


def train_xgb(X: pd.DataFrame, 
              y: pd.Series, 
              train_idx: np.ndarray, 
              valid_idx: np.ndarray, 
              seed: int = 42, 
              n_jobs: int = -1, 
              with_odds: bool = False) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """
    Train XGBoost classifier for UFC fight prediction.
    
    Args:
        X: Feature matrix
        y: Target variable (1 for Red win, 0 for Blue win)
        train_idx: Training indices
        valid_idx: Validation indices
        seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
        with_odds: Whether to include betting odds features
        
    Returns:
        Tuple of (fitted_model, evaluation_metrics)
        
    Example:
        >>> model, metrics = train_xgb(X, y, train_idx, valid_idx)
        >>> print(f"Validation ROC AUC: {metrics['roc_auc']:.3f}")
    """
    # Create copy of feature matrix
    X_train = X.copy()
    
    # Drop odds features if not requested
    if not with_odds:
        odds_columns = [col for col in X_train.columns if 'Odds' in col]
        if odds_columns:
            X_train = X_train.drop(columns=odds_columns)
            print(f"Dropped odds columns: {odds_columns}")
    
    # Split data
    X_train_split = X_train.iloc[train_idx]
    y_train_split = y.iloc[train_idx]
    X_valid_split = X_train.iloc[valid_idx]
    y_valid_split = y.iloc[valid_idx]
    
    print(f"Training set: {X_train_split.shape[0]} samples, {X_train_split.shape[1]} features")
    print(f"Validation set: {X_valid_split.shape[0]} samples")
    
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        eval_metric="logloss",
        n_jobs=n_jobs,
        random_state=seed,
        early_stopping_rounds=100,
        verbose=0
    )
    
    # Train model with early stopping
    print("Training XGBoost model...")
    model.fit(
        X_train_split, 
        y_train_split,
        eval_set=[(X_valid_split, y_valid_split)],
        verbose=False
    )
    
    # Get best iteration
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
    print(f"Best iteration: {best_iteration}")
    
    # Make predictions on validation set
    y_pred_proba = model.predict_proba(X_valid_split)[:, 1]
    y_pred = model.predict(X_valid_split)
    
    # Calculate evaluation metrics
    metrics = calculate_eval_metrics(y_valid_split, y_pred, y_pred_proba)
    
    # Add training info to metrics
    metrics.update({
        "best_iteration": best_iteration,
        "n_features": X_train_split.shape[1],
        "feature_names": X_train_split.columns.tolist(),
        "n_train_samples": len(train_idx),
        "n_valid_samples": len(valid_idx)
    })
    
    print(f"Validation metrics:")
    print(f"  Log Loss: {metrics['logloss']:.4f}")
    print(f"  Brier Score: {metrics['brier']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    return model, metrics


def calculate_eval_metrics(y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['logloss'] = log_loss(y_true, y_pred_proba)
    metrics['brier'] = brier_score_loss(y_true, y_pred_proba)
    
    # ROC AUC (handle edge cases)
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        # Handle case where only one class is present
        metrics['roc_auc'] = 0.5
    
    # Additional metrics
    metrics['accuracy'] = (y_true == y_pred).mean()
    metrics['red_win_rate'] = y_true.mean()
    metrics['pred_red_win_rate'] = y_pred.mean()
    
    return metrics


def save_model(model: xgb.XGBClassifier, 
               metrics: Dict[str, Any], 
               output_path: str, 
               model_name: str = "model") -> None:
    """
    Save trained model and metrics.
    
    Args:
        model: Trained XGBoost model
        metrics: Evaluation metrics
        output_path: Directory to save model
        model_name: Name for the model file
        
    Example:
        >>> save_model(model, metrics, "data/models", "skill_fold_1")
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = output_path / f"{model_name}.joblib"
    joblib.dump(model, model_file)
    
    # Save metrics
    metrics_file = output_path / f"{model_name}_metrics.json"
    import json
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"Model saved to: {model_file}")
    print(f"Metrics saved to: {metrics_file}")


def load_model(model_path: str) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Load trained model and metrics.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (model, metrics)
        
    Example:
        >>> model, metrics = load_model("data/models/skill_fold_1.joblib")
    """
    model_path = Path(model_path)
    
    # Load model
    model = joblib.load(model_path)
    
    # Try to load metrics
    metrics_path = model_path.parent / f"{model_path.stem}_metrics.json"
    metrics = {}
    if metrics_path.exists():
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    return model, metrics


def get_feature_importance(model: xgb.XGBClassifier, 
                          feature_names: Optional[list] = None, 
                          top_n: int = 20) -> pd.DataFrame:
    """
    Extract feature importance from trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance scores
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
    
    # Get feature importance
    importance_scores = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    
    # Sort by importance and get top features
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    return importance_df


def cross_validate_model(X: pd.DataFrame, 
                        y: pd.Series, 
                        splits: list, 
                        model_func: callable, 
                        **kwargs) -> Dict[str, Any]:
    """
    Perform cross-validation using provided splits.
    
    Args:
        X: Feature matrix
        y: Target variable
        splits: List of (train_idx, valid_idx) tuples
        model_func: Function to train model (e.g., train_xgb)
        **kwargs: Additional arguments for model_func
        
    Returns:
        Dictionary with cross-validation results
    """
    cv_results = {
        'fold_metrics': [],
        'models': [],
        'feature_importance': []
    }
    
    for i, (train_idx, valid_idx) in enumerate(splits):
        print(f"\n--- Fold {i+1}/{len(splits)} ---")
        
        try:
            # Train model
            model, metrics = model_func(X, y, train_idx, valid_idx, **kwargs)
            
            # Store results
            cv_results['fold_metrics'].append(metrics)
            cv_results['models'].append(model)
            
            # Get feature importance
            feature_importance = get_feature_importance(model, X.columns)
            cv_results['feature_importance'].append(feature_importance)
            
            print(f"Fold {i+1} completed successfully")
            
        except Exception as e:
            print(f"Error in fold {i+1}: {e}")
            continue
    
    # Calculate average metrics
    if cv_results['fold_metrics']:
        avg_metrics = {}
        metric_names = cv_results['fold_metrics'][0].keys()
        
        for metric in metric_names:
            if metric not in ['feature_names', 'n_features']:  # Skip non-numeric metrics
                values = [fold[metric] for fold in cv_results['fold_metrics'] if isinstance(fold[metric], (int, float))]
                if values:
                    avg_metrics[f'avg_{metric}'] = np.mean(values)
                    avg_metrics[f'std_{metric}'] = np.std(values)
        
        cv_results['average_metrics'] = avg_metrics
    
    return cv_results
