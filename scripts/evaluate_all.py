#!/usr/bin/env python3
"""
Comprehensive evaluation script for UFC ML predictor models.
Evaluates all folds, applies calibration, and generates reliability plots.
"""

import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ufcml.config import get_config
from ufcml.calibrate import fit_calibrator, apply_calibrator
from ufcml.evaluate import compute_metrics, plot_reliability, create_evaluation_report
from ufcml.split import load_splits


def load_model_and_data(fold: int, config: any) -> Tuple[any, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load trained model and corresponding data for a specific fold.
    
    Args:
        fold: Fold number (1-based)
        config: Configuration object
        
    Returns:
        Tuple of (model, X, y, meta)
    """
    # Load model
    model_path = config.MODELS_DIR / f"skill_fold_{fold}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")
    
    # Load data
    X = pd.read_parquet(config.PROCESSED_DIR / "X.parquet")
    y = pd.read_parquet(config.PROCESSED_DIR / "y.parquet")
    meta = pd.read_parquet(config.PROCESSED_DIR / "meta.parquet")
    
    # If y is DataFrame, extract series
    if isinstance(y, pd.DataFrame):
        if "Winner" in y.columns:
            y = y["Winner"]
        else:
            y = y.iloc[:, 0]
    
    print(f"Loaded data: X={X.shape}, y={y.shape}, meta={meta.shape}")
    
    return model, X, y, meta


def evaluate_fold(fold: int, 
                 model: any, 
                 X: pd.DataFrame, 
                 y: pd.Series, 
                 meta: pd.DataFrame,
                 splits: List[Tuple[np.ndarray, np.ndarray]],
                 config: any,
                 calibration_method: str = "isotonic") -> Dict:
    """
    Evaluate a single fold with calibration.
    
    Args:
        fold: Fold number (1-based)
        model: Trained XGBoost model
        X: Feature matrix
        y: Target variable
        meta: Metadata DataFrame
        splits: List of (train_idx, valid_idx) tuples
        config: Configuration object
        calibration_method: Calibration method to use
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*50}")
    print(f"Evaluating Fold {fold}")
    print(f"{'='*50}")
    
    # Get validation indices for this fold
    train_idx, valid_idx = splits[fold - 1]  # 0-based indexing
    
    # Get validation data
    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]
    meta_valid = meta.iloc[valid_idx]
    
    print(f"Validation set: {len(valid_idx)} samples")
    
    # Drop odds columns to match what the model was trained on
    odds_columns = [col for col in X_valid.columns if 'Odds' in col]
    if odds_columns:
        X_valid = X_valid.drop(columns=odds_columns)
        print(f"Dropped odds columns: {odds_columns}")
    
    # Get raw predictions
    print("Getting raw predictions...")
    y_pred_proba_raw = model.predict_proba(X_valid)[:, 1]
    
    # Compute raw metrics
    raw_metrics = compute_metrics(y_valid.values, y_pred_proba_raw)
    print(f"Raw metrics:")
    print(f"  ROC AUC: {raw_metrics['roc_auc']:.4f}")
    print(f"  Log Loss: {raw_metrics['logloss']:.4f}")
    print(f"  Brier Score: {raw_metrics['brier']:.4f}")
    
    # Fit calibrator
    print(f"Fitting {calibration_method} calibrator...")
    calibrator = fit_calibrator(y_pred_proba_raw, y_valid.values, method=calibration_method)
    
    # Apply calibration
    print("Applying calibration...")
    y_pred_proba_calibrated = apply_calibrator(y_pred_proba_raw, calibrator)
    
    # Compute calibrated metrics
    calibrated_metrics = compute_metrics(y_valid.values, y_pred_proba_calibrated)
    print(f"Calibrated metrics:")
    print(f"  ROC AUC: {calibrated_metrics['roc_auc']:.4f}")
    print(f"  Log Loss: {calibrated_metrics['logloss']:.4f}")
    print(f"  Brier Score: {calibrated_metrics['brier']:.4f}")
    
    # Create evaluation report
    fold_output_dir = config.REPORTS_DIR / f"fold_{fold}_evaluation"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating evaluation report...")
    report = create_evaluation_report(
        y_valid.values, 
        y_pred_proba_calibrated, 
        meta_valid, 
        str(fold_output_dir)
    )
    
    # Save calibrator
    calibrator_path = fold_output_dir / f"calibrator_{calibration_method}.joblib"
    joblib.dump(calibrator, calibrator_path)
    print(f"Calibrator saved to: {calibrator_path}")
    
    # Create reliability plot for raw vs calibrated
    comparison_plot_path = fold_output_dir / "raw_vs_calibrated_reliability.png"
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    
    # Plot raw predictions
    plot_reliability(y_valid.values, y_pred_proba_raw, 
                    str(fold_output_dir / "raw_reliability.png"), 
                    f"Fold {fold} - Raw Predictions")
    
    # Plot calibrated predictions
    plot_reliability(y_valid.values, y_pred_proba_calibrated, 
                    str(fold_output_dir / "calibrated_reliability.png"), 
                    f"Fold {fold} - Calibrated Predictions")
    
    # Create comparison plot
    from ufcml.evaluate import compare_models_side_by_side
    compare_models_side_by_side(
        y_valid.values,
        {
            'Raw': y_pred_proba_raw,
            'Calibrated': y_pred_proba_calibrated
        },
        str(comparison_plot_path),
        f"Fold {fold} - Raw vs Calibrated Predictions"
    )
    
    # Return results
    results = {
        'fold': fold,
        'raw_metrics': raw_metrics,
        'calibrated_metrics': calibrated_metrics,
        'improvement': {
            'logloss': raw_metrics['logloss'] - calibrated_metrics['logloss'],
            'brier': raw_metrics['brier'] - calibrated_metrics['brier'],
            'roc_auc': calibrated_metrics['roc_auc'] - raw_metrics['roc_auc']
        },
        'output_dir': str(fold_output_dir)
    }
    
    return results


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate all trained models with calibration")
    parser.add_argument(
        "--n-folds", 
        type=int, 
        default=3,
        help="Number of folds to evaluate (default: 3)"
    )
    parser.add_argument(
        "--calibration-method",
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="Calibration method to use (default: isotonic)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory (default: data/reports)"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.REPORTS_DIR / "comprehensive_evaluation"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION PIPELINE")
    print("="*60)
    print(f"Number of folds: {args.n_folds}")
    print(f"Calibration method: {args.calibration_method}")
    print(f"Output directory: {output_dir}")
    print(f"Reports directory: {config.REPORTS_DIR}")
    
    # Load splits
    print("\nLoading time-based splits...")
    try:
        splits = load_splits(config.PROCESSED_DIR / "splits")
        print(f"Loaded {len(splits)} splits")
    except Exception as e:
        print(f"Error loading splits: {e}")
        print("Please run make_splits.py first to generate splits")
        sys.exit(1)
    
    if len(splits) < args.n_folds:
        print(f"Warning: Only {len(splits)} splits available, but {args.n_folds} requested")
        args.n_folds = len(splits)
    
    # Evaluate each fold
    all_results = []
    
    for fold in range(1, args.n_folds + 1):
        try:
            # Load model and data
            model, X, y, meta = load_model_and_data(fold, config)
            
            # Evaluate fold
            results = evaluate_fold(fold, model, X, y, meta, splits, config, args.calibration_method)
            all_results.append(results)
            
            print(f"Fold {fold} evaluation completed successfully")
            
        except Exception as e:
            print(f"Error evaluating fold {fold}: {e}")
            continue
    
    if not all_results:
        print("No folds were evaluated successfully")
        sys.exit(1)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATED EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Calculate averages
    raw_metrics_avg = {}
    calibrated_metrics_avg = {}
    improvements_avg = {}
    
    metric_names = ['roc_auc', 'logloss', 'brier', 'accuracy']
    
    for metric in metric_names:
        raw_values = [r['raw_metrics'].get(metric, np.nan) for r in all_results if metric in r['raw_metrics']]
        calibrated_values = [r['calibrated_metrics'].get(metric, np.nan) for r in all_results if metric in r['calibrated_metrics']]
        
        if raw_values and not all(np.isnan(raw_values)):
            raw_metrics_avg[f'avg_{metric}'] = np.mean(raw_values)
            raw_metrics_avg[f'std_{metric}'] = np.std(raw_values)
        
        if calibrated_values and not all(np.isnan(calibrated_values)):
            calibrated_metrics_avg[f'avg_{metric}'] = np.mean(calibrated_values)
            calibrated_metrics_avg[f'std_{metric}'] = np.std(calibrated_values)
    
    # Calculate average improvements
    improvement_metrics = ['logloss', 'brier', 'roc_auc']
    for metric in improvement_metrics:
        values = [r['improvement'].get(metric, 0) for r in all_results]
        if values:
            improvements_avg[f'avg_{metric}_improvement'] = np.mean(values)
            improvements_avg[f'std_{metric}_improvement'] = np.std(values)
    
    # Print results
    print(f"\nRAW PREDICTIONS (Average ± Std):")
    for metric in metric_names:
        avg_key = f'avg_{metric}'
        std_key = f'std_{metric}'
        if avg_key in raw_metrics_avg:
            print(f"  {metric.upper()}: {raw_metrics_avg[avg_key]:.4f} ± {raw_metrics_avg[std_key]:.4f}")
    
    print(f"\nCALIBRATED PREDICTIONS (Average ± Std):")
    for metric in metric_names:
        avg_key = f'avg_{metric}'
        std_key = f'std_{metric}'
        if avg_key in calibrated_metrics_avg:
            print(f"  {metric.upper()}: {calibrated_metrics_avg[avg_key]:.4f} ± {calibrated_metrics_avg[std_key]:.4f}")
    
    print(f"\nCALIBRATION IMPROVEMENTS (Average ± Std):")
    for metric in improvement_metrics:
        avg_key = f'avg_{metric}_improvement'
        std_key = f'std_{metric}_improvement'
        if avg_key in improvements_avg:
            print(f"  {metric.upper()}: {improvements_avg[avg_key]:.4f} ± {improvements_avg[std_key]:.4f}")
    
    # Save comprehensive report
    comprehensive_report = {
        'pipeline_info': {
            'n_folds': args.n_folds,
            'calibration_method': args.calibration_method,
            'output_directory': str(output_dir)
        },
        'fold_results': all_results,
        'aggregated_metrics': {
            'raw': raw_metrics_avg,
            'calibrated': calibrated_metrics_avg,
            'improvements': improvements_avg
        }
    }
    
    report_path = output_dir / "comprehensive_evaluation_report.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(comprehensive_report, f, indent=2, default=str)
    
    print(f"\nComprehensive evaluation report saved to: {report_path}")
    print(f"\nIndividual fold reports saved to:")
    for result in all_results:
        print(f"  Fold {result['fold']}: {result['output_dir']}")
    
    print(f"\n{'='*60}")
    print("EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
