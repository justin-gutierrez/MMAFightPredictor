#!/usr/bin/env python3
"""
Skill-based model training script for UFC ML predictor.
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from ufcml.train import train_xgb, save_model, cross_validate_model
from ufcml.config import get_config


def load_splits_from_json(splits_dir: str, n_folds: int = 5) -> list:
    """
    Load time-based splits from JSON files.
    
    Args:
        splits_dir: Directory containing split files
        n_folds: Number of folds to load
        
    Returns:
        List of (train_idx, valid_idx) tuples
    """
    splits_dir = Path(splits_dir)
    splits = []
    
    for fold in range(1, n_folds + 1):
        fold_file = splits_dir / f"time_based_cv_fold_{fold}.json"
        
        if fold_file.exists():
            with open(fold_file, 'r') as f:
                fold_data = json.load(f)
            
            train_idx = np.array(fold_data['train_indices'])
            valid_idx = np.array(fold_data['valid_indices'])
            splits.append((train_idx, valid_idx))
        else:
            print(f"Warning: Split file not found: {fold_file}")
    
    return splits


def main():
    """Main function for skill-based model training."""
    parser = argparse.ArgumentParser(
        description="Train skill-based XGBoost models using time-based cross-validation"
    )
    parser.add_argument(
        "--X", 
        type=str, 
        default="data/processed/X.parquet",
        help="Path to feature matrix file (default: data/processed/X.parquet)"
    )
    parser.add_argument(
        "--y", 
        type=str, 
        default="data/processed/y.parquet",
        help="Path to target variable file (default: data/processed/y.parquet)"
    )
    parser.add_argument(
        "--meta", 
        type=str, 
        default="data/processed/meta.parquet",
        help="Path to meta data file (default: data/processed/meta.parquet)"
    )
    parser.add_argument(
        "--splits", 
        type=str, 
        default="data/processed/splits",
        help="Directory containing time-based splits (default: data/processed/splits)"
    )
    parser.add_argument(
        "--models-dir", 
        type=str, 
        default="data/models",
        help="Directory to save trained models (default: data/models)"
    )
    parser.add_argument(
        "--reports-dir", 
        type=str, 
        default="data/reports",
        help="Directory to save training reports (default: data/reports)"
    )
    parser.add_argument(
        "--n-folds", 
        type=int, 
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--n-jobs", 
        type=int, 
        default=-1,
        help="Number of parallel jobs (-1 for all cores, default: -1)"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Set paths
    X_path = Path(args.X)
    y_path = Path(args.y)
    meta_path = Path(args.meta)
    splits_dir = Path(args.splits)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    
    # Validate input files exist
    for path, name in [(X_path, "Feature matrix"), (y_path, "Target variable"), 
                       (meta_path, "Meta data"), (splits_dir, "Splits directory")]:
        if not path.exists():
            print(f"Error: {name} not found: {path}")
            sys.exit(1)
    
    print(f"Starting skill-based model training...")
    print(f"Feature matrix: {X_path}")
    print(f"Target variable: {y_path}")
    print(f"Meta data: {meta_path}")
    print(f"Splits directory: {splits_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Reports directory: {reports_dir}")
    print(f"Number of folds: {args.n_folds}")
    print(f"Random seed: {args.seed}")
    print(f"Parallel jobs: {args.n_jobs}")
    
    try:
        # Load data
        print("\nLoading data...")
        
        # Try to load as parquet first, then CSV
        try:
            X = pd.read_parquet(X_path)
            print(f"Loaded feature matrix: {X.shape}")
        except:
            X = pd.read_csv(X_path)
            print(f"Loaded feature matrix: {X.shape}")
        
        try:
            y = pd.read_parquet(y_path)
            print(f"Loaded target variable: {y.shape}")
        except:
            y = pd.read_csv(y_path)
            print(f"Loaded target variable: {y.shape}")
        
        try:
            meta = pd.read_parquet(meta_path)
            print(f"Loaded meta data: {meta.shape}")
        except:
            meta = pd.read_csv(meta_path)
            print(f"Loaded meta data: {meta.shape}")
        
        # If y is a DataFrame, extract the series
        if isinstance(y, pd.DataFrame):
            if "Winner" in y.columns:
                y = y["Winner"]
            else:
                y = y.iloc[:, 0]  # Take first column
        
        print(f"Data loaded successfully")
        
        # Load time-based splits
        print(f"\nLoading time-based splits...")
        splits = load_splits_from_json(splits_dir, args.n_folds)
        
        if len(splits) == 0:
            print("Error: No valid splits found")
            sys.exit(1)
        
        print(f"Loaded {len(splits)} splits")
        for i, (train_idx, valid_idx) in enumerate(splits):
            print(f"  Split {i+1}: Train {len(train_idx)}, Valid {len(valid_idx)}")
        
        # Train models on each fold
        print(f"\nTraining XGBoost models on each fold...")
        fold_results = []
        
        for fold, (train_idx, valid_idx) in enumerate(splits):
            print(f"\n{'='*50}")
            print(f"Training Fold {fold + 1}/{len(splits)}")
            print(f"{'='*50}")
            
            try:
                # Train model
                model, metrics = train_xgb(
                    X, y, train_idx, valid_idx,
                    seed=args.seed,
                    n_jobs=args.n_jobs,
                    with_odds=False  # Skill-based model without odds
                )
                
                # Save model
                model_name = f"skill_fold_{fold + 1}"
                save_model(model, metrics, models_dir, model_name)
                
                # Store results
                fold_results.append({
                    'fold': fold + 1,
                    'metrics': metrics,
                    'model_name': model_name
                })
                
                print(f"Fold {fold + 1} completed successfully")
                
            except Exception as e:
                print(f"Error in fold {fold + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not fold_results:
            print("Error: No models were trained successfully")
            sys.exit(1)
        
        # Calculate average metrics across folds
        print(f"\nCalculating average metrics across folds...")
        
        # Get metric names from first fold
        metric_names = [k for k in fold_results[0]['metrics'].keys() 
                       if isinstance(fold_results[0]['metrics'][k], (int, float))]
        
        avg_metrics = {}
        for metric in metric_names:
            values = [fold['metrics'][metric] for fold in fold_results]
            avg_metrics[f'avg_{metric}'] = np.mean(values)
            avg_metrics[f'std_{metric}'] = np.std(values)
            avg_metrics[f'min_{metric}'] = np.min(values)
            avg_metrics[f'max_{metric}'] = np.max(values)
        
        # Create comprehensive results
        cv_results = {
            'pipeline_info': {
                'model_type': 'XGBoost',
                'training_strategy': 'skill_based_no_odds',
                'n_folds': len(fold_results),
                'random_seed': args.seed,
                'n_jobs': args.n_jobs,
                'total_samples': len(X),
                'n_features': X.shape[1]
            },
            'fold_results': fold_results,
            'average_metrics': avg_metrics
        }
        
        # Save comprehensive results
        print(f"\nSaving results...")
        
        # Save CV metrics
        reports_dir.mkdir(parents=True, exist_ok=True)
        cv_metrics_path = reports_dir / "skill_cv_metrics.json"
        
        with open(cv_metrics_path, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)
        
        print(f"Cross-validation results saved to: {cv_metrics_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SKILL-BASED MODEL TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Models trained: {len(fold_results)}/{len(splits)}")
        print(f"Models saved to: {models_dir}")
        print(f"Results saved to: {cv_metrics_path}")
        
        print(f"\nAverage Metrics Across Folds:")
        for metric in ['roc_auc', 'logloss', 'brier', 'accuracy']:
            if f'avg_{metric}' in avg_metrics:
                avg_val = avg_metrics[f'avg_{metric}']
                std_val = avg_metrics[f'std_{metric}']
                print(f"  {metric.upper()}: {avg_val:.4f} ± {std_val:.4f}")
        
        print(f"\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
