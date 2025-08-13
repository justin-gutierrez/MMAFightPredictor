#!/usr/bin/env python3
"""
Data splitting pipeline script for UFC ML predictor.
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
from ufcml.split import (
    time_based_split, 
    final_holdout_split, 
    get_split_summary, 
    validate_splits, 
    save_splits
)
from ufcml.io import read_csv
from ufcml.config import get_config


def main():
    """Main function for the data splitting pipeline."""
    parser = argparse.ArgumentParser(
        description="Create time-based data splits for UFC fight prediction"
    )
    parser.add_argument(
        "--meta", 
        type=str, 
        default="data/processed/meta.parquet",
        help="Path to meta data file (default: data/processed/meta.parquet)"
    )
    parser.add_argument(
        "--target", 
        type=str, 
        default="data/processed/y.parquet",
        help="Path to target variable file (default: data/processed/y.parquet)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed/splits",
        help="Output directory for splits (default: data/processed/splits)"
    )
    parser.add_argument(
        "--n-folds", 
        type=int, 
        default=3,
        help="Number of cross-validation folds (default: 3)"
    )
    parser.add_argument(
        "--group-by",
        choices=["month", "date"],
        default="month",
        help="Grouping strategy for time-based splits (default: month)"
    )
    parser.add_argument(
        "--min-valid-size",
        type=int,
        default=150,
        help="Minimum validation set size for month-based splits (default: 150)"
    )
    parser.add_argument(
        "--cutoff-date", 
        type=str, 
        default="2024-01-01",
        help="Cutoff date for final holdout split (default: 2024-01-01)"
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate splits for data leakage"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Set paths
    meta_path = Path(args.meta)
    target_path = Path(args.target)
    output_path = Path(args.output)
    
    # Validate input files exist
    if not meta_path.exists():
        print(f"Error: Meta file not found: {meta_path}")
        sys.exit(1)
    
    if not target_path.exists():
        print(f"Error: Target file not found: {target_path}")
        sys.exit(1)
    
    print(f"Starting data splitting pipeline...")
    print(f"Meta file: {meta_path}")
    print(f"Target file: {target_path}")
    print(f"Output directory: {output_path}")
    print(f"Number of folds: {args.n_folds}")
    print(f"Group by: {args.group_by}")
    print(f"Min valid size: {args.min_valid_size}")
    print(f"Cutoff date: {args.cutoff_date}")
    
    try:
        # Load meta data and target variable
        print("Loading data...")
        
        # Try to load as parquet first, then CSV
        try:
            meta_df = pd.read_parquet(meta_path)
            print(f"Loaded meta data: {meta_df.shape}")
        except:
            meta_df = pd.read_csv(meta_path)
            print(f"Loaded meta data: {meta_df.shape}")
        
        try:
            y = pd.read_parquet(target_path)
            print(f"Loaded target variable: {y.shape}")
        except:
            y = pd.read_csv(target_path)
            print(f"Loaded target variable: {y.shape}")
        
        # If y is a DataFrame, extract the series
        if isinstance(y, pd.DataFrame):
            if "Winner" in y.columns:
                y = y["Winner"]
            else:
                y = y.iloc[:, 0]  # Take first column
        
        print(f"Data loaded successfully")
        print(f"Meta columns: {list(meta_df.columns)}")
        print(f"Date range: {meta_df['Date'].min()} to {meta_df['Date'].max()}")
        
        # Create time-based cross-validation splits
        print(f"\nCreating {args.n_folds}-fold time-based splits (grouped by {args.group_by})...")
        cv_splits = time_based_split(
            meta_df, y, 
            n_folds=args.n_folds,
            group_by=args.group_by,
            min_valid_size=args.min_valid_size
        )
        
        print(f"Created {len(cv_splits)} CV folds")
        for i, (train_idx, valid_idx) in enumerate(cv_splits):
            print(f"  Fold {i+1}: Train {len(train_idx)}, Valid {len(valid_idx)}")
        
        # Create final holdout split
        print(f"\nCreating final holdout split (cutoff: {args.cutoff_date})...")
        train_idx, test_idx = final_holdout_split(meta_df, args.cutoff_date)
        
        print(f"Final holdout: Train {len(train_idx)}, Test {len(test_idx)}")
        
        # Generate split summaries
        print("\nGenerating split summaries...")
        cv_summary = get_split_summary(cv_splits, meta_df, y)
        
        holdout_summary = {
            "split_type": "final_holdout",
            "cutoff_date": args.cutoff_date,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "train_red_wins": int(y.iloc[train_idx].sum()) if len(train_idx) > 0 else 0,
            "test_red_wins": int(y.iloc[test_idx].sum()) if len(test_idx) > 0 else 0,
            "train_red_win_rate": float(y.iloc[train_idx].mean()) if len(train_idx) > 0 else 0.0,
            "test_red_win_rate": float(y.iloc[test_idx].mean()) if len(test_idx) > 0 else 0.0,
            "train_date_range": {
                "start": pd.to_datetime(meta_df.iloc[train_idx]["Date"]).min().strftime("%Y-%m-%d"),
                "end": pd.to_datetime(meta_df.iloc[train_idx]["Date"]).max().strftime("%Y-%m-%d")
            } if len(train_idx) > 0 else {},
            "test_date_range": {
                "start": pd.to_datetime(meta_df.iloc[test_idx]["Date"]).min().strftime("%Y-%m-%d"),
                "end": pd.to_datetime(meta_df.iloc[test_idx]["Date"]).max().strftime("%Y-%m-%d")
            } if len(test_idx) > 0 else {}
        }
        
        # Validate splits if requested
        if args.validate:
            print("\nValidating splits for data leakage...")
            validation_results = validate_splits(cv_splits, meta_df)
            
            if validation_results["is_valid"]:
                print("✅ All splits are valid - no data leakage detected")
            else:
                print("❌ Data leakage detected:")
                for error in validation_results["errors"]:
                    print(f"  - {error}")
            
            if validation_results["warnings"]:
                print("⚠️  Warnings:")
                for warning in validation_results["warnings"]:
                    print(f"  - {warning}")
        
        # Save all splits
        print("\nSaving splits...")
        
        # Save CV splits
        save_splits(cv_splits, output_path, "time_based_cv")
        
        # Save final holdout split
        holdout_data = {
            "split_type": "final_holdout",
            "cutoff_date": args.cutoff_date,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "n_train": len(train_idx),
            "n_test": len(test_idx)
        }
        
        holdout_path = output_path / "final_holdout.json"
        with open(holdout_path, 'w') as f:
            json.dump(holdout_data, f, indent=2)
        
        # Save comprehensive summary
        comprehensive_summary = {
            "pipeline_info": {
                "n_folds": args.n_folds,
                "cutoff_date": args.cutoff_date,
                "total_samples": len(meta_df),
                "date_range": {
                    "start": pd.to_datetime(meta_df["Date"]).min().strftime("%Y-%m-%d"),
                    "end": pd.to_datetime(meta_df["Date"]).max().strftime("%Y-%m-%d")
                }
            },
            "cv_splits": cv_summary,
            "final_holdout": holdout_summary
        }
        
        summary_path = output_path / "splits_comprehensive_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(comprehensive_summary, f, indent=2, default=str)
        
        print(f"\nData splitting pipeline completed successfully!")
        print(f"CV splits saved to: {output_path}")
        print(f"Final holdout saved to: {holdout_path}")
        print(f"Comprehensive summary saved to: {summary_path}")
        
        # Print final statistics
        print(f"\nFinal Statistics:")
        print(f"  Total samples: {len(meta_df)}")
        print(f"  CV folds: {len(cv_splits)}")
        print(f"  Training samples (final): {len(train_idx)}")
        print(f"  Test samples (final): {len(test_idx)}")
        print(f"  Overall Red win rate: {y.mean():.3f}")
        
    except Exception as e:
        print(f"Error during data splitting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
