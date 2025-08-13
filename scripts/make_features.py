#!/usr/bin/env python3
"""
Feature engineering pipeline script for UFC ML predictor.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
from ufcml.features import assemble_feature_matrix, get_feature_summary, save_feature_matrix
from ufcml.elo import build_elo
from ufcml.io import read_csv
from ufcml.config import get_config


def main():
    """Main function for the feature engineering pipeline."""
    parser = argparse.ArgumentParser(
        description="Create feature matrix from cleaned UFC fight data"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/interim/prefight_clean.csv",
        help="Path to cleaned fight data CSV (default: data/interim/prefight_clean.csv)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed",
        help="Output directory for feature matrix (default: data/processed)"
    )
    parser.add_argument(
        "--no-elo", 
        action="store_true",
        help="Disable Elo ratings (default: Elo ratings are enabled)"
    )
    parser.add_argument(
        "--save-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Format to save feature matrix (default: parquet)"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Set paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Validate input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Starting feature engineering pipeline...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Add Elo: {not args.no_elo}")
    print(f"Save format: {args.save_format}")
    
    try:
        # Read cleaned data
        print("Reading cleaned fight data...")
        df = read_csv(input_path)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Add Elo ratings (enabled by default unless --no-elo is specified)
        if not args.no_elo:
            print("Adding Elo ratings...")
            # Ensure data is sorted by date for Elo calculation
            df_sorted = df.sort_values("Date").reset_index(drop=True)
            df_with_elo = build_elo(df_sorted)
            
            # Merge Elo ratings back to original data using Date, RedFighter, BlueFighter
            elo_columns = ["RedElo", "BlueElo", "EloDiff"]
            df = df.merge(
                df_with_elo[["Date", "RedFighter", "BlueFighter"] + elo_columns], 
                on=["Date", "RedFighter", "BlueFighter"], 
                how="left"
            )
            
            print("Elo ratings added successfully")
            print(f"Elo columns: {elo_columns}")
            print(f"Data shape after Elo merge: {df.shape}")
        else:
            print("Skipping Elo ratings (--no-elo flag specified)")
        
        # Assemble feature matrix
        print("Assembling feature matrix...")
        X, y, meta = assemble_feature_matrix(df)
        
        print(f"Feature matrix assembled:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  meta shape: {meta.shape}")
        
        # Generate feature summary
        print("Generating feature summary...")
        feature_summary = get_feature_summary(X, y)
        
        print("\nFeature Summary:")
        print(f"  Samples: {feature_summary['n_samples']}")
        print(f"  Features: {feature_summary['n_features']}")
        print(f"  Target distribution: {feature_summary['target_distribution']}")
        print(f"  Missing values: {feature_summary['missing_values']}")
        
        # Save feature matrix
        print("Saving feature matrix...")
        if args.save_format == "parquet":
            save_feature_matrix(X, y, meta, output_path)
        else:
            # Save as CSV
            output_path.mkdir(parents=True, exist_ok=True)
            X.to_csv(output_path / "X.csv", index=False)
            y.to_frame("Winner").to_csv(output_path / "y.csv", index=False)
            meta.to_csv(output_path / "meta.csv", index=False)
            print(f"Feature matrix saved to {output_path}")
            print(f"  X.csv: {X.shape}")
            print(f"  y.csv: {y.shape}")
            print(f"  meta.csv: {meta.shape}")
        
        # Save feature summary
        import json
        summary_path = output_path / "feature_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(feature_summary, f, indent=2, default=str)
        print(f"Feature summary saved to: {summary_path}")
        
        print("\nFeature engineering pipeline completed successfully!")
        
        # Print sample of features
        print(f"\nSample features (first 10):")
        print(X.columns[:10].tolist())
        
        if not args.no_elo:
            print(f"\nElo features included: {[col for col in X.columns if 'Elo' in col]}")
        
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
