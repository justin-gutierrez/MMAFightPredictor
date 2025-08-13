#!/usr/bin/env python3
"""
Data cleaning pipeline script for UFC ML predictor.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ufcml.clean import clean_prefight_columns, validate_cleaned_data, get_cleaning_summary
from ufcml.io import read_csv, write_csv, save_json
from ufcml.config import get_config


def main():
    """Main function for the data cleaning pipeline."""
    parser = argparse.ArgumentParser(
        description="Clean UFC fight data and save to interim directory"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to input raw CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output path (default: data/interim/prefight_clean.csv)"
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Run validation and save validation report"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Set default output path if not specified
    if args.output is None:
        output_path = config.INTERIM_DIR / "prefight_clean.csv"
    else:
        output_path = Path(args.output)
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Starting data cleaning pipeline...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    try:
        # Read raw data
        print("Reading raw data...")
        df_raw = read_csv(input_path)
        print(f"Raw data shape: {df_raw.shape}")
        
        # Clean data
        print("Cleaning data...")
        df_clean = clean_prefight_columns(df_raw)
        print(f"Cleaned data shape: {df_clean.shape}")
        
        # Generate cleaning summary
        summary = get_cleaning_summary(df_raw, df_clean)
        print("\nCleaning Summary:")
        print(f"  Original shape: {summary['original_shape']}")
        print(f"  Cleaned shape: {summary['cleaned_shape']}")
        print(f"  Rows removed: {summary['rows_removed']}")
        print(f"  Columns removed: {summary['columns_removed']}")
        print(f"  Duplicates removed: {summary['duplicates_removed']}")
        
        # Save cleaned data
        print("Saving cleaned data...")
        write_csv(df_clean, output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
        # Run validation if requested
        if args.validate:
            print("Running validation...")
            validation_results = validate_cleaned_data(df_clean)
            
            # Save validation report
            validation_report_path = output_path.parent / "cleaning_validation_report.json"
            save_json(validation_results, validation_report_path)
            print(f"Validation report saved to: {validation_report_path}")
            
            # Print validation summary
            print("\nValidation Summary:")
            print(f"  Total rows: {validation_results['total_rows']}")
            print(f"  Total columns: {validation_results['total_columns']}")
            print(f"  Duplicate rows: {validation_results['duplicate_rows']}")
            
            if validation_results['missing_required_columns']:
                print(f"  Missing required columns: {validation_results['missing_required_columns']}")
            else:
                print("  All required columns present")
        
        print("\nData cleaning pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
