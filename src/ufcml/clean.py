"""
Data cleaning and preprocessing for UFC ML predictor.
"""

import pandas as pd
from typing import List


def clean_prefight_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize pre-fight UFC data columns.
    
    This function processes raw UFC fight data by:
    - Keeping only essential pre-fight columns
    - Converting data types appropriately
    - Cleaning string values
    - Removing duplicates
    
    Args:
        df: Raw UFC fight DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized columns and data types
        
    Example:
        >>> raw_df = pd.read_csv("data/raw/ufc-mastercsv.csv")
        >>> clean_df = clean_prefight_columns(raw_df)
        >>> print(clean_df.shape)
        (5000, 40)
    """
    # Define the columns to keep (only if present in the data)
    columns_to_keep = [
        "RedFighter", "BlueFighter", "Date", "Country", "TitleBout", 
        "WeightClass", "Gender", "NumberOfRounds", "EmptyArena",
        "AgeDif", "HeightDif", "ReachDif", "WinStreakDif", "LoseStreakDif", 
        "LongestWinStreakDif", "WinDif", "LossDif", "TotalRoundDif", 
        "TotalTitleBoutDif", "KODif", "SubDif", "SigStrDif", "AvgSubAttDif", 
        "AvgTDDif", "RedStance", "BlueStance", "RMatchWCRank", "BMatchWCRank", 
        "RPFPRank", "BPFPRank", "RedOdds", "BlueOdds", "Winner"
    ]
    
    # Add optional historical columns if present (for computing form features)
    optional_hist_cols = ["Finish", "FinishDetails"]
    for col in optional_hist_cols:
        if col in df.columns and col not in columns_to_keep:
            columns_to_keep.append(col)
    
    # Filter to only keep columns that exist in the DataFrame
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df_clean = df[existing_columns].copy()
    
    # Convert Date to datetime
    if "Date" in df_clean.columns:
        df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors='coerce')
    
    # Convert boolean columns
    boolean_columns = ["TitleBout", "EmptyArena"]
    for col in boolean_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(bool)
    
    # Convert numeric columns
    numeric_columns = [
        "AgeDif", "HeightDif", "ReachDif", "WinStreakDif", "LoseStreakDif",
        "LongestWinStreakDif", "WinDif", "LossDif", "TotalRoundDif",
        "TotalTitleBoutDif", "KODif", "SubDif", "SigStrDif", "AvgSubAttDif",
        "AvgTDDif", "RMatchWCRank", "BMatchWCRank", "RPFPRank", "BPFPRank",
        "RedOdds", "BlueOdds", "NumberOfRounds"
    ]
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Clean string columns (strip whitespace and handle missing values)
    string_columns = ["RedFighter", "BlueFighter", "Country", "WeightClass", 
                     "Gender", "RedStance", "BlueStance", "Winner"]
    
    for col in string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            # Replace 'nan' strings with actual NaN
            df_clean[col] = df_clean[col].replace('nan', pd.NA)
    
    # Drop duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean


def validate_cleaned_data(df: pd.DataFrame) -> dict:
    """
    Validate the cleaned DataFrame for data quality issues.
    
    Args:
        df: Cleaned DataFrame to validate
        
    Returns:
        dict: Dictionary containing validation results and statistics
    """
    validation_results = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "data_types": df.dtypes.to_dict()
    }
    
    # Check for required columns
    required_columns = ["RedFighter", "BlueFighter", "Winner"]
    missing_required = [col for col in required_columns if col not in df.columns]
    validation_results["missing_required_columns"] = missing_required
    
    return validation_results


def get_cleaning_summary(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    Generate a summary of the cleaning process.
    
    Args:
        df_before: Original DataFrame before cleaning
        df_after: Cleaned DataFrame after cleaning
        
    Returns:
        dict: Summary statistics of the cleaning process
    """
    summary = {
        "original_shape": df_before.shape,
        "cleaned_shape": df_after.shape,
        "rows_removed": df_before.shape[0] - df_after.shape[0],
        "columns_removed": df_before.shape[1] - df_after.shape[1],
        "duplicates_removed": df_before.duplicated().sum() - df_after.duplicated().sum()
    }
    
    return summary
