"""
Feature engineering for UFC ML predictor.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def add_stance_matchup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create categorical feature for stance matchups between fighters.
    
    Args:
        df: DataFrame with RedStance and BlueStance columns
        
    Returns:
        pd.DataFrame: DataFrame with added StanceMatchup column
        
    Example:
        >>> df = add_stance_matchup(df)
        >>> print(df["StanceMatchup"].value_counts())
        orthodox_vs_southpaw    150
        orthodox_vs_orthodox    120
        southpaw_vs_southpaw     80
        unknown                  50
    """
    df = df.copy()
    
    # Handle missing stances
    red_stance = df["RedStance"].fillna("unknown").str.lower()
    blue_stance = df["BlueStance"].fillna("unknown").str.lower()
    
    # Create stance matchup feature
    stance_matchup = red_stance + "_vs_" + blue_stance
    
    # Replace "unknown_vs_unknown" with just "unknown"
    stance_matchup = stance_matchup.replace("unknown_vs_unknown", "unknown")
    
    df["StanceMatchup"] = stance_matchup
    
    return df


def add_rank_diffs(df: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    """
    Create rank difference features between red and blue fighters.
    
    Args:
        df: DataFrame with ranking columns
        drop_original: Whether to drop original rank columns
        
    Returns:
        pd.DataFrame: DataFrame with added rank difference columns
        
    Example:
        >>> df = add_rank_diffs(df)
        >>> print(df[["MatchRankDiff", "PfpRankDiff"]].head())
    """
    df = df.copy()
    
    # Match weight class rank difference
    if "RMatchWCRank" in df.columns and "BMatchWCRank" in df.columns:
        df["MatchRankDiff"] = df["RMatchWCRank"] - df["BMatchWCRank"]
    
    # P4P rank difference (coalesce missing values to 100)
    if "RPFPRank" in df.columns and "BPFPRank" in df.columns:
        r_pfp = df["RPFPRank"].fillna(100)
        b_pfp = df["BPFPRank"].fillna(100)
        df["PfpRankDiff"] = r_pfp - b_pfp
    
    # Drop original rank columns if requested
    if drop_original:
        rank_cols_to_drop = ["RMatchWCRank", "BMatchWCRank", "RPFPRank", "BPFPRank"]
        df = df.drop(columns=[col for col in rank_cols_to_drop if col in df.columns])
    
    return df


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode categorical variables for machine learning.
    
    Args:
        df: DataFrame with categorical columns to encode
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Encoded DataFrame and list of feature column names
        
    Example:
        >>> df_encoded, feature_cols = encode_categoricals(df)
        >>> print(f"Number of features: {len(feature_cols)}")
    """
    df = df.copy()
    feature_columns = []
    
    # One-hot encode WeightClass (top 20 + other)
    if "WeightClass" in df.columns:
        weight_class_counts = df["WeightClass"].value_counts()
        top_weight_classes = weight_class_counts.head(20).index.tolist()
        
        # Create binary columns for top weight classes
        for wc in top_weight_classes:
            col_name = f"WeightClass_{wc.replace(' ', '_').replace('-', '_')}"
            df[col_name] = (df["WeightClass"] == wc).astype(int)
            feature_columns.append(col_name)
        
        # Add "other" category for remaining weight classes
        df["WeightClass_other"] = (~df["WeightClass"].isin(top_weight_classes)).astype(int)
        feature_columns.append("WeightClass_other")
    
    # One-hot encode Gender
    if "Gender" in df.columns:
        gender_dummies = pd.get_dummies(df["Gender"], prefix="Gender")
        df = pd.concat([df, gender_dummies], axis=1)
        feature_columns.extend(gender_dummies.columns.tolist())
    
    # One-hot encode Country (top 20 + other)
    if "Country" in df.columns:
        country_counts = df["Country"].value_counts()
        top_countries = country_counts.head(20).index.tolist()
        
        # Create binary columns for top countries
        for country in top_countries:
            col_name = f"Country_{country.replace(' ', '_').replace('-', '_')}"
            df[col_name] = (df["Country"] == country).astype(int)
            feature_columns.append(col_name)
        
        # Add "other" category for remaining countries
        df["Country_other"] = (~df["Country"].isin(top_countries)).astype(int)
        feature_columns.append("Country_other")
    
    # Convert boolean columns to int
    if "TitleBout" in df.columns:
        df["TitleBout"] = df["TitleBout"].astype(int)
        feature_columns.append("TitleBout")
    
    if "EmptyArena" in df.columns:
        df["EmptyArena"] = df["EmptyArena"].astype(int)
        feature_columns.append("EmptyArena")
    
    # One-hot encode StanceMatchup
    if "StanceMatchup" in df.columns:
        stance_dummies = pd.get_dummies(df["StanceMatchup"], prefix="Stance")
        df = pd.concat([df, stance_dummies], axis=1)
        feature_columns.extend(stance_dummies.columns.tolist())
    
    return df, feature_columns


def assemble_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Assemble complete feature matrix for machine learning.
    
    Args:
        df: DataFrame with fight data
        
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]: Feature matrix X, target y, meta columns
        
    Example:
        >>> X, y, meta = assemble_feature_matrix(df)
        >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    """
    df = df.copy()
    
    # Apply feature engineering functions
    df = add_stance_matchup(df)
    df = add_rank_diffs(df, drop_original=False)
    
    # Select numeric difference features
    numeric_diff_features = [
        "AgeDif", "HeightDif", "ReachDif", "WinStreakDif", "LoseStreakDif",
        "LongestWinStreakDif", "WinDif", "LossDif", "TotalRoundDif", 
        "TotalTitleBoutDif", "KODif", "SubDif", "SigStrDif", "AvgSubAttDif", "AvgTDDif"
    ]
    
    # Add EloDiff if present
    if "EloDiff" in df.columns:
        numeric_diff_features.append("EloDiff")
    
    # Add odds if present (optional)
    odds_features = []
    if "RedOdds" in df.columns:
        odds_features.append("RedOdds")
    if "BlueOdds" in df.columns:
        odds_features.append("BlueOdds")
    
    # Combine all numeric features
    all_numeric_features = numeric_diff_features + odds_features
    
    # Filter to only existing columns
    existing_numeric_features = [col for col in all_numeric_features if col in df.columns]
    
    # Encode categorical variables
    df_encoded, categorical_features = encode_categoricals(df)
    
    # Select all feature columns
    all_feature_columns = existing_numeric_features + categorical_features
    
    # Create feature matrix X
    X = df_encoded[all_feature_columns].copy()
    
    # Handle missing values in numeric features
    X = X.fillna(0)
    
    # Create target variable y (Winner=="Red" as 1, else 0)
    if "Winner" in df.columns:
        y = (df["Winner"] == "Red").astype(int)
    else:
        raise ValueError("Winner column not found in DataFrame")
    
    # Create meta columns
    meta_columns = ["Date", "RedFighter", "BlueFighter"]
    meta = df[meta_columns].copy()
    
    return X, y, meta


def get_feature_summary(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Generate summary statistics for the feature matrix.
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        "n_samples": int(len(X)),
        "n_features": int(len(X.columns)),
        "feature_names": X.columns.tolist(),
        "target_distribution": y.value_counts().to_dict(),
        "missing_values": int(X.isnull().sum().sum()),
        "feature_types": {str(k): str(v) for k, v in X.dtypes.value_counts().to_dict().items()}
    }
    
    return summary


def save_feature_matrix(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, 
                       output_dir: str) -> None:
    """
    Save feature matrix components to parquet files.
    
    Args:
        X: Feature matrix
        y: Target variable
        meta: Meta columns
        output_dir: Directory to save files
        
    Example:
        >>> save_feature_matrix(X, y, meta, "data/processed")
    """
    import os
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save feature matrix
    X.to_parquet(output_path / "X.parquet")
    
    # Save target variable
    y.to_frame("Winner").to_parquet(output_path / "y.parquet")
    
    # Save meta columns
    meta.to_parquet(output_path / "meta.parquet")
    
    print(f"Feature matrix saved to {output_path}")
    print(f"  X.parquet: {X.shape}")
    print(f"  y.parquet: {y.shape}")
    print(f"  meta.parquet: {meta.shape}")
