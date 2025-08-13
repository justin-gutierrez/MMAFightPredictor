"""
Feature engineering for UFC ML predictor.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

# Constants for finish type classification
FINISH_KO_LABELS = {"KO/TKO", "KO", "TKO", "TKO - Doctor's Stoppage", "Doctor Stoppage", "TKO - Doctor Stoppage"}
FINISH_SUB_LABELS = {"Submission", "SUB"}


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


def _normalize_long_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide fight rows into per-fighter long rows.
    Columns required: Date, RedFighter, BlueFighter, Winner.
    Optional: Finish (to derive recent KO/SUB shares).
    Output columns: [Date, Fighter, Opponent, Won, FinishType]
    """
    base_cols = ["Date", "RedFighter", "BlueFighter", "Winner"]
    for c in base_cols:
        if c not in df.columns:
            raise ValueError(f"Required column missing for long view: {c}")

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])

    # Map finish type into coarse labels (optional)
    finish = d["Finish"].fillna("DEC") if "Finish" in d.columns else pd.Series(["DEC"] * len(d), index=d.index)
    # Standardize a coarse finish label per bout, used for both fighters equally
    finish_type = np.where(finish.astype(str).str.upper().isin({s.upper() for s in FINISH_KO_LABELS}), "KO",
                    np.where(finish.astype(str).str.upper().str.contains("SUB"), "SUB", "DEC"))

    # Red rows
    red = pd.DataFrame({
        "Date": d["Date"].values,
        "Fighter": d["RedFighter"].values,
        "Opponent": d["BlueFighter"].values,
        "Won": (d["Winner"].astype(str).str.upper() == "RED").astype(int).values,
        "FinishType": finish_type
    })

    # Blue rows
    blue = pd.DataFrame({
        "Date": d["Date"].values,
        "Fighter": d["BlueFighter"].values,
        "Opponent": d["RedFighter"].values,
        "Won": (d["Winner"].astype(str).str.upper() == "BLUE").astype(int).values,
        "FinishType": finish_type
    })

    long_df = pd.concat([red, blue], ignore_index=True)
    long_df = long_df.sort_values(["Fighter", "Date"]).reset_index(drop=True)
    return long_df


def _rolling_last3_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lagged rolling features for a single fighter, sorted by Date.
    Ensures no leakage via shift(1).
    """
    g = g.sort_values("Date").copy()
    # last 3 fights win rate (shifted)
    g["RecentWinRate3"] = g["Won"].rolling(window=3, min_periods=1).mean().shift(1)

    # recent KO/SUB share (based on last 3 fights) if FinishType present
    if "FinishType" in g.columns:
        g["is_ko"] = (g["FinishType"] == "KO").astype(int)
        g["is_sub"] = (g["FinishType"] == "SUB").astype(int)
        g["RecentKOShare3"] = g["is_ko"].rolling(window=3, min_periods=1).mean().shift(1)
        g["RecentSubShare3"] = g["is_sub"].rolling(window=3, min_periods=1).mean().shift(1)
    else:
        g["RecentKOShare3"] = np.nan
        g["RecentSubShare3"] = np.nan

    # fights in past 365 days (strictly before current bout)
    g["FightsPast365"] = (
        g.set_index("Date")
         .assign(one=1)
         .groupby(level=0)["one"]
         .transform(lambda s: 1)  # placeholder; we'll compute with rolling window below
    )
    # efficient time-based count: for each row, count prior rows within 365 days
    # We'll use expanding window and search; vectorized approach:
    dates = g["Date"].values.astype("datetime64[D]")
    counts = np.zeros(len(g), dtype=int)
    start = 0
    for i in range(len(g)):
        # move start until dates[i] - dates[start] < 365 days
        while start < i and (dates[i] - dates[start]).astype("timedelta64[D]").astype(int) >= 365:
            start += 1
        counts[i] = i - start  # number of prior fights within 365d
    g["FightsPast365"] = pd.Series(counts, index=g.index)

    # days since last fight (lagged)
    g["DaysSinceLastFight"] = g["Date"].diff().dt.days.shift(0)  # diff from previous
    g["DaysSinceLastFight"] = g["DaysSinceLastFight"].shift(1)   # shift so current bout sees prior gap only

    # Fill early NaNs
    g["RecentWinRate3"] = g["RecentWinRate3"].fillna(0.5)  # neutral prior
    g["RecentKOShare3"] = g["RecentKOShare3"].fillna(0.0)
    g["RecentSubShare3"] = g["RecentSubShare3"].fillna(0.0)
    g["FightsPast365"] = g["FightsPast365"].fillna(0).astype(int)
    g["DaysSinceLastFight"] = g["DaysSinceLastFight"].fillna(400)  # large layoff default

    return g


def _compute_form_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build long per-fighter table and compute lagged form metrics.
    Returns a dataframe with columns:
      Fighter, Date, RecentWinRate3, FightsPast365, DaysSinceLastFight, RecentKOShare3, RecentSubShare3
    """
    long_df = _normalize_long_view(df)
    long_df = long_df.groupby("Fighter", group_keys=False).apply(_rolling_last3_features)
    cols = ["Fighter", "Date", "RecentWinRate3", "FightsPast365", "DaysSinceLastFight", "RecentKOShare3", "RecentSubShare3"]
    return long_df[cols]


def add_short_term_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge short-term form features back to the wide fight table and create DIFF features:
      - RecentWinRate3Dif
      - FightsPast365Dif
      - DaysSinceLastFightDif
      - RecentKOShare3Dif (if Finish present)
      - RecentSubShare3Dif (if Finish present)
    Also returns the df with intermediate per-corner columns dropped, keeping only diffs.
    """
    if not {"RedFighter", "BlueFighter", "Date", "Winner"}.issubset(df.columns):
        raise ValueError("add_short_term_form_features requires RedFighter, BlueFighter, Date, Winner columns.")

    form_tbl = _compute_form_table(df)

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])

    # Merge for Red
    d = d.merge(form_tbl.add_prefix("Red_"), left_on=["RedFighter", "Date"],
                right_on=["Red_Fighter", "Red_Date"], how="left")
    # Merge for Blue
    d = d.merge(form_tbl.add_prefix("Blue_"), left_on=["BlueFighter", "Date"],
                right_on=["Blue_Fighter", "Blue_Date"], how="left")

    # Compute diffs (Red - Blue)
    d["RecentWinRate3Dif"]     = d["Red_RecentWinRate3"] - d["Blue_RecentWinRate3"]
    d["FightsPast365Dif"]      = d["Red_FightsPast365"] - d["Blue_FightsPast365"]
    d["DaysSinceLastFightDif"] = d["Red_DaysSinceLastFight"] - d["Blue_DaysSinceLastFight"]
    d["RecentKOShare3Dif"]     = d["Red_RecentKOShare3"] - d["Blue_RecentKOShare3"]
    d["RecentSubShare3Dif"]    = d["Red_RecentSubShare3"] - d["Blue_RecentSubShare3"]

    # Drop intermediate merged columns
    drop_cols = [c for c in d.columns if c.startswith("Red_") or c.startswith("Blue_")]
    d = d.drop(columns=drop_cols, errors="ignore")

    # Fill any residual NaNs (e.g., fighters with <1 prior bout)
    d["RecentWinRate3Dif"]     = d["RecentWinRate3Dif"].fillna(0.0)
    d["FightsPast365Dif"]      = d["FightsPast365Dif"].fillna(0)
    d["DaysSinceLastFightDif"] = d["DaysSinceLastFightDif"].fillna(0.0)
    d["RecentKOShare3Dif"]     = d["RecentKOShare3Dif"].fillna(0.0)
    d["RecentSubShare3Dif"]    = d["RecentSubShare3Dif"].fillna(0.0)

    return d


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
    
    # Add short-term form features (after other feature engineering)
    df = add_short_term_form_features(df)
    
    # Select numeric difference features
    numeric_diff_features = [
        "AgeDif", "HeightDif", "ReachDif", "WinStreakDif", "LoseStreakDif",
        "LongestWinStreakDif", "WinDif", "LossDif", "TotalRoundDif", 
        "TotalTitleBoutDif", "KODif", "SubDif", "SigStrDif", "AvgSubAttDif", "AvgTDDif",
        # Elo
        "EloDiff",
        # NEW short-term form diffs
        "RecentWinRate3Dif", "FightsPast365Dif", "DaysSinceLastFightDif",
        "RecentKOShare3Dif", "RecentSubShare3Dif"
    ]
    
    # Only keep those that exist
    numeric_diff_features = [c for c in numeric_diff_features if c in df.columns]
    
    # Add EloDiff if present (keep for backward compatibility)
    if "EloDiff" in df.columns and "EloDiff" not in numeric_diff_features:
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
