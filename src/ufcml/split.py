"""
Data splitting strategies for UFC ML predictor.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime


def time_based_split(
    meta_df: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 3,
    group_by: str = "month",
    min_train_size: int = 2500,
    min_valid_size: int = 800,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Ensure meta_df has Date column
    if "Date" not in meta_df.columns:
        raise ValueError("meta_df must contain 'Date' column")

    dates = pd.to_datetime(meta_df["Date"])

    if group_by == "month":
        return _month_based_split(meta_df, dates, n_folds, min_train_size, min_valid_size)
    else:
        return _date_based_split(meta_df, dates, n_folds)


def _month_based_split(
    meta_df: pd.DataFrame,
    dates: pd.Series,
    n_folds: int,
    min_train_size: int,
    min_valid_size: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window, month-grouped CV:
      - Start with a warm-up training window (>= min_train_size rows).
      - Next contiguous months become validation (>= min_valid_size rows).
      - Append (train, valid) fold.
      - Expand training to include that validation, then repeat until n_folds or data exhausted.
    """
    # Group rows by month in chronological order
    month_periods = dates.dt.to_period("M")
    g = meta_df.assign(_month=month_periods).sort_values("Date").reset_index()
    groups: list[tuple[pd.Period, np.ndarray]] = []
    for m, dfm in g.groupby("_month", sort=True):
        groups.append((m, dfm["index"].to_numpy()))

    if not groups:
        return []

    sizes = [len(ix) for _, ix in groups]

    # Build initial warm-up training window
    train_end = 0
    train_count = 0
    while train_end < len(groups) and train_count < min_train_size:
        train_count += sizes[train_end]
        train_end += 1

    if train_count < min_train_size:
        raise ValueError(
            f"Not enough data to satisfy min_train_size={min_train_size}. "
            f"Only {train_count} rows available."
        )

    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    # Build up to n_folds validation windows
    while len(folds) < n_folds and train_end < len(groups):
        # Accumulate months for validation until we reach min_valid_size
        valid_start = train_end
        valid_end = valid_start
        valid_count = 0
        while valid_end < len(groups) and valid_count < min_valid_size:
            valid_count += sizes[valid_end]
            valid_end += 1

        if valid_count == 0:
            break  # no more data left for a validation window

        # Stitch indices
        train_idx = np.concatenate([ix for _, ix in groups[:train_end]])
        valid_idx = np.concatenate([ix for _, ix in groups[valid_start:valid_end]])

        # Safety: ensure time ordering (max train date < min valid date)
        max_train_date = dates.iloc[train_idx].max()
        min_valid_date = dates.iloc[valid_idx].min()
        if max_train_date >= min_valid_date:
            raise ValueError(
                "Time ordering violated: max train date "
                f"{max_train_date} >= min valid date {min_valid_date}"
            )

        folds.append((train_idx, valid_idx))

        # Expand training to include the just-used validation window
        train_end = valid_end

    return folds


def _date_based_split(meta_df: pd.DataFrame, dates: pd.Series, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create date-based cross-validation splits (original implementation).
    
    Args:
        meta_df: DataFrame with fight data
        dates: Series of datetime dates
        n_folds: Number of cross-validation folds
        
    Returns:
        List of tuples: [(train_idx, valid_idx), ...] for each fold
    """
    # Sort by date ascending
    sorted_indices = dates.sort_values().index
    sorted_dates = dates.loc[sorted_indices]
    
    # Get unique dates to ensure whole dates stay in the same fold
    unique_dates = sorted_dates.unique()
    n_unique_dates = len(unique_dates)
    
    if n_folds > n_unique_dates:
        raise ValueError(f"Number of folds ({n_folds}) cannot exceed number of unique dates ({n_unique_dates})")
    
    # Calculate fold boundaries
    fold_size = n_unique_dates // n_folds
    remainder = n_unique_dates % n_folds
    
    splits = []
    
    for fold in range(n_folds):
        # Calculate start and end indices for this fold
        start_idx = fold * fold_size + min(fold, remainder)
        end_idx = start_idx + fold_size + (1 if fold < remainder else 0)
        
        # Get the cutoff date for this fold
        if fold < n_folds - 1:
            cutoff_date = unique_dates[end_idx - 1]
        else:
            # Last fold: use all remaining data
            cutoff_date = unique_dates[-1]
        
        # Find indices for training (before cutoff) and validation (at cutoff)
        if fold == 0:
            # First fold: no training data, only validation
            train_indices = np.array([], dtype=int)
        else:
            # Training: all data before the cutoff date
            train_mask = sorted_dates < cutoff_date
            train_indices = sorted_indices[train_mask].values
        
        # Validation: data at the cutoff date
        valid_mask = sorted_dates == cutoff_date
        valid_indices = sorted_indices[valid_mask].values
        
        # Only add fold if there's validation data
        if len(valid_indices) > 0:
            splits.append((train_indices, valid_indices))
    
    return splits


def final_holdout_split(meta_df: pd.DataFrame, cutoff_date: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create final holdout split for model evaluation.
    
    This function splits the data into training and test sets based on a cutoff date.
    All fights before the cutoff date are used for training, and all fights on or after
    the cutoff date are held out for final testing.
    
    Args:
        meta_df: DataFrame with Date column
        cutoff_date: Date string in format 'YYYY-MM-DD' or similar
        
    Returns:
        Tuple of (train_idx, test_idx) arrays
        
    Example:
        >>> train_idx, test_idx = final_holdout_split(meta_df, "2024-01-01")
        >>> print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    """
    # Ensure meta_df has Date column
    if "Date" not in meta_df.columns:
        raise ValueError("meta_df must contain 'Date' column")
    
    # Convert cutoff_date to datetime
    try:
        cutoff_dt = pd.to_datetime(cutoff_date)
    except ValueError:
        raise ValueError(f"Invalid date format: {cutoff_date}. Use format like 'YYYY-MM-DD'")
    
    # Convert meta_df dates to datetime
    dates = pd.to_datetime(meta_df["Date"])
    
    # Create masks for train and test
    train_mask = dates < cutoff_dt
    test_mask = dates >= cutoff_dt
    
    # Get indices
    train_idx = meta_df[train_mask].index.values
    test_idx = meta_df[test_mask].index.values
    
    return train_idx, test_idx


def get_split_summary(splits: List[Tuple[np.ndarray, np.ndarray]], 
                     meta_df: pd.DataFrame, 
                     y: pd.Series) -> Dict[str, Any]:
    """
    Generate summary statistics for the data splits.
    
    Args:
        splits: List of (train_idx, valid_idx) tuples from time_based_split
        meta_df: DataFrame with Date column
        y: Target variable series
        
    Returns:
        Dictionary with split statistics
    """
    # Ensure Date column is datetime
    dates = pd.to_datetime(meta_df["Date"])
    
    summary = {
        "n_folds": len(splits),
        "fold_details": [],
        "total_samples": len(meta_df),
        "date_range": {
            "start": dates.min().strftime("%Y-%m-%d"),
            "end": dates.max().strftime("%Y-%m-%d")
        }
    }
    
    for i, (train_idx, valid_idx) in enumerate(splits):
        fold_info = {
            "fold": i + 1,
            "train_samples": len(train_idx),
            "valid_samples": len(valid_idx),
            "train_red_wins": int(y.iloc[train_idx].sum()) if len(train_idx) > 0 else 0,
            "valid_red_wins": int(y.iloc[valid_idx].sum()) if len(valid_idx) > 0 else 0,
            "train_red_win_rate": float(y.iloc[train_idx].mean()) if len(train_idx) > 0 else 0.0,
            "valid_red_win_rate": float(y.iloc[valid_idx].mean()) if len(valid_idx) > 0 else 0.0
        }
        
        # Add date ranges for this fold
        if len(train_idx) > 0:
            train_dates = pd.to_datetime(meta_df.iloc[train_idx]["Date"])
            fold_info["train_date_range"] = {
                "start": train_dates.min().strftime("%Y-%m-%d"),
                "end": train_dates.max().strftime("%Y-%m-%d")
            }
        if len(valid_idx) > 0:
            valid_dates = pd.to_datetime(meta_df.iloc[valid_idx]["Date"])
            fold_info["valid_date_range"] = {
                "start": valid_dates.min().strftime("%Y-%m-%d"),
                "end": valid_dates.max().strftime("%Y-%m-%d")
            }
        
        summary["fold_details"].append(fold_info)
    
    return summary


def validate_splits(splits: List[Tuple[np.ndarray, np.ndarray]], 
                   meta_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that splits are properly time-ordered with no data leakage.
    
    Args:
        splits: List of (train_idx, valid_idx) tuples
        meta_df: DataFrame with Date column
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    dates = pd.to_datetime(meta_df["Date"])
    
    for i, (train_idx, valid_idx) in enumerate(splits):
        if len(train_idx) == 0:
            continue
            
        if len(valid_idx) == 0:
            validation_results["warnings"].append(f"Fold {i+1}: No validation data")
            continue
        
        # Check that all training dates are before validation dates
        train_dates = dates.iloc[train_idx]
        valid_dates = dates.iloc[valid_idx]
        
        max_train_date = train_dates.max()
        min_valid_date = valid_dates.min()
        
        if max_train_date >= min_valid_date:
            validation_results["is_valid"] = False
            validation_results["errors"].append(
                f"Fold {i+1}: Data leakage detected. "
                f"Max train date ({max_train_date.strftime('%Y-%m-%d')}) >= "
                f"Min valid date ({min_valid_date.strftime('%Y-%m-%d')})"
            )
    
    return validation_results


def save_splits(splits: List[Tuple[np.ndarray, np.ndarray]], 
                output_dir: str, 
                split_name: str = "cv_splits") -> None:
    """
    Save split indices to JSON files.
    
    Args:
        splits: List of (train_idx, valid_idx) tuples
        output_dir: Directory to save split files
        split_name: Base name for the split files
        
    Example:
        >>> save_splits(splits, "data/processed/splits", "time_based_cv")
    """
    import json
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save each fold separately
    for i, (train_idx, valid_idx) in enumerate(splits):
        fold_data = {
            "fold": i + 1,
            "train_indices": train_idx.tolist(),
            "valid_indices": valid_idx.tolist(),
            "n_train": len(train_idx),
            "n_valid": len(valid_idx)
        }
        
        fold_filename = f"{split_name}_fold_{i+1}.json"
        with open(output_path / fold_filename, 'w') as f:
            json.dump(fold_data, f, indent=2)
    
    # Save summary file
    summary_data = {
        "split_name": split_name,
        "n_folds": len(splits),
        "total_samples": sum(len(train_idx) + len(valid_idx) for train_idx, valid_idx in splits)
    }
    
    summary_filename = f"{split_name}_summary.json"
    with open(output_path / summary_filename, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Saved {len(splits)} splits to {output_path}")
    print(f"Summary saved to: {output_path / summary_filename}")


def load_splits(splits_dir: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load saved split indices from JSON files.
    
    Args:
        splits_dir: Directory containing split files
        
    Returns:
        List of (train_idx, valid_idx) tuples
        
    Example:
        >>> splits = load_splits("data/processed/splits")
    """
    import json
    from pathlib import Path
    
    splits_path = Path(splits_dir)
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
    
    # Find all fold files
    fold_files = list(splits_path.glob("*_fold_*.json"))
    if not fold_files:
        raise FileNotFoundError(f"No fold files found in {splits_dir}")
    
    # Sort by fold number
    fold_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    splits = []
    for fold_file in fold_files:
        with open(fold_file, 'r') as f:
            fold_data = json.load(f)
            
        train_idx = np.array(fold_data["train_indices"])
        valid_idx = np.array(fold_data["valid_indices"])
        splits.append((train_idx, valid_idx))
    
    print(f"Loaded {len(splits)} splits from {splits_dir}")
    for i, (train_idx, valid_idx) in enumerate(splits):
        print(f"  Split {i+1}: Train {len(train_idx)}, Valid {len(valid_idx)}")
    
    return splits
