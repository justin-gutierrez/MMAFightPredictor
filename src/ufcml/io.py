"""
Data input/output operations for UFC ML predictor.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import joblib


def read_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.
    
    Args:
        path: Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        pd.errors.EmptyDataError: If the file is empty.
        
    Example:
        >>> df = read_csv("data/raw/fights.csv")
        >>> print(df.shape)
        (1000, 15)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """
    Write a pandas DataFrame to a CSV file.
    
    Args:
        df: DataFrame to write.
        path: Path where to save the CSV file.
        **kwargs: Additional arguments passed to pd.DataFrame.to_csv().
        
    Example:
        >>> write_csv(df, "data/processed/features.csv", index=False)
    """
    path = Path(path)
    mkdir_p(path.parent)
    
    df.to_csv(path, **kwargs)


def mkdir_p(path: Union[str, Path]) -> None:
    """
    Create directories recursively (equivalent to mkdir -p).
    
    Args:
        path: Directory path to create.
        
    Example:
        >>> mkdir_p("data/interim/features")
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_joblib(obj: Any, path: Union[str, Path]) -> None:
    """
    Save a Python object using joblib.
    
    Args:
        obj: Object to save.
        path: Path where to save the object.
        
    Example:
        >>> save_joblib(model, "data/models/xgboost_model.joblib")
    """
    path = Path(path)
    mkdir_p(path.parent)
    
    joblib.dump(obj, path)


def load_joblib(path: Union[str, Path]) -> Any:
    """
    Load a Python object using joblib.
    
    Args:
        path: Path to the saved object.
        
    Returns:
        Any: The loaded object.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        
    Example:
        >>> model = load_joblib("data/models/xgboost_model.joblib")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Joblib file not found: {path}")
    
    return joblib.load(path)


def save_json(data: Dict[str, Any], path: Union[str, Path], **kwargs) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save.
        path: Path where to save the JSON file.
        **kwargs: Additional arguments passed to json.dump().
        
    Example:
        >>> save_json({"accuracy": 0.75, "f1_score": 0.72}, "data/reports/metrics.json")
    """
    path = Path(path)
    mkdir_p(path.parent)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, **kwargs)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.
    
    Args:
        path: Path to the JSON file.
        
    Returns:
        Dict[str, Any]: Dictionary containing the JSON data.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        
    Example:
        >>> metrics = load_json("data/reports/metrics.json")
        >>> print(metrics["accuracy"])
        0.75
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def file_exists(path: Union[str, Path]) -> bool:
    """
    Check if a file exists.
    
    Args:
        path: Path to check.
        
    Returns:
        bool: True if file exists, False otherwise.
        
    Example:
        >>> if file_exists("data/models/model.joblib"):
        ...     model = load_joblib("data/models/model.joblib")
    """
    return Path(path).exists()


def get_file_size(path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        path: Path to the file.
        
    Returns:
        int: File size in bytes.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        
    Example:
        >>> size = get_file_size("data/raw/fights.csv")
        >>> print(f"File size: {size / 1024:.2f} KB")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return path.stat().st_size
