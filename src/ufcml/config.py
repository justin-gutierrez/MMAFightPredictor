"""
Configuration management for UFC ML predictor.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration class for UFC ML predictor paths and settings."""
    
    RAW_DIR: Path
    INTERIM_DIR: Path
    PROCESSED_DIR: Path
    MODELS_DIR: Path
    REPORTS_DIR: Path
    
    # Model and training constants
    RANDOM_STATE: int = 42
    N_THREADS: int = -1
    
    def __post_init__(self):
        """Ensure all directories exist after initialization."""
        for path in [self.RAW_DIR, self.INTERIM_DIR, self.PROCESSED_DIR, 
                    self.MODELS_DIR, self.REPORTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)


def load_config(project_path: Optional[str] = None) -> Config:
    """
    Load configuration with paths rooted at the project directory.
    
    Args:
        project_path: Optional path to project root. If None, uses current working directory.
        
    Returns:
        Config: Configuration object with all paths and settings.
        
    Example:
        >>> config = load_config()
        >>> print(config.RAW_DIR)
        PosixPath('/path/to/project/data/raw')
    """
    if project_path is None:
        project_path = os.getcwd()
    
    project_root = Path(project_path)
    
    # Define data directories relative to project root
    data_dir = project_root / "data"
    
    config = Config(
        RAW_DIR=data_dir / "raw",
        INTERIM_DIR=data_dir / "interim", 
        PROCESSED_DIR=data_dir / "processed",
        MODELS_DIR=data_dir / "models",
        REPORTS_DIR=data_dir / "reports"
    )
    
    return config


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance, creating it if it doesn't exist.
    
    Returns:
        Config: Global configuration instance.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """
    Set the global configuration instance.
    
    Args:
        config: Configuration object to set as global.
    """
    global _config
    _config = config
