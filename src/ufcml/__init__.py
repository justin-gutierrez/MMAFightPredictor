"""
UFC Machine Learning Predictor Package

A comprehensive ML pipeline for UFC fight outcome prediction.
"""

__version__ = "0.1.0"
__author__ = "UFC ML Team"
__email__ = "team@ufcml.com"

from . import (
    config,
    io,
    clean,
    features,
    elo,
    split,
    train,
    calibrate,
    evaluate,
    predict,
    api,
    utils,
    calibration,
)

__all__ = [
    "config",
    "io", 
    "clean",
    "features",
    "elo",
    "split",
    "train",
    "calibrate",
    "evaluate",
    "predict",
    "api",
    "utils",
]
