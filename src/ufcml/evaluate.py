"""
Model evaluation and analysis for UFC ML predictor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from pathlib import Path
import json


def compute_metrics(y_true: np.ndarray, 
                   p: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels (0 or 1)
        p: Predicted probabilities for positive class
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Basic classification metrics
    try:
        metrics['logloss'] = log_loss(y_true, p)
    except ValueError:
        metrics['logloss'] = np.nan
        
    try:
        metrics['brier'] = brier_score_loss(y_true, p)
    except ValueError:
        metrics['brier'] = np.nan
        
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, p)
    except ValueError:
        metrics['roc_auc'] = 0.5
        
    # Additional metrics
    y_pred = (p > 0.5).astype(int)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['red_win_rate'] = y_true.mean()
    metrics['pred_red_win_rate'] = p.mean()
    
    return metrics


def plot_reliability(y_true: np.ndarray, 
                    p: np.ndarray, 
                    out_path: str,
                    title: str = "Reliability Plot",
                    n_bins: int = 10) -> None:
    """
    Create and save reliability plot (calibration curve).
    
    Args:
        y_true: True labels
        p: Predicted probabilities
        out_path: Path to save the plot
        title: Plot title
        n_bins: Number of probability bins
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(p, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate reliability data
    reliability_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_probs = p[mask]
            bin_true = y_true[mask]
            bin_mean_prob = bin_probs.mean()
            bin_true_rate = bin_true.mean()
            bin_count = mask.sum()
            
            reliability_data.append({
                'mean_prob': bin_mean_prob,
                'true_rate': bin_true_rate,
                'count': bin_count
            })
    
    if not reliability_data:
        print("Warning: No reliability data to plot")
        return
    
    reliability_df = pd.DataFrame(reliability_data)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Plot reliability curve
    plt.plot(reliability_df['mean_prob'], reliability_df['true_rate'], 
             'o-', linewidth=2, markersize=8, label='Model')
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, label='Perfect Calibration')
    
    # Plot histogram of predictions
    ax2 = plt.twinx()
    ax2.hist(p, bins=n_bins, alpha=0.3, color='lightblue', edgecolor='black')
    ax2.set_ylabel('Count', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Customize plot
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Fraction of Positives', color='red')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add metrics text
    metrics = compute_metrics(y_true, p)
    textstr = f'Log Loss: {metrics["logloss"]:.4f}\nBrier: {metrics["brier"]:.4f}\nROC AUC: {metrics["roc_auc"]:.4f}'
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Reliability plot saved to: {out_path}")


def bucket_by_odds(meta: pd.DataFrame, 
                   p: np.ndarray, 
                   bins: List[float] = None) -> pd.DataFrame:
    """
    Create betting analysis by bucketing predictions into confidence intervals.
    
    Args:
        meta: Metadata DataFrame with fight information
        p: Predicted probabilities
        bins: Probability thresholds for bucketing
        
    Returns:
        DataFrame with betting analysis
    """
    if bins is None:
        bins = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    # Create betting buckets
    betting_data = []
    
    for i in range(len(bins) - 1):
        lower, upper = bins[i], bins[i + 1]
        mask = (p >= lower) & (p < upper)
        
        if mask.sum() > 0:
            bucket_probs = p[mask]
            bucket_meta = meta.iloc[mask]
            
            # Calculate betting metrics
            avg_prob = bucket_probs.mean()
            count = mask.sum()
            
            betting_data.append({
                'prob_range': f'{lower:.2f}-{upper:.2f}',
                'lower_prob': lower,
                'upper_prob': upper,
                'avg_prob': avg_prob,
                'count': count,
                'fraction_of_total': count / len(p)
            })
    
    return pd.DataFrame(betting_data)


def create_evaluation_report(y_true: np.ndarray, 
                           p: np.ndarray, 
                           meta: pd.DataFrame,
                           out_path: str) -> Dict[str, any]:
    """
    Create comprehensive evaluation report.
    
    Args:
        y_true: True labels
        p: Predicted probabilities
        meta: Metadata DataFrame
        out_path: Directory to save report
        
    Returns:
        Dictionary with evaluation results
    """
    # Compute metrics
    metrics = compute_metrics(y_true, p)
    
    # Create reliability plot
    plot_path = Path(out_path) / "reliability_plot.png"
    plot_reliability(y_true, p, str(plot_path), "Model Calibration")
    
    # Create betting analysis
    betting_df = bucket_by_odds(meta, p)
    
    # Save betting analysis
    betting_path = Path(out_path) / "betting_analysis.csv"
    betting_df.to_csv(betting_path, index=False)
    
    # Create comprehensive report
    report = {
        'metrics': metrics,
        'betting_analysis': betting_df.to_dict('records'),
        'plot_path': str(plot_path),
        'betting_path': str(betting_path),
        'n_samples': len(y_true),
        'probability_stats': {
            'mean': float(p.mean()),
            'std': float(p.std()),
            'min': float(p.min()),
            'max': float(p.max()),
            'q25': float(np.percentile(p, 25)),
            'q50': float(np.percentile(p, 50)),
            'q75': float(np.percentile(p, 75))
        }
    }
    
    # Save report
    report_path = Path(out_path) / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Evaluation report saved to: {report_path}")
    return report


def compare_models_side_by_side(y_true: np.ndarray,
                               probs_dict: Dict[str, np.ndarray],
                               out_path: str,
                               title: str = "Model Comparison") -> None:
    """
    Create side-by-side comparison of multiple models.
    
    Args:
        y_true: True labels
        probs_dict: Dictionary of {model_name: probabilities}
        out_path: Path to save comparison plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (model_name, probs) in enumerate(probs_dict.items()):
        color = colors[i % len(colors)]
        
        # Calculate reliability data
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        reliability_data = []
        for j in range(n_bins):
            mask = bin_indices == j
            if mask.sum() > 0:
                bin_probs = probs[mask]
                bin_true = y_true[mask]
                bin_mean_prob = bin_probs.mean()
                bin_true_rate = bin_true.mean()
                reliability_data.append([bin_mean_prob, bin_true_rate])
        
        if reliability_data:
            reliability_array = np.array(reliability_data)
            plt.plot(reliability_array[:, 0], reliability_array[:, 1], 
                     'o-', linewidth=2, markersize=6, label=model_name, color=color)
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, label='Perfect Calibration')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Fraction of Positives')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison plot saved to: {out_path}")
