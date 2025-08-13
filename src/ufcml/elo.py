"""
ELO rating system implementation for UFC ML predictor.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime, timedelta


def calculate_elo_decay(days_since_fight: int, lambda_factor: float = 0.995) -> float:
    """
    Calculate Elo decay factor based on time since last fight.
    
    Args:
        days_since_fight: Number of days since the fighter's last fight
        lambda_factor: Decay factor per 30 days (default: 0.995)
        
    Returns:
        float: Decay factor to multiply K by
        
    Example:
        >>> decay = calculate_elo_decay(90)  # 3 months
        >>> print(f"Decay factor: {decay:.4f}")
        Decay factor: 0.9851
    """
    months_since_fight = days_since_fight / 30.0
    decay_factor = lambda_factor ** months_since_fight
    return decay_factor


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score for player A against player B using Elo formula.
    
    Args:
        rating_a: Elo rating of player A
        rating_b: Elo rating of player B
        
    Returns:
        float: Expected score (probability of A winning) between 0 and 1
        
    Example:
        >>> expected = calculate_expected_score(1500, 1400)
        >>> print(f"Expected score: {expected:.3f}")
        Expected score: 0.640
    """
    rating_diff = rating_b - rating_a
    expected_score = 1 / (1 + 10 ** (rating_diff / 400))
    return expected_score


def update_elo_rating(
    current_rating: float, 
    expected_score: float, 
    actual_score: float, 
    k_factor: float
) -> float:
    """
    Update Elo rating based on match result.
    
    Args:
        current_rating: Current Elo rating
        expected_score: Expected score (0-1)
        actual_score: Actual score (0 or 1)
        k_factor: K-factor for this match
        
    Returns:
        float: New Elo rating
        
    Example:
        >>> new_rating = update_elo_rating(1500, 0.5, 1.0, 32)
        >>> print(f"New rating: {new_rating:.1f}")
        New rating: 1516.0
    """
    rating_change = k_factor * (actual_score - expected_score)
    new_rating = current_rating + rating_change
    return new_rating


def build_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build time-aware Elo ratings for UFC fighters.
    
    This function computes Elo ratings for fighters based on their fight history,
    with time decay and title bout adjustments. Each fighter starts at 1500 rating.
    
    Args:
        df: DataFrame with columns ["Date", "RedFighter", "BlueFighter", "Winner"]
            Must be sorted by Date ascending.
            
    Returns:
        pd.DataFrame: Original DataFrame with added columns:
            - RedElo: Elo rating of red fighter before the fight
            - BlueElo: Elo rating of blue fighter before the fight  
            - EloDiff: Difference in Elo ratings (RedElo - BlueElo)
            
    Raises:
        ValueError: If required columns are missing or data is not sorted by date
        
    Example:
        >>> df_with_elo = build_elo(fights_df)
        >>> print(df_with_elo[["RedFighter", "BlueFighter", "RedElo", "BlueElo", "EloDiff"]].head())
    """
    # Validate required columns
    required_columns = ["Date", "RedFighter", "BlueFighter", "Winner"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate data is sorted by date
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by Date ascending")
    
    # Initialize fighter ratings and last fight dates
    fighter_ratings: Dict[str, float] = {}
    fighter_last_fight: Dict[str, datetime] = {}
    
    # Constants
    BASE_K = 32
    TITLE_BOUT_K = 40
    LAMBDA_FACTOR = 0.995
    INITIAL_RATING = 1500
    
    # Create copy of DataFrame to avoid modifying original
    df_elo = df.copy()
    
    # Initialize new columns
    df_elo["RedElo"] = np.nan
    df_elo["BlueElo"] = np.nan
    df_elo["EloDiff"] = np.nan
    
    # Process each fight chronologically
    for idx, row in df_elo.iterrows():
        date = pd.to_datetime(row["Date"])
        red_fighter = row["RedFighter"]
        blue_fighter = row["BlueFighter"]
        winner = row["Winner"]
        
        # Initialize fighters if first time seeing them
        if red_fighter not in fighter_ratings:
            fighter_ratings[red_fighter] = INITIAL_RATING
            fighter_last_fight[red_fighter] = date
        if blue_fighter not in fighter_ratings:
            fighter_ratings[blue_fighter] = INITIAL_RATING
            fighter_last_fight[blue_fighter] = date
        
        # Calculate time decay for each fighter
        red_days_since = (date - fighter_last_fight[red_fighter]).days
        blue_days_since = (date - fighter_last_fight[blue_fighter]).days
        
        red_decay = calculate_elo_decay(red_days_since, LAMBDA_FACTOR)
        blue_decay = calculate_elo_decay(blue_days_since, LAMBDA_FACTOR)
        
        # Get current ratings
        red_rating = fighter_ratings[red_fighter]
        blue_rating = fighter_ratings[blue_fighter]
        
        # Store ratings before the fight
        df_elo.at[idx, "RedElo"] = red_rating
        df_elo.at[idx, "BlueElo"] = blue_rating
        df_elo.at[idx, "EloDiff"] = red_rating - blue_rating
        
        # Calculate expected scores
        red_expected = calculate_expected_score(red_rating, blue_rating)
        blue_expected = calculate_expected_score(blue_rating, red_rating)
        
        # Determine actual scores
        if winner == "Red":
            red_actual = 1.0
            blue_actual = 0.0
        elif winner == "Blue":
            red_actual = 0.0
            blue_actual = 1.0
        else:
            # Draw or unknown result - treat as 0.5 for both
            red_actual = 0.5
            blue_actual = 0.5
        
        # Determine K factor (check if title bout)
        is_title_bout = "TitleBout" in df.columns and row.get("TitleBout", False)
        k_factor = TITLE_BOUT_K if is_title_bout else BASE_K
        
        # Apply time decay to K factor
        red_k = k_factor * red_decay
        blue_k = k_factor * blue_decay
        
        # Update ratings
        fighter_ratings[red_fighter] = update_elo_rating(
            red_rating, red_expected, red_actual, red_k
        )
        fighter_ratings[blue_fighter] = update_elo_rating(
            blue_rating, blue_expected, blue_actual, blue_k
        )
        
        # Update last fight dates
        fighter_last_fight[red_fighter] = date
        fighter_last_fight[blue_fighter] = date
    
    return df_elo


def get_fighter_elo_summary(df_elo: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of final Elo ratings for all fighters.
    
    Args:
        df_elo: DataFrame with Elo ratings from build_elo()
        
    Returns:
        pd.DataFrame: Summary with fighter names and their final Elo ratings
    """
    # Get the last fight for each fighter to extract final ratings
    fighter_final_ratings = {}
    
    for _, row in df_elo.iterrows():
        red_fighter = row["RedFighter"]
        blue_fighter = row["BlueFighter"]
        red_elo = row["RedElo"]
        blue_elo = row["BlueElo"]
        
        # Update final ratings (last occurrence wins)
        fighter_final_ratings[red_fighter] = red_elo
        fighter_final_ratings[blue_fighter] = blue_elo
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([
        {"Fighter": fighter, "FinalElo": rating}
        for fighter, rating in fighter_final_ratings.items()
    ])
    
    # Sort by Elo rating descending
    summary_df = summary_df.sort_values("FinalElo", ascending=False).reset_index(drop=True)
    
    return summary_df


def calculate_elo_win_probability(elo_diff: float) -> float:
    """
    Calculate win probability based on Elo difference.
    
    Args:
        elo_diff: Elo difference (positive = red fighter favored)
        
    Returns:
        float: Probability of red fighter winning (0-1)
        
    Example:
        >>> prob = calculate_elo_win_probability(100)  # Red fighter +100 Elo
        >>> print(f"Red fighter win probability: {prob:.3f}")
        Red fighter win probability: 0.640
    """
    probability = 1 / (1 + 10 ** (-elo_diff / 400))
    return probability
