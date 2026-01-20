"""
Contribution Decomposition for Marketing Mix Modeling

Decomposes sales into components:
- Baseline (organic sales without media)
- Seasonality (time-based patterns)
- Trend (long-term growth/decline)
- Media contributions (per channel)
- Events/promotions
- Residual (unexplained variance)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression


def decompose_sales(
    df: pd.DataFrame,
    date_col: str,
    sales_col: str,
    media_cols: List[str],
    model,
    include_seasonality: bool = True,
    include_trend: bool = True
) -> pd.DataFrame:
    """
    Decompose sales into baseline + media contributions + seasonality + trend
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with time series data
    date_col : str
        Column name for dates
    sales_col : str
        Column name for sales
    media_cols : List[str]
        List of media column names (adstocked/saturated)
    model : sklearn Pipeline or model
        Trained MMM model
    include_seasonality : bool
        Whether to extract seasonality component
    include_trend : bool
        Whether to extract trend component
        
    Returns:
    --------
    pd.DataFrame : Original df with additional columns:
        - baseline: Organic sales
        - seasonality: Seasonal component
        - trend: Trend component
        - {channel}_contribution: Contribution per channel
        - residual: Unexplained variance
    
    Example:
    --------
    >>> df_decomposed = decompose_sales(
    ...     df, 'Date', 'Sales', ['TV_adstock', 'Facebook_adstock'], model
    ... )
    >>> # df_decomposed has new columns: baseline, TV_contribution, etc.
    """
    df_result = df.copy()
    
    # Get model coefficients
    if hasattr(model, 'named_steps'):
        # Pipeline
        regressor = model.named_steps['regressor']
        scaler = model.named_steps['scaler']
        
        # Get scaled features
        X_scaled = scaler.transform(df[media_cols])
        coeffs = regressor.coef_
        intercept = regressor.intercept_
    else:
        # Direct model
        X_scaled = df[media_cols].values
        coeffs = model.coef_
        intercept = model.intercept_
    
    # Calculate baseline (intercept)
    df_result['baseline'] = intercept
    
    # Calculate media contributions
    for i, channel in enumerate(media_cols):
        contribution = X_scaled[:, i] * coeffs[i]
        channel_name = channel.replace('_adstock', '').replace('_saturated', '')
        df_result[f'{channel_name}_contribution'] = contribution
    
    # Extract seasonality (if requested)
    if include_seasonality:
        df_result['seasonality'] = extract_seasonality(df[sales_col].values)
    else:
        df_result['seasonality'] = 0
    
    # Extract trend (if requested)
    if include_trend:
        df_result['trend'] = extract_trend(df[sales_col].values)
    else:
        df_result['trend'] = 0
    
    # Calculate residual
    predicted = model.predict(df[media_cols])
    df_result['residual'] = df[sales_col] - predicted
    
    return df_result


def extract_seasonality(sales: np.ndarray, period: int = 52) -> np.ndarray:
    """
    Extract seasonal component using moving average
    
    Parameters:
    -----------
    sales : np.ndarray
        Sales time series
    period : int
        Seasonality period (52 for weekly data = yearly seasonality)
        
    Returns:
    --------
    np.ndarray : Seasonal component
    """
    if len(sales) < period:
        return np.zeros_like(sales)
    
    # Detrend first
    trend = extract_trend(sales)
    detrended = sales - trend
    
    # Calculate seasonal indices
    seasonal = np.zeros_like(sales)
    
    for i in range(period):
        # Average of all observations at this position in the cycle
        indices = list(range(i, len(sales), period))
        if len(indices) > 0:
            seasonal_value = np.mean(detrended[indices])
            seasonal[indices] = seasonal_value
    
    # Center around zero
    seasonal = seasonal - np.mean(seasonal)
    
    return seasonal


def extract_trend(sales: np.ndarray) -> np.ndarray:
    """
    Extract trend component using linear regression
    
    Parameters:
    -----------
    sales : np.ndarray
        Sales time series
        
    Returns:
    --------
    np.ndarray : Trend component
    """
    X = np.arange(len(sales)).reshape(-1, 1)
    y = sales
    
    model = LinearRegression()
    model.fit(X, y)
    
    trend = model.predict(X)
    
    return trend


def calculate_channel_roas(
    df: pd.DataFrame,
    spend_cols: List[str],
    contribution_cols: List[str]
) -> pd.DataFrame:
    """
    Calculate ROAS (Return on Ad Spend) for each channel
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with spend and contribution data
    spend_cols : List[str]
        Column names for media spend
    contribution_cols : List[str]
        Column names for media contributions
        
    Returns:
    --------
    pd.DataFrame : ROAS summary with columns:
        - Channel
        - Total_Spend
        - Total_Contribution
        - ROAS
        - Contribution_Pct (% of total sales)
    
    Example:
    --------
    >>> roas_df = calculate_channel_roas(
    ...     df_decomposed,
    ...     ['TV', 'Facebook'],
    ...     ['TV_contribution', 'Facebook_contribution']
    ... )
    """
    results = []
    
    for spend_col, contrib_col in zip(spend_cols, contribution_cols):
        channel_name = spend_col.replace('_adstock', '').replace('_saturated', '')
        
        total_spend = df[spend_col].sum()
        total_contribution = df[contrib_col].sum()
        
        roas = total_contribution / total_spend if total_spend > 0 else 0
        
        results.append({
            'Channel': channel_name,
            'Total_Spend': total_spend,
            'Total_Contribution': total_contribution,
            'ROAS': roas,
            'Contribution_Pct': total_contribution / df[contrib_col].sum() if df[contrib_col].sum() > 0 else 0
        })
    
    return pd.DataFrame(results)


def calculate_incremental_roas(
    df: pd.DataFrame,
    spend_cols: List[str],
    contribution_cols: List[str],
    baseline_col: str = 'baseline'
) -> pd.DataFrame:
    """
    Calculate incremental ROAS (iROAS)
    
    iROAS = (Sales with media - Sales without media) / Media Spend
    
    More accurate than ROAS because it accounts for baseline sales.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with decomposed sales
    spend_cols : List[str]
        Column names for media spend
    contribution_cols : List[str]
        Column names for media contributions
    baseline_col : str
        Column name for baseline sales
        
    Returns:
    --------
    pd.DataFrame : iROAS summary
    """
    results = []
    
    total_sales = df[contribution_cols].sum(axis=1) + df[baseline_col]
    baseline_sales = df[baseline_col]
    
    for spend_col, contrib_col in zip(spend_cols, contribution_cols):
        channel_name = spend_col.replace('_adstock', '').replace('_saturated', '')
        
        total_spend = df[spend_col].sum()
        
        # Sales with this channel
        sales_with = total_sales.sum()
        
        # Sales without this channel (counterfactual)
        sales_without = sales_with - df[contrib_col].sum()
        
        # Incremental sales
        incremental_sales = sales_with - sales_without
        
        # iROAS
        iroas = incremental_sales / total_spend if total_spend > 0 else 0
        
        results.append({
            'Channel': channel_name,
            'Total_Spend': total_spend,
            'Incremental_Sales': incremental_sales,
            'iROAS': iroas,
            'Lift_Pct': (incremental_sales / sales_without) if sales_without > 0 else 0
        })
    
    return pd.DataFrame(results)


def create_waterfall_data(
    df: pd.DataFrame,
    baseline_col: str,
    seasonality_col: str,
    trend_col: str,
    contribution_cols: List[str],
    residual_col: str
) -> Dict[str, float]:
    """
    Prepare data for waterfall chart
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with decomposed sales
    baseline_col : str
        Column name for baseline
    seasonality_col : str
        Column name for seasonality
    trend_col : str
        Column name for trend
    contribution_cols : List[str]
        Column names for media contributions
    residual_col : str
        Column name for residual
        
    Returns:
    --------
    Dict[str, float] : Waterfall data
        Keys: component names
        Values: total contribution
    
    Example:
    --------
    >>> waterfall_data = create_waterfall_data(
    ...     df_decomposed,
    ...     'baseline',
    ...     'seasonality',
    ...     'trend',
    ...     ['TV_contribution', 'Facebook_contribution'],
    ...     'residual'
    ... )
    >>> # {'Baseline': 500M, 'Seasonality': 50M, 'TV': 100M, ...}
    """
    waterfall = {}
    
    # Baseline
    waterfall['Baseline'] = df[baseline_col].sum()
    
    # Seasonality
    if seasonality_col in df.columns:
        waterfall['Seasonality'] = df[seasonality_col].sum()
    
    # Trend
    if trend_col in df.columns:
        waterfall['Trend'] = df[trend_col].sum()
    
    # Media contributions
    for col in contribution_cols:
        channel_name = col.replace('_contribution', '').replace('_', ' ').title()
        waterfall[channel_name] = df[col].sum()
    
    # Residual
    if residual_col in df.columns:
        waterfall['Unexplained'] = df[residual_col].sum()
    
    return waterfall


def calculate_contribution_over_time(
    df: pd.DataFrame,
    date_col: str,
    contribution_cols: List[str],
    freq: str = 'M'
) -> pd.DataFrame:
    """
    Aggregate contributions over time (for stacked area chart)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with decomposed sales
    date_col : str
        Column name for dates
    contribution_cols : List[str]
        Column names for media contributions
    freq : str
        Aggregation frequency ('D', 'W', 'M', 'Q', 'Y')
        
    Returns:
    --------
    pd.DataFrame : Aggregated contributions
    
    Example:
    --------
    >>> monthly_contrib = calculate_contribution_over_time(
    ...     df_decomposed,
    ...     'Date',
    ...     ['TV_contribution', 'Facebook_contribution'],
    ...     freq='M'
    ... )
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy = df_copy.set_index(date_col)
    
    # Resample and sum
    df_agg = df_copy[contribution_cols].resample(freq).sum().reset_index()
    
    return df_agg


def calculate_marginal_roas(
    model,
    current_allocation: np.ndarray,
    channel_names: List[str],
    step_size: float = 1_000_000
) -> pd.DataFrame:
    """
    Calculate marginal ROAS for each channel
    
    Marginal ROAS = Additional sales from $1 more spend
    
    Parameters:
    -----------
    model : sklearn Pipeline or model
        Trained MMM model
    current_allocation : np.ndarray
        Current budget allocation
    channel_names : List[str]
        List of channel names
    step_size : float
        Budget increment to test (default: 1M)
        
    Returns:
    --------
    pd.DataFrame : Marginal ROAS per channel
    
    Example:
    --------
    >>> marginal_roas = calculate_marginal_roas(
    ...     model,
    ...     current_allocation,
    ...     ['TV', 'Facebook', 'Instagram', 'Google']
    ... )
    """
    # Baseline prediction
    try:
        baseline_sales = model.predict(current_allocation.reshape(1, -1))[0]
    except:
        baseline_sales = model.predict([current_allocation])[0]
    
    results = []
    
    for i, channel in enumerate(channel_names):
        # Increment this channel
        new_allocation = current_allocation.copy()
        new_allocation[i] += step_size
        
        # Predict new sales
        try:
            new_sales = model.predict(new_allocation.reshape(1, -1))[0]
        except:
            new_sales = model.predict([new_allocation])[0]
        
        # Marginal sales
        marginal_sales = new_sales - baseline_sales
        
        # Marginal ROAS
        marginal_roas = marginal_sales / step_size
        
        results.append({
            'Channel': channel,
            'Current_Spend': current_allocation[i],
            'Marginal_Sales': marginal_sales,
            'Marginal_ROAS': marginal_roas,
            'Recommendation': 'Increase' if marginal_roas > 1.5 else ('Maintain' if marginal_roas > 1.0 else 'Decrease')
        })
    
    return pd.DataFrame(results).sort_values('Marginal_ROAS', ascending=False)
