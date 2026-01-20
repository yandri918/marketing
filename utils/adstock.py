"""
Advanced Adstock Transformation Functions for Marketing Mix Modeling

Adstock represents the carryover effect of advertising - how ads seen today 
continue to influence sales in future periods.

Different adstock types:
- Geometric: Exponential decay (traditional, fast decay)
- Weibull: Flexible decay shape (can model delayed peak effects)
- Delayed: Ads take time to impact (e.g., brand campaigns)
"""

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_func
from typing import Union, List


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Geometric (Exponential) Adstock Transformation
    
    Classic adstock with exponential decay. Simple but effective.
    
    Formula: adstock[t] = x[t] + decay * adstock[t-1]
    
    Parameters:
    -----------
    x : np.ndarray
        Media spend time series
    decay : float
        Decay rate (0-1). Higher = longer carryover effect
        - 0.0 = no carryover (immediate effect only)
        - 0.5 = moderate carryover (social media)
        - 0.8 = strong carryover (TV, brand campaigns)
        
    Returns:
    --------
    np.ndarray : Adstocked media values
    
    Example:
    --------
    >>> spend = np.array([100, 0, 0, 0])
    >>> geometric_adstock(spend, decay=0.5)
    array([100.0, 50.0, 25.0, 12.5])  # Exponential decay
    """
    if not 0 <= decay < 1:
        raise ValueError(f"Decay must be in [0, 1), got {decay}")
    
    adstocked = np.zeros_like(x, dtype=float)
    adstocked[0] = x[0]
    
    for t in range(1, len(x)):
        adstocked[t] = x[t] + decay * adstocked[t-1]
    
    return adstocked


def weibull_adstock(x: np.ndarray, shape: float, scale: float, peak_delay: int = 0) -> np.ndarray:
    """
    Weibull Adstock Transformation
    
    More flexible than geometric - can model delayed peak effects.
    Used by Google's Meridian and Meta's Robyn.
    
    The Weibull distribution allows for:
    - shape < 1: Fast initial decay (social media, search)
    - shape = 1: Geometric decay (equivalent to exponential)
    - shape > 1: Delayed peak (TV, brand campaigns, billboards)
    
    Parameters:
    -----------
    x : np.ndarray
        Media spend time series
    shape : float
        Shape parameter (k in Weibull distribution)
        - < 1: Decreasing hazard (fast initial decay)
        - = 1: Constant hazard (geometric)
        - > 1: Increasing hazard (delayed peak)
    scale : float
        Scale parameter (λ in Weibull distribution)
        Controls how quickly the effect decays
        Higher = longer lasting effect
    peak_delay : int
        Number of periods to delay the peak effect (default=0)
        
    Returns:
    --------
    np.ndarray : Adstocked media values
    
    Example:
    --------
    >>> spend = np.array([100, 0, 0, 0, 0, 0])
    >>> # Delayed peak (TV campaign)
    >>> weibull_adstock(spend, shape=2.0, scale=3.0)
    # Effect peaks at t=2-3, then decays
    """
    if shape <= 0 or scale <= 0:
        raise ValueError(f"Shape and scale must be positive, got shape={shape}, scale={scale}")
    
    n = len(x)
    adstocked = np.zeros(n, dtype=float)
    
    # Weibull PDF for decay weights
    def weibull_pdf(t, k, lam):
        """Weibull probability density function"""
        if t <= 0:
            return 0
        return (k / lam) * (t / lam)**(k - 1) * np.exp(-(t / lam)**k)
    
    # Create convolution kernel (decay weights)
    max_lag = min(n, int(scale * 5))  # Truncate at 5*scale for efficiency
    kernel = np.array([weibull_pdf(t + peak_delay, shape, scale) for t in range(max_lag)])
    
    # Normalize kernel to preserve total effect
    if kernel.sum() > 0:
        kernel = kernel / kernel.sum()
    
    # Apply convolution (adstock transformation)
    for t in range(n):
        for lag in range(min(t + 1, max_lag)):
            adstocked[t] += x[t - lag] * kernel[lag]
    
    return adstocked


def delayed_adstock(x: np.ndarray, theta: float, L: int) -> np.ndarray:
    """
    Delayed Adstock (Adstock with Lag)
    
    Models scenarios where advertising takes time to impact sales.
    Example: Brand awareness campaigns, PR, content marketing.
    
    Parameters:
    -----------
    x : np.ndarray
        Media spend time series
    theta : float
        Retention rate (0-1). Similar to decay in geometric adstock.
    L : int
        Maximum lag length (how many periods the effect lasts)
        
    Returns:
    --------
    np.ndarray : Adstocked media values
    
    Example:
    --------
    >>> spend = np.array([100, 0, 0, 0, 0])
    >>> delayed_adstock(spend, theta=0.7, L=4)
    # Effect distributed over 4 periods with 0.7 retention
    """
    if not 0 <= theta < 1:
        raise ValueError(f"Theta must be in [0, 1), got {theta}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    
    n = len(x)
    adstocked = np.zeros(n, dtype=float)
    
    # Create decay weights
    weights = np.array([theta**i for i in range(L)])
    weights = weights / weights.sum()  # Normalize
    
    # Apply convolution
    for t in range(n):
        for lag in range(min(t + 1, L)):
            adstocked[t] += x[t - lag] * weights[lag]
    
    return adstocked


def carryover_adstock(x: np.ndarray, peak: int, decay: float, concentration: float = 1.0) -> np.ndarray:
    """
    Carryover Adstock with Peak Delay
    
    Combines delayed peak with decay. Most flexible option.
    Used in advanced MMM implementations.
    
    Parameters:
    -----------
    x : np.ndarray
        Media spend time series
    peak : int
        Period when effect peaks (0 = immediate, 1 = next period, etc.)
    decay : float
        Decay rate after peak (0-1)
    concentration : float
        How concentrated the effect is around the peak
        Higher = more concentrated (sharper peak)
        Lower = more distributed (smoother curve)
        
    Returns:
    --------
    np.ndarray : Adstocked media values
    """
    if not 0 <= decay < 1:
        raise ValueError(f"Decay must be in [0, 1), got {decay}")
    if peak < 0:
        raise ValueError(f"Peak must be >= 0, got {peak}")
    if concentration <= 0:
        raise ValueError(f"Concentration must be > 0, got {concentration}")
    
    n = len(x)
    max_lag = min(n, peak + int(20 / (1 - decay)))  # Adaptive max lag
    
    # Create asymmetric kernel
    kernel = np.zeros(max_lag)
    for t in range(max_lag):
        if t <= peak:
            # Build-up phase (before peak)
            kernel[t] = (t / peak)**concentration if peak > 0 else 1.0
        else:
            # Decay phase (after peak)
            kernel[t] = kernel[peak] * (decay**(t - peak))
    
    # Normalize
    if kernel.sum() > 0:
        kernel = kernel / kernel.sum()
    
    # Apply convolution
    adstocked = np.zeros(n, dtype=float)
    for t in range(n):
        for lag in range(min(t + 1, max_lag)):
            adstocked[t] += x[t - lag] * kernel[lag]
    
    return adstocked


def apply_adstock_to_dataframe(
    df: pd.DataFrame,
    channels: List[str],
    adstock_type: str = 'geometric',
    params: dict = None
) -> pd.DataFrame:
    """
    Apply adstock transformation to multiple channels in a DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with media spend columns
    channels : List[str]
        List of column names to transform
    adstock_type : str
        Type of adstock: 'geometric', 'weibull', 'delayed', 'carryover'
    params : dict
        Parameters for each channel. Format:
        {
            'channel_name': {'param1': value1, 'param2': value2},
            ...
        }
        If None, uses default parameters for all channels.
        
    Returns:
    --------
    pd.DataFrame : Original df with new adstocked columns (suffix: _adstock)
    
    Example:
    --------
    >>> params = {
    ...     'TV': {'decay': 0.8},
    ...     'Facebook': {'decay': 0.4}
    ... }
    >>> df_adstocked = apply_adstock_to_dataframe(df, ['TV', 'Facebook'], 'geometric', params)
    """
    df_result = df.copy()
    
    if params is None:
        params = {}
    
    for channel in channels:
        if channel not in df.columns:
            raise ValueError(f"Channel '{channel}' not found in DataFrame")
        
        x = df[channel].values
        channel_params = params.get(channel, {})
        
        # Apply appropriate adstock function
        if adstock_type == 'geometric':
            decay = channel_params.get('decay', 0.5)
            adstocked = geometric_adstock(x, decay)
            
        elif adstock_type == 'weibull':
            shape = channel_params.get('shape', 1.0)
            scale = channel_params.get('scale', 2.0)
            peak_delay = channel_params.get('peak_delay', 0)
            adstocked = weibull_adstock(x, shape, scale, peak_delay)
            
        elif adstock_type == 'delayed':
            theta = channel_params.get('theta', 0.7)
            L = channel_params.get('L', 4)
            adstocked = delayed_adstock(x, theta, L)
            
        elif adstock_type == 'carryover':
            peak = channel_params.get('peak', 0)
            decay = channel_params.get('decay', 0.5)
            concentration = channel_params.get('concentration', 1.0)
            adstocked = carryover_adstock(x, peak, decay, concentration)
            
        else:
            raise ValueError(f"Unknown adstock type: {adstock_type}")
        
        df_result[f"{channel}_adstock"] = adstocked
    
    return df_result


def get_adstock_curve(adstock_type: str, params: dict, periods: int = 20) -> np.ndarray:
    """
    Generate adstock decay curve for visualization
    
    Useful for showing how a single unit of spend decays over time.
    
    Parameters:
    -----------
    adstock_type : str
        Type of adstock
    params : dict
        Parameters for the adstock function
    periods : int
        Number of periods to simulate
        
    Returns:
    --------
    np.ndarray : Decay curve (effect over time from single unit spend)
    
    Example:
    --------
    >>> curve = get_adstock_curve('geometric', {'decay': 0.7}, periods=10)
    >>> # curve[0] = 1.0, curve[1] = 0.7, curve[2] = 0.49, ...
    """
    # Create impulse (single unit at t=0)
    impulse = np.zeros(periods)
    impulse[0] = 1.0
    
    # Apply adstock
    if adstock_type == 'geometric':
        return geometric_adstock(impulse, params['decay'])
    elif adstock_type == 'weibull':
        return weibull_adstock(impulse, params['shape'], params['scale'], params.get('peak_delay', 0))
    elif adstock_type == 'delayed':
        return delayed_adstock(impulse, params['theta'], params['L'])
    elif adstock_type == 'carryover':
        return carryover_adstock(impulse, params['peak'], params['decay'], params.get('concentration', 1.0))
    else:
        raise ValueError(f"Unknown adstock type: {adstock_type}")
