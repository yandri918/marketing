"""
Saturation Transformation Functions for Marketing Mix Modeling

Saturation models the diminishing returns of advertising spend.
Key insight: Doubling spend doesn't double sales.

Different saturation curves:
- Hill: S-shaped curve (most common in MMM)
- Logistic: Similar to Hill, different parameterization
- Michaelis-Menten: From biochemistry, models enzyme kinetics (and ad saturation!)
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Tuple, Union, List


def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Hill Saturation Transformation
    
    The gold standard for MMM saturation. Creates S-shaped response curve.
    Used by Google, Meta, Uber, and most modern MMM platforms.
    
    Formula: S(x) = x^α / (x^α + γ^α)
    
    Parameters:
    -----------
    x : np.ndarray
        Media spend (adstocked)
    alpha : float
        Shape parameter (controls curve steepness)
        - α < 1: Concave (diminishing returns from start)
        - α = 1: Linear (no saturation)
        - α > 1: S-shaped (slow start, then rapid growth, then saturation)
        Typical range: 0.3 - 3.0
    gamma : float
        Half-saturation point (inflection point)
        The spend level where response = 50% of maximum
        Higher γ = saturation happens at higher spend levels
        
    Returns:
    --------
    np.ndarray : Saturated values (0-1 range)
    
    Properties:
    -----------
    - Output range: [0, 1]
    - S(0) = 0
    - S(∞) = 1
    - S(γ) = 0.5 (half-saturation)
    
    Example:
    --------
    >>> spend = np.array([0, 10, 50, 100, 500])
    >>> hill_saturation(spend, alpha=1.0, gamma=50)
    array([0.0, 0.167, 0.5, 0.667, 0.909])
    # At spend=50 (gamma), response is 50%
    """
    if alpha <= 0:
        raise ValueError(f"Alpha must be positive, got {alpha}")
    if gamma <= 0:
        raise ValueError(f"Gamma must be positive, got {gamma}")
    
    # Avoid division by zero
    x = np.maximum(x, 1e-10)
    
    return x**alpha / (x**alpha + gamma**alpha)


def logistic_saturation(x: np.ndarray, k: float, x0: float, L: float = 1.0) -> np.ndarray:
    """
    Logistic Saturation Transformation
    
    Alternative to Hill saturation. Also creates S-shaped curve.
    
    Formula: S(x) = L / (1 + exp(-k * (x - x0)))
    
    Parameters:
    -----------
    x : np.ndarray
        Media spend (adstocked)
    k : float
        Steepness of the curve (growth rate)
        Higher k = steeper transition
    x0 : float
        Midpoint (inflection point)
        Spend level where response = L/2
    L : float
        Maximum value (carrying capacity)
        Default = 1.0 for normalized output
        
    Returns:
    --------
    np.ndarray : Saturated values (0-L range)
    
    Example:
    --------
    >>> spend = np.array([0, 50, 100, 150, 200])
    >>> logistic_saturation(spend, k=0.05, x0=100, L=1.0)
    # S-shaped curve centered at spend=100
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    
    return L / (1 + np.exp(-k * (x - x0)))


def michaelis_menten_saturation(x: np.ndarray, vmax: float, km: float) -> np.ndarray:
    """
    Michaelis-Menten Saturation
    
    Originally from enzyme kinetics, works great for ad saturation!
    Simpler than Hill (only 2 parameters vs 3).
    
    Formula: S(x) = (Vmax * x) / (Km + x)
    
    Parameters:
    -----------
    x : np.ndarray
        Media spend (adstocked)
    vmax : float
        Maximum response (asymptotic limit)
    km : float
        Michaelis constant (half-saturation point)
        Spend level where response = Vmax/2
        
    Returns:
    --------
    np.ndarray : Saturated values (0-vmax range)
    
    Note:
    -----
    This is equivalent to Hill saturation with α=1
    
    Example:
    --------
    >>> spend = np.array([0, 25, 50, 100, 200])
    >>> michaelis_menten_saturation(spend, vmax=1.0, km=50)
    array([0.0, 0.333, 0.5, 0.667, 0.8])
    """
    if vmax <= 0:
        raise ValueError(f"Vmax must be positive, got {vmax}")
    if km <= 0:
        raise ValueError(f"Km must be positive, got {km}")
    
    return (vmax * x) / (km + x)


def fit_hill_saturation(
    x: np.ndarray,
    y: np.ndarray,
    initial_guess: Tuple[float, float] = None
) -> Tuple[float, float, float]:
    """
    Automatically fit Hill saturation parameters to data
    
    Uses non-linear least squares to find optimal α and γ.
    
    Parameters:
    -----------
    x : np.ndarray
        Media spend (adstocked)
    y : np.ndarray
        Observed response (sales, conversions, etc.)
    initial_guess : Tuple[float, float]
        Initial guess for (alpha, gamma)
        If None, uses smart defaults based on data
        
    Returns:
    --------
    Tuple[float, float, float] : (alpha, gamma, r_squared)
        - alpha: Fitted shape parameter
        - gamma: Fitted half-saturation point
        - r_squared: Goodness of fit (0-1)
        
    Example:
    --------
    >>> spend = np.array([10, 50, 100, 200, 500])
    >>> sales = np.array([100, 300, 450, 550, 600])
    >>> alpha, gamma, r2 = fit_hill_saturation(spend, sales)
    >>> print(f"Optimal α={alpha:.2f}, γ={gamma:.2f}, R²={r2:.2%}")
    """
    # Normalize y to [0, 1] for fitting
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else y
    
    # Smart initial guess if not provided
    if initial_guess is None:
        # Gamma: approximate as median of x
        gamma_init = np.median(x)
        # Alpha: start with 1.0 (linear-ish)
        alpha_init = 1.0
        initial_guess = (alpha_init, gamma_init)
    
    try:
        # Fit using scipy's curve_fit
        popt, _ = curve_fit(
            hill_saturation,
            x,
            y_norm,
            p0=initial_guess,
            bounds=([0.1, x.min()], [5.0, x.max() * 2]),  # Reasonable bounds
            maxfev=5000
        )
        
        alpha_fit, gamma_fit = popt
        
        # Calculate R²
        y_pred = hill_saturation(x, alpha_fit, gamma_fit)
        ss_res = np.sum((y_norm - y_pred)**2)
        ss_tot = np.sum((y_norm - y_norm.mean())**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return alpha_fit, gamma_fit, r_squared
        
    except Exception as e:
        # If fitting fails, return defaults
        print(f"Warning: Saturation fitting failed ({e}). Using defaults.")
        return 1.0, np.median(x), 0.0


def apply_saturation_to_dataframe(
    df: pd.DataFrame,
    channels: List[str],
    saturation_type: str = 'hill',
    params: dict = None,
    auto_fit: bool = False,
    target_col: str = None
) -> pd.DataFrame:
    """
    Apply saturation transformation to multiple channels
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with media spend columns (preferably adstocked)
    channels : List[str]
        List of column names to transform
    saturation_type : str
        Type: 'hill', 'logistic', 'michaelis_menten'
    params : dict
        Parameters for each channel. Format:
        {
            'channel_name': {'alpha': 1.0, 'gamma': 50.0},
            ...
        }
    auto_fit : bool
        If True, automatically fit saturation parameters using target_col
    target_col : str
        Target column for auto-fitting (e.g., 'Sales')
        Required if auto_fit=True
        
    Returns:
    --------
    pd.DataFrame : Original df with new saturated columns (suffix: _saturated)
    
    Example:
    --------
    >>> # Manual parameters
    >>> params = {
    ...     'TV_adstock': {'alpha': 0.8, 'gamma': 50_000_000},
    ...     'Facebook_adstock': {'alpha': 1.2, 'gamma': 30_000_000}
    ... }
    >>> df_sat = apply_saturation_to_dataframe(df, channels, 'hill', params)
    >>>
    >>> # Auto-fit
    >>> df_sat = apply_saturation_to_dataframe(
    ...     df, channels, 'hill', auto_fit=True, target_col='Sales'
    ... )
    """
    df_result = df.copy()
    
    if params is None:
        params = {}
    
    if auto_fit and target_col is None:
        raise ValueError("target_col required when auto_fit=True")
    
    for channel in channels:
        if channel not in df.columns:
            raise ValueError(f"Channel '{channel}' not found in DataFrame")
        
        x = df[channel].values
        
        # Auto-fit if requested
        if auto_fit:
            y = df[target_col].values
            if saturation_type == 'hill':
                alpha, gamma, r2 = fit_hill_saturation(x, y)
                params[channel] = {'alpha': alpha, 'gamma': gamma}
                print(f"Auto-fitted {channel}: α={alpha:.3f}, γ={gamma:,.0f}, R²={r2:.2%}")
        
        channel_params = params.get(channel, {})
        
        # Apply saturation
        if saturation_type == 'hill':
            alpha = channel_params.get('alpha', 1.0)
            gamma = channel_params.get('gamma', np.median(x))
            saturated = hill_saturation(x, alpha, gamma)
            
        elif saturation_type == 'logistic':
            k = channel_params.get('k', 0.01)
            x0 = channel_params.get('x0', np.median(x))
            L = channel_params.get('L', 1.0)
            saturated = logistic_saturation(x, k, x0, L)
            
        elif saturation_type == 'michaelis_menten':
            vmax = channel_params.get('vmax', 1.0)
            km = channel_params.get('km', np.median(x))
            saturated = michaelis_menten_saturation(x, vmax, km)
            
        else:
            raise ValueError(f"Unknown saturation type: {saturation_type}")
        
        df_result[f"{channel}_saturated"] = saturated
    
    return df_result


def get_saturation_curve(
    saturation_type: str,
    params: dict,
    x_range: Tuple[float, float] = (0, 100),
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate saturation curve for visualization
    
    Parameters:
    -----------
    saturation_type : str
        Type of saturation
    params : dict
        Parameters for the saturation function
    x_range : Tuple[float, float]
        Range of x values to plot (min, max)
    n_points : int
        Number of points in the curve
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : (x_values, y_values)
    
    Example:
    --------
    >>> x, y = get_saturation_curve('hill', {'alpha': 1.0, 'gamma': 50}, (0, 200))
    >>> # Plot: plt.plot(x, y)
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    if saturation_type == 'hill':
        y = hill_saturation(x, params['alpha'], params['gamma'])
    elif saturation_type == 'logistic':
        y = logistic_saturation(x, params['k'], params['x0'], params.get('L', 1.0))
    elif saturation_type == 'michaelis_menten':
        y = michaelis_menten_saturation(x, params['vmax'], params['km'])
    else:
        raise ValueError(f"Unknown saturation type: {saturation_type}")
    
    return x, y


def calculate_optimal_spend(
    saturation_type: str,
    params: dict,
    target_efficiency: float = 0.7
) -> float:
    """
    Calculate optimal spend point based on efficiency threshold
    
    "Optimal" = point where marginal ROI drops below threshold
    
    Parameters:
    -----------
    saturation_type : str
        Type of saturation
    params : dict
        Saturation parameters
    target_efficiency : float
        Efficiency threshold (0-1)
        0.7 = stop when marginal return is 70% of initial
        
    Returns:
    --------
    float : Optimal spend level
    
    Example:
    --------
    >>> optimal = calculate_optimal_spend('hill', {'alpha': 1.0, 'gamma': 50}, 0.7)
    >>> print(f"Optimal spend: {optimal:.0f}")
    """
    # For Hill saturation, we can calculate analytically
    if saturation_type == 'hill':
        alpha = params['alpha']
        gamma = params['gamma']
        
        # Derivative of Hill function
        # S'(x) = (α * γ^α * x^(α-1)) / (x^α + γ^α)^2
        # Find x where S'(x) / S'(0) = target_efficiency
        
        # Simplified: use gamma as proxy for optimal point
        # (at gamma, we're at 50% saturation with good marginal return)
        return gamma * (target_efficiency ** (1 / alpha))
    
    else:
        # For other types, use numerical search
        x_test = np.linspace(0, params.get('gamma', 100) * 3, 1000)
        
        if saturation_type == 'logistic':
            y = logistic_saturation(x_test, params['k'], params['x0'], params.get('L', 1.0))
        elif saturation_type == 'michaelis_menten':
            y = michaelis_menten_saturation(x_test, params['vmax'], params['km'])
        else:
            return 0.0
        
        # Calculate marginal return
        dy = np.diff(y) / np.diff(x_test)
        
        # Find where marginal return drops below threshold
        initial_dy = dy[0] if len(dy) > 0 else 1.0
        threshold_dy = initial_dy * target_efficiency
        
        idx = np.where(dy < threshold_dy)[0]
        if len(idx) > 0:
            return x_test[idx[0]]
        else:
            return x_test[-1]
