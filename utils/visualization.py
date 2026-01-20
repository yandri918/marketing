"""
Advanced Visualization Functions for Marketing Mix Modeling

Provides publication-quality charts for:
- Adstock decay curves
- Saturation curves with optimal points
- Contribution waterfall charts
- Pareto frontier (multi-objective optimization)
- SHAP value plots
- Time-series decomposition
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional


def plot_adstock_curves(
    adstock_configs: Dict[str, Dict],
    periods: int = 20,
    title: str = "Adstock Decay Curves by Channel"
) -> go.Figure:
    """
    Visualize how different channels' effects decay over time
    
    Parameters:
    -----------
    adstock_configs : Dict[str, Dict]
        Configuration for each channel. Format:
        {
            'TV': {'type': 'geometric', 'params': {'decay': 0.8}},
            'Facebook': {'type': 'weibull', 'params': {'shape': 2.0, 'scale': 3.0}},
            ...
        }
    periods : int
        Number of periods to show
    title : str
        Chart title
        
    Returns:
    --------
    go.Figure : Plotly figure
    
    Example:
    --------
    >>> configs = {
    ...     'TV': {'type': 'geometric', 'params': {'decay': 0.8}},
    ...     'Social': {'type': 'geometric', 'params': {'decay': 0.3}}
    ... }
    >>> fig = plot_adstock_curves(configs)
    >>> fig.show()
    """
    from .adstock import get_adstock_curve
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for i, (channel, config) in enumerate(adstock_configs.items()):
        adstock_type = config['type']
        params = config['params']
        
        curve = get_adstock_curve(adstock_type, params, periods)
        
        fig.add_trace(go.Scatter(
            x=list(range(periods)),
            y=curve,
            mode='lines+markers',
            name=channel,
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=6),
            hovertemplate=f'<b>{channel}</b><br>Period: %{{x}}<br>Effect: %{{y:.2%}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        xaxis_title="Periods After Ad Exposure",
        yaxis_title="Remaining Effect (%)",
        yaxis=dict(tickformat='.0%'),
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add reference line at 50%
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="50% decay", annotation_position="right")
    
    return fig


def plot_saturation_curves(
    saturation_configs: Dict[str, Dict],
    x_range: Tuple[float, float] = (0, 100),
    show_optimal: bool = True,
    title: str = "Saturation Curves by Channel"
) -> go.Figure:
    """
    Visualize diminishing returns (saturation) for each channel
    
    Parameters:
    -----------
    saturation_configs : Dict[str, Dict]
        Configuration for each channel. Format:
        {
            'TV': {'type': 'hill', 'params': {'alpha': 0.8, 'gamma': 50}},
            ...
        }
    x_range : Tuple[float, float]
        Range of spend to visualize (min, max)
    show_optimal : bool
        If True, mark optimal spend points
    title : str
        Chart title
        
    Returns:
    --------
    go.Figure : Plotly figure
    """
    from .saturation import get_saturation_curve, calculate_optimal_spend
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, (channel, config) in enumerate(saturation_configs.items()):
        sat_type = config['type']
        params = config['params']
        
        x, y = get_saturation_curve(sat_type, params, x_range, n_points=200)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=channel,
            line=dict(width=3, color=colors[i % len(colors)]),
            hovertemplate=f'<b>{channel}</b><br>Spend: %{{x:,.0f}}<br>Response: %{{y:.1%}}<extra></extra>'
        ))
        
        # Mark optimal point
        if show_optimal:
            optimal_x = calculate_optimal_spend(sat_type, params, target_efficiency=0.7)
            if sat_type == 'hill':
                from .saturation import hill_saturation
                optimal_y = hill_saturation(np.array([optimal_x]), params['alpha'], params['gamma'])[0]
            else:
                # Approximate for other types
                optimal_y = 0.5
            
            fig.add_trace(go.Scatter(
                x=[optimal_x],
                y=[optimal_y],
                mode='markers',
                marker=dict(size=12, color=colors[i % len(colors)], symbol='star'),
                name=f'{channel} Optimal',
                showlegend=False,
                hovertemplate=f'<b>{channel} Optimal</b><br>Spend: {optimal_x:,.0f}<br>Response: {optimal_y:.1%}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        xaxis_title="Media Spend",
        yaxis_title="Response (Normalized)",
        yaxis=dict(tickformat='.0%'),
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_contribution_waterfall(
    contributions: Dict[str, float],
    baseline: float,
    actual_sales: float,
    title: str = "Sales Contribution Decomposition"
) -> go.Figure:
    """
    Create waterfall chart showing how each factor contributes to sales
    
    Parameters:
    -----------
    contributions : Dict[str, float]
        Contribution of each factor. Format:
        {
            'Baseline': 500_000_000,
            'Seasonality': 50_000_000,
            'TV': 100_000_000,
            'Facebook': 80_000_000,
            ...
        }
    baseline : float
        Baseline sales (organic)
    actual_sales : float
        Actual observed sales
    title : str
        Chart title
        
    Returns:
    --------
    go.Figure : Plotly waterfall chart
    
    Example:
    --------
    >>> contributions = {
    ...     'Baseline': 500_000_000,
    ...     'TV': 100_000_000,
    ...     'Facebook': 50_000_000
    ... }
    >>> fig = plot_contribution_waterfall(contributions, 500_000_000, 650_000_000)
    """
    # Prepare data
    labels = ['Baseline'] + list(contributions.keys()) + ['Total']
    values = [baseline] + list(contributions.values()) + [actual_sales - sum(contributions.values()) - baseline]
    
    # Create measure types
    measures = ['absolute'] + ['relative'] * len(contributions) + ['total']
    
    # Colors
    colors = ['#3498db'] + ['#2ecc71'] * len(contributions) + ['#e74c3c']
    
    fig = go.Figure(go.Waterfall(
        name="Sales",
        orientation="v",
        measure=measures,
        x=labels,
        textposition="outside",
        text=[f"Rp {v/1e9:.1f}B" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        increasing={"marker": {"color": "#2ecc71"}},
        totals={"marker": {"color": "#3498db"}}
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        yaxis_title="Sales (Rp)",
        showlegend=False,
        template='plotly_white',
        height=600
    )
    
    return fig


def plot_actual_vs_predicted(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    predicted_col: str,
    credible_interval: Optional[Tuple[str, str]] = None,
    title: str = "Actual vs Predicted Sales"
) -> go.Figure:
    """
    Plot actual vs predicted sales with optional credible intervals
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with time series data
    date_col : str
        Column name for dates
    actual_col : str
        Column name for actual values
    predicted_col : str
        Column name for predicted values
    credible_interval : Optional[Tuple[str, str]]
        Column names for (lower_bound, upper_bound) of credible interval
        For Bayesian models only
    title : str
        Chart title
        
    Returns:
    --------
    go.Figure : Plotly figure
    """
    fig = go.Figure()
    
    # Actual sales
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[actual_col],
        mode='lines',
        name='Actual Sales',
        line=dict(color='#34495e', width=2),
        hovertemplate='<b>Actual</b><br>Date: %{x}<br>Sales: Rp %{y:,.0f}<extra></extra>'
    ))
    
    # Predicted sales
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[predicted_col],
        mode='lines',
        name='Predicted Sales',
        line=dict(color='#e74c3c', width=2, dash='dot'),
        hovertemplate='<b>Predicted</b><br>Date: %{x}<br>Sales: Rp %{y:,.0f}<extra></extra>'
    ))
    
    # Credible interval (if Bayesian)
    if credible_interval:
        lower_col, upper_col = credible_interval
        
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[upper_col],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[lower_col],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(231, 76, 60, 0.2)',
            fill='tonexty',
            name='95% Credible Interval',
            hovertemplate='<b>95% CI</b><br>Lower: Rp %{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        xaxis_title="Date",
        yaxis_title="Sales (Rp)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_channel_contributions_over_time(
    df: pd.DataFrame,
    date_col: str,
    channel_cols: List[str],
    title: str = "Channel Contributions Over Time"
) -> go.Figure:
    """
    Stacked area chart showing how each channel contributes over time
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with contribution data
    date_col : str
        Column name for dates
    channel_cols : List[str]
        List of column names for channel contributions
    title : str
        Chart title
        
    Returns:
    --------
    go.Figure : Plotly stacked area chart
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Pastel
    
    for i, channel in enumerate(channel_cols):
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[channel],
            mode='lines',
            name=channel.replace('_contribution', '').replace('_', ' ').title(),
            stackgroup='one',
            fillcolor=colors[i % len(colors)],
            line=dict(width=0.5, color=colors[i % len(colors)]),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Contribution: Rp %{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        xaxis_title="Date",
        yaxis_title="Contribution to Sales (Rp)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_pareto_frontier(
    solutions: pd.DataFrame,
    obj1_col: str,
    obj2_col: str,
    obj1_name: str = "Objective 1",
    obj2_name: str = "Objective 2",
    selected_idx: Optional[int] = None,
    title: str = "Pareto Frontier - Trade-off Analysis"
) -> go.Figure:
    """
    Visualize Pareto frontier for multi-objective optimization
    
    Parameters:
    -----------
    solutions : pd.DataFrame
        DataFrame with Pareto-optimal solutions
    obj1_col : str
        Column name for first objective
    obj2_col : str
        Column name for second objective
    obj1_name : str
        Display name for first objective
    obj2_name : str
        Display name for second objective
    selected_idx : Optional[int]
        Index of selected solution to highlight
    title : str
        Chart title
        
    Returns:
    --------
    go.Figure : Plotly scatter plot
    """
    fig = go.Figure()
    
    # Pareto frontier
    fig.add_trace(go.Scatter(
        x=solutions[obj1_col],
        y=solutions[obj2_col],
        mode='markers+lines',
        name='Pareto Frontier',
        marker=dict(size=10, color='#3498db', line=dict(width=2, color='white')),
        line=dict(color='#3498db', width=2, dash='dot'),
        hovertemplate=f'<b>Solution</b><br>{obj1_name}: %{{x:,.0f}}<br>{obj2_name}: %{{y:.2%}}<extra></extra>'
    ))
    
    # Highlight selected solution
    if selected_idx is not None:
        fig.add_trace(go.Scatter(
            x=[solutions.iloc[selected_idx][obj1_col]],
            y=[solutions.iloc[selected_idx][obj2_col]],
            mode='markers',
            name='Selected Solution',
            marker=dict(size=20, color='#e74c3c', symbol='star', line=dict(width=2, color='white')),
            hovertemplate=f'<b>SELECTED</b><br>{obj1_name}: %{{x:,.0f}}<br>{obj2_name}: %{{y:.2%}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        xaxis_title=obj1_name,
        yaxis_title=obj2_name,
        yaxis=dict(tickformat='.1%'),
        template='plotly_white',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_tornado_chart(
    sensitivities: Dict[str, float],
    baseline: float,
    title: str = "Sensitivity Analysis - Tornado Chart"
) -> go.Figure:
    """
    Tornado chart for sensitivity analysis
    
    Shows which variables have the biggest impact on the outcome.
    
    Parameters:
    -----------
    sensitivities : Dict[str, float]
        Impact of each variable. Format:
        {
            'TV Budget +10%': 5_000_000,
            'TV Budget -10%': -4_500_000,
            'Facebook Budget +10%': 3_000_000,
            ...
        }
    baseline : float
        Baseline outcome value
    title : str
        Chart title
        
    Returns:
    --------
    go.Figure : Plotly tornado chart
    """
    # Sort by absolute impact
    sorted_items = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
    
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(color=colors),
        text=[f"Rp {v/1e9:+.1f}B" for v in values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Impact: Rp %{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        xaxis_title="Impact on Sales (Rp)",
        yaxis_title="Variable",
        template='plotly_white',
        height=max(400, len(labels) * 40),
        showlegend=False
    )
    
    # Add baseline reference line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def plot_coefficient_importance(
    coefficients: Dict[str, float],
    title: str = "Channel Coefficients (Impact per Unit Spend)"
) -> go.Figure:
    """
    Bar chart showing model coefficients (feature importance)
    
    Parameters:
    -----------
    coefficients : Dict[str, float]
        Coefficients for each channel
    title : str
        Chart title
        
    Returns:
    --------
    go.Figure : Plotly bar chart
    """
    # Sort by value
    sorted_items = sorted(coefficients.items(), key=lambda x: x[1], reverse=True)
    
    channels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=channels,
        y=values,
        marker=dict(
            color=values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Coefficient")
        ),
        text=[f"{v:.4f}" for v in values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Coefficient: %{y:.6f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        xaxis_title="Channel",
        yaxis_title="Coefficient",
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig
