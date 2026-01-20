"""
Marketing Mix Modeling Utilities

This package provides advanced MMM capabilities:
- Adstock transformations (geometric, Weibull, delayed, carryover)
- Saturation functions (Hill, logistic, Michaelis-Menten)
- Optimization (single-objective, multi-objective, scenario planning)
- Decomposition (ROAS, iROAS, contribution analysis)
- Visualization (adstock curves, saturation curves, waterfall charts, Pareto frontiers)
"""

from .adstock import (
    geometric_adstock,
    weibull_adstock,
    delayed_adstock,
    carryover_adstock,
    apply_adstock_to_dataframe,
    get_adstock_curve
)

from .saturation import (
    hill_saturation,
    logistic_saturation,
    michaelis_menten_saturation,
    fit_hill_saturation,
    apply_saturation_to_dataframe,
    get_saturation_curve,
    calculate_optimal_spend
)

from .optimization import (
    single_objective_optimizer,
    multi_objective_optimizer,
    scenario_analysis,
    sensitivity_analysis
)

from .decomposition import (
    decompose_sales,
    calculate_channel_roas,
    calculate_incremental_roas,
    create_waterfall_data,
    calculate_contribution_over_time,
    calculate_marginal_roas
)

from .visualization import (
    plot_adstock_curves,
    plot_saturation_curves,
    plot_contribution_waterfall,
    plot_actual_vs_predicted,
    plot_channel_contributions_over_time,
    plot_pareto_frontier,
    plot_tornado_chart,
    plot_coefficient_importance
)

__all__ = [
    # Adstock
    'geometric_adstock',
    'weibull_adstock',
    'delayed_adstock',
    'carryover_adstock',
    'apply_adstock_to_dataframe',
    'get_adstock_curve',
    
    # Saturation
    'hill_saturation',
    'logistic_saturation',
    'michaelis_menten_saturation',
    'fit_hill_saturation',
    'apply_saturation_to_dataframe',
    'get_saturation_curve',
    'calculate_optimal_spend',
    
    # Optimization
    'single_objective_optimizer',
    'multi_objective_optimizer',
    'scenario_analysis',
    'sensitivity_analysis',
    
    # Decomposition
    'decompose_sales',
    'calculate_channel_roas',
    'calculate_incremental_roas',
    'create_waterfall_data',
    'calculate_contribution_over_time',
    'calculate_marginal_roas',
    
    # Visualization
    'plot_adstock_curves',
    'plot_saturation_curves',
    'plot_contribution_waterfall',
    'plot_actual_vs_predicted',
    'plot_channel_contributions_over_time',
    'plot_pareto_frontier',
    'plot_tornado_chart',
    'plot_coefficient_importance',
]

__version__ = '2.0.0'
