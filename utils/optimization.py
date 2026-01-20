"""
Advanced Optimization Functions for Marketing Mix Modeling

Provides:
- Single-objective optimization (maximize sales, ROI, or brand awareness)
- Multi-objective optimization (Pareto frontier)
- Constraint-based optimization (min/max spend per channel)
- Scenario planning and sensitivity analysis
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


def single_objective_optimizer(
    model,
    total_budget: float,
    objective: str = 'sales',
    constraints: Optional[Dict] = None,
    channel_names: List[str] = None
) -> Tuple[np.ndarray, float]:
    """
    Optimize budget allocation for a single objective
    
    Parameters:
    -----------
    model : sklearn Pipeline or custom model
        Trained MMM model with predict() method
    total_budget : float
        Total marketing budget to allocate
    objective : str
        Optimization objective:
        - 'sales': Maximize predicted sales
        - 'roi': Maximize ROI (sales / spend)
        - 'efficiency': Maximize sales per dollar
    constraints : Optional[Dict]
        Budget constraints. Format:
        {
            'min_spend': {'TV': 10_000_000, 'Facebook': 5_000_000},
            'max_spend': {'TV': 100_000_000, 'Facebook': 50_000_000},
            'min_pct': {'TV': 0.2},  # TV must be at least 20% of budget
            'max_pct': {'TV': 0.5},  # TV cannot exceed 50% of budget
        }
    channel_names : List[str]
        List of channel names (for reference)
        
    Returns:
    --------
    Tuple[np.ndarray, float] : (optimal_allocation, predicted_outcome)
    
    Example:
    --------
    >>> constraints = {
    ...     'min_spend': {'TV': 20_000_000},
    ...     'max_pct': {'TV': 0.4, 'Facebook': 0.3}
    ... }
    >>> allocation, sales = single_objective_optimizer(model, 200_000_000, 'sales', constraints)
    """
    n_channels = len(channel_names) if channel_names else 4
    
    # Objective function
    def objective_func(spends):
        """Calculate objective value (negative for minimization)"""
        # Predict sales using model
        try:
            pred_sales = model.predict(spends.reshape(1, -1))[0]
        except:
            # If model expects different format, try this
            pred_sales = model.predict([spends])[0]
        
        if objective == 'sales':
            return -pred_sales  # Negative because we minimize
        elif objective == 'roi':
            total_spend = spends.sum()
            roi = pred_sales / total_spend if total_spend > 0 else 0
            return -roi
        elif objective == 'efficiency':
            total_spend = spends.sum()
            efficiency = pred_sales / total_spend if total_spend > 0 else 0
            return -efficiency
        else:
            return -pred_sales
    
    # Constraints
    constraint_list = []
    
    # Budget constraint (equality)
    constraint_list.append({
        'type': 'eq',
        'fun': lambda x: np.sum(x) - total_budget
    })
    
    # Process custom constraints
    if constraints:
        # Min spend constraints
        if 'min_spend' in constraints:
            for i, channel in enumerate(channel_names):
                if channel in constraints['min_spend']:
                    min_val = constraints['min_spend'][channel]
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=i, mv=min_val: x[idx] - mv
                    })
        
        # Max spend constraints
        if 'max_spend' in constraints:
            for i, channel in enumerate(channel_names):
                if channel in constraints['max_spend']:
                    max_val = constraints['max_spend'][channel]
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=i, mv=max_val: mv - x[idx]
                    })
        
        # Min percentage constraints
        if 'min_pct' in constraints:
            for i, channel in enumerate(channel_names):
                if channel in constraints['min_pct']:
                    min_pct = constraints['min_pct'][channel]
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=i, mp=min_pct: x[idx] - (mp * total_budget)
                    })
        
        # Max percentage constraints
        if 'max_pct' in constraints:
            for i, channel in enumerate(channel_names):
                if channel in constraints['max_pct']:
                    max_pct = constraints['max_pct'][channel]
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=i, mp=max_pct: (mp * total_budget) - x[idx]
                    })
    
    # Bounds (no negative spend, max = total budget)
    bounds = tuple((0, total_budget) for _ in range(n_channels))
    
    # Initial guess (equal allocation)
    x0 = np.array([total_budget / n_channels] * n_channels)
    
    # Optimize
    result = minimize(
        objective_func,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraint_list,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if result.success:
        return result.x, -result.fun
    else:
        print(f"Warning: Optimization did not converge. Using equal allocation.")
        return x0, -objective_func(x0)


def multi_objective_optimizer(
    model,
    total_budget: float,
    objectives: List[str] = ['sales', 'roi'],
    n_solutions: int = 20,
    constraints: Optional[Dict] = None,
    channel_names: List[str] = None
) -> pd.DataFrame:
    """
    Multi-objective optimization using NSGA-II (genetic algorithm)
    
    Finds Pareto-optimal solutions (trade-offs between objectives).
    
    Parameters:
    -----------
    model : sklearn Pipeline or custom model
        Trained MMM model
    total_budget : float
        Total marketing budget
    objectives : List[str]
        List of objectives to optimize:
        ['sales', 'roi', 'brand_awareness', 'efficiency']
    n_solutions : int
        Number of Pareto-optimal solutions to find
    constraints : Optional[Dict]
        Budget constraints (same format as single_objective_optimizer)
    channel_names : List[str]
        List of channel names
        
    Returns:
    --------
    pd.DataFrame : Pareto-optimal solutions with columns:
        - One column per channel (allocation)
        - One column per objective (outcome)
    
    Example:
    --------
    >>> solutions = multi_objective_optimizer(
    ...     model, 200_000_000, ['sales', 'roi'], n_solutions=30
    ... )
    >>> # solutions has 30 rows, each is a Pareto-optimal allocation
    """
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize as pymoo_minimize
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        
        n_channels = len(channel_names) if channel_names else 4
        
        class MMMProblem(Problem):
            """Custom optimization problem for pymoo"""
            
            def __init__(self):
                super().__init__(
                    n_var=n_channels,
                    n_obj=len(objectives),
                    n_constr=1,  # Budget constraint
                    xl=np.zeros(n_channels),  # Lower bounds
                    xu=np.full(n_channels, total_budget)  # Upper bounds
                )
            
            def _evaluate(self, X, out, *args, **kwargs):
                """Evaluate objectives and constraints"""
                n_samples = X.shape[0]
                
                # Objectives (to minimize, so negate for maximization)
                obj_values = np.zeros((n_samples, len(objectives)))
                
                for i in range(n_samples):
                    spends = X[i]
                    
                    # Predict sales
                    try:
                        pred_sales = model.predict(spends.reshape(1, -1))[0]
                    except:
                        pred_sales = model.predict([spends])[0]
                    
                    total_spend = spends.sum()
                    
                    for j, obj in enumerate(objectives):
                        if obj == 'sales':
                            obj_values[i, j] = -pred_sales  # Negate for minimization
                        elif obj == 'roi':
                            roi = pred_sales / total_spend if total_spend > 0 else 0
                            obj_values[i, j] = -roi
                        elif obj == 'efficiency':
                            eff = pred_sales / total_spend if total_spend > 0 else 0
                            obj_values[i, j] = -eff
                        else:
                            obj_values[i, j] = -pred_sales
                
                out["F"] = obj_values
                
                # Constraint: sum(spends) = total_budget
                out["G"] = np.abs(X.sum(axis=1) - total_budget)
        
        # Create problem
        problem = MMMProblem()
        
        # Create algorithm
        algorithm = NSGA2(
            pop_size=n_solutions,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # Run optimization
        res = pymoo_minimize(
            problem,
            algorithm,
            ('n_gen', 100),
            verbose=False
        )
        
        # Extract Pareto front
        pareto_X = res.X
        pareto_F = res.F
        
        # Create DataFrame
        results = pd.DataFrame(pareto_X, columns=channel_names)
        
        for i, obj in enumerate(objectives):
            results[obj] = -pareto_F[:, i]  # Un-negate
        
        return results
        
    except ImportError:
        print("Warning: pymoo not installed. Falling back to grid search.")
        return _multi_objective_grid_search(model, total_budget, objectives, n_solutions, channel_names)


def _multi_objective_grid_search(
    model,
    total_budget: float,
    objectives: List[str],
    n_solutions: int,
    channel_names: List[str]
) -> pd.DataFrame:
    """
    Fallback: Simple grid search for multi-objective optimization
    
    Used when pymoo is not available.
    """
    n_channels = len(channel_names)
    
    # Generate random allocations
    np.random.seed(42)
    solutions = []
    
    for _ in range(n_solutions * 10):  # Generate more, then filter
        # Random allocation that sums to total_budget
        weights = np.random.dirichlet(np.ones(n_channels))
        allocation = weights * total_budget
        
        # Predict sales
        try:
            pred_sales = model.predict(allocation.reshape(1, -1))[0]
        except:
            pred_sales = model.predict([allocation])[0]
        
        total_spend = allocation.sum()
        
        solution = {channel: allocation[i] for i, channel in enumerate(channel_names)}
        
        for obj in objectives:
            if obj == 'sales':
                solution[obj] = pred_sales
            elif obj == 'roi':
                solution[obj] = pred_sales / total_spend if total_spend > 0 else 0
            elif obj == 'efficiency':
                solution[obj] = pred_sales / total_spend if total_spend > 0 else 0
        
        solutions.append(solution)
    
    df = pd.DataFrame(solutions)
    
    # Filter to Pareto front (simple 2D case)
    if len(objectives) == 2:
        df = df.sort_values(objectives[0], ascending=False)
        pareto = [df.iloc[0]]
        
        for i in range(1, len(df)):
            if df.iloc[i][objectives[1]] > pareto[-1][objectives[1]]:
                pareto.append(df.iloc[i])
        
        df = pd.DataFrame(pareto).reset_index(drop=True)
    
    return df.head(n_solutions)


def scenario_analysis(
    model,
    base_allocation: np.ndarray,
    scenarios: Dict[str, Dict],
    channel_names: List[str]
) -> pd.DataFrame:
    """
    Analyze what-if scenarios
    
    Parameters:
    -----------
    model : sklearn Pipeline or custom model
        Trained MMM model
    base_allocation : np.ndarray
        Current/baseline budget allocation
    scenarios : Dict[str, Dict]
        Scenarios to test. Format:
        {
            'Cut TV 50%': {'TV': -0.5},
            'Double Facebook': {'Facebook': 1.0},
            'Shift 20M from TV to Digital': {'TV': -20_000_000, 'Facebook': +10_000_000, 'Instagram': +10_000_000}
        }
    channel_names : List[str]
        List of channel names
        
    Returns:
    --------
    pd.DataFrame : Results with columns:
        - Scenario name
        - New allocation per channel
        - Predicted sales
        - Change vs baseline
    
    Example:
    --------
    >>> scenarios = {
    ...     'Cut TV 50%': {'TV': -0.5},
    ...     'Double Social': {'Facebook': 1.0, 'Instagram': 1.0}
    ... }
    >>> results = scenario_analysis(model, current_allocation, scenarios, channels)
    """
    # Baseline prediction
    try:
        baseline_sales = model.predict(base_allocation.reshape(1, -1))[0]
    except:
        baseline_sales = model.predict([base_allocation])[0]
    
    results = []
    
    # Add baseline
    baseline_row = {'Scenario': 'Baseline (Current)'}
    for i, channel in enumerate(channel_names):
        baseline_row[channel] = base_allocation[i]
    baseline_row['Predicted_Sales'] = baseline_sales
    baseline_row['Change_vs_Baseline'] = 0
    baseline_row['Change_Pct'] = 0.0
    results.append(baseline_row)
    
    # Test scenarios
    for scenario_name, changes in scenarios.items():
        new_allocation = base_allocation.copy()
        
        for channel, change in changes.items():
            if channel in channel_names:
                idx = channel_names.index(channel)
                
                if isinstance(change, float) and -1 <= change <= 10:
                    # Percentage change
                    new_allocation[idx] *= (1 + change)
                else:
                    # Absolute change
                    new_allocation[idx] += change
                
                # Ensure non-negative
                new_allocation[idx] = max(0, new_allocation[idx])
        
        # Predict
        try:
            pred_sales = model.predict(new_allocation.reshape(1, -1))[0]
        except:
            pred_sales = model.predict([new_allocation])[0]
        
        scenario_row = {'Scenario': scenario_name}
        for i, channel in enumerate(channel_names):
            scenario_row[channel] = new_allocation[i]
        scenario_row['Predicted_Sales'] = pred_sales
        scenario_row['Change_vs_Baseline'] = pred_sales - baseline_sales
        scenario_row['Change_Pct'] = (pred_sales - baseline_sales) / baseline_sales if baseline_sales > 0 else 0
        
        results.append(scenario_row)
    
    return pd.DataFrame(results)


def sensitivity_analysis(
    model,
    base_allocation: np.ndarray,
    channel_names: List[str],
    perturbation_pct: float = 0.1
) -> Dict[str, float]:
    """
    Calculate sensitivity of sales to each channel's budget
    
    Shows which channels have the biggest impact on sales.
    
    Parameters:
    -----------
    model : sklearn Pipeline or custom model
        Trained MMM model
    base_allocation : np.ndarray
        Current budget allocation
    channel_names : List[str]
        List of channel names
    perturbation_pct : float
        Percentage to perturb each channel (default: 10%)
        
    Returns:
    --------
    Dict[str, float] : Sensitivity (impact) for each channel
    
    Example:
    --------
    >>> sensitivities = sensitivity_analysis(model, current_allocation, channels, 0.1)
    >>> # {'TV': 50_000_000, 'Facebook': 30_000_000, ...}
    """
    # Baseline
    try:
        baseline_sales = model.predict(base_allocation.reshape(1, -1))[0]
    except:
        baseline_sales = model.predict([base_allocation])[0]
    
    sensitivities = {}
    
    for i, channel in enumerate(channel_names):
        # Increase channel budget by perturbation_pct
        perturbed_allocation = base_allocation.copy()
        perturbed_allocation[i] *= (1 + perturbation_pct)
        
        # Predict
        try:
            perturbed_sales = model.predict(perturbed_allocation.reshape(1, -1))[0]
        except:
            perturbed_sales = model.predict([perturbed_allocation])[0]
        
        # Calculate impact
        impact = perturbed_sales - baseline_sales
        sensitivities[f"{channel} +{perturbation_pct:.0%}"] = impact
        
        # Decrease channel budget
        perturbed_allocation = base_allocation.copy()
        perturbed_allocation[i] *= (1 - perturbation_pct)
        
        try:
            perturbed_sales = model.predict(perturbed_allocation.reshape(1, -1))[0]
        except:
            perturbed_sales = model.predict([perturbed_allocation])[0]
        
        impact = perturbed_sales - baseline_sales
        sensitivities[f"{channel} -{perturbation_pct:.0%}"] = impact
    
    return sensitivities
