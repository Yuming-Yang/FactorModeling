import pandas as pd
import numpy as np
import scipy.stats as stats
import cvxpy as cp

def icir_top_selector(metrics_df, factors_win, returns_win, factor_ret_win, today, window, icir_threshold=0.03, top_x=5, use_rank_icir=True, **kwargs):
    """
    Select top-N factors by IC_IR or rank_IC_IR from metrics_df.
    Parameters:
    - metrics_df: DataFrame of per-factor metrics (from single_factor_metrics)
    - factors_win: DataFrame of factor exposures in the window (not used)
    - returns_win: Series of asset returns in the window (not used)
    - factor_ret_win: DataFrame of factor returns in the window (not used)
    - today: Current date
    - window: Lookback window size (list of dates)
    - icir_threshold: Minimum IC_IR or rank_IC_IR
    - top_x: Number of top factors to select
    - use_rank_icir: Whether to use rank_IC_IR (True) or IC_IR (False)
    """
    col = "rank_IC_IR" if use_rank_icir else "IC_IR"
    selected = metrics_df[metrics_df[col] > icir_threshold].nlargest(top_x, col)
    vec = pd.Series(0.0, index=metrics_df.index, name=today)
    vec.loc[selected.index] = 1.0
    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec

def factor_momentum_selector(metrics_df, factors_win, returns_win, factor_ret_win, today, window, max_weight=1.0, **kwargs):
    """
    Select factors based on momentum (cumulative return over the window).
    Assign higher weights to factors with higher cumulative return in the window.
    Weights are clipped between 0 and max_weight, and min weight is 0.
    
    Parameters:
    - metrics_df: DataFrame of per-factor metrics (from single_factor_metrics)
    - factors_win: DataFrame of factor exposures in the window
    - returns_win: Series of asset returns in the window
    - factor_ret_win: DataFrame of factor returns in the window
    - today: Current date
    - window: Lookback window size (list of dates)
    - max_weight: Maximum weight per factor
    """
    # Use the factor names from metrics_df
    factor_names = metrics_df.index.tolist()
    # Use factor_ret_win directly
    factor_ret_win = factor_ret_win.loc[:, factor_names]
    # Sum returns over the window for each factor (momentum)
    momentum = factor_ret_win.sum()
    # Set negative or zero momentum to zero weight
    momentum = momentum.clip(lower=0)
    # Cap weights at max_weight
    if max_weight < 1.0:
        momentum = momentum.clip(upper=max_weight)
    # Normalize to sum to 1 if any positive
    vec = pd.Series(0.0, index=momentum.index, name = today)
    if momentum.sum() > 0:
        vec = momentum / momentum.sum()
    return vec

def ledoit_wolf_shrinkage(returns):
    """
    Apply Ledoit-Wolf shrinkage to sample covariance matrix.
    
    Parameters:
    - returns: DataFrame of factor returns (time x factors)
    
    Returns:
    - Shrunk covariance matrix
    """
    n, p = returns.shape  # n = observations, p = factors
    
    # Sample covariance matrix
    sample_cov = np.cov(returns, rowvar=False)
    
    # Target matrix (constant correlation)
    var = np.diag(sample_cov)
    std = np.sqrt(var)
    
    # --- Robustly calculate the average correlation ---
    correlations = []
    # Loop over unique pairs of factors
    for i in range(p):
        for j in range(i + 1, p):
            # Only calculate correlation if both stds are non-zero
            if std[i] > 0 and std[j] > 0:
                correlations.append(sample_cov[i, j] / (std[i] * std[j]))
    
    # If there are valid correlations, take the mean, otherwise default to 0
    mean_corr = np.mean(correlations) if correlations else 0
    
    target = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i == j:
                target[i, j] = var[i]
            else:
                target[i, j] = mean_corr * std[i] * std[j]
    
    # Compute shrinkage intensity
    d = np.sum((sample_cov - target) ** 2)
    
    # Compute variance of sample covariance entries
    returns_centered = returns - returns.mean(axis=0)
    cov_factors = np.zeros((p, p))
    for k in range(n):
        cov_k = np.outer(returns_centered[k], returns_centered[k])
        cov_factors += (cov_k - sample_cov) ** 2
    cov_factors /= n
    
    # Compute optimal shrinkage intensity
    lambda_opt = np.sum(cov_factors) / d
    lambda_opt = max(0, min(1, lambda_opt))
    
    # Apply shrinkage
    shrunk_cov = lambda_opt * target + (1 - lambda_opt) * sample_cov
    
    return shrunk_cov

def mvo_selector(
    metrics_df, factors_win, returns_win, factor_ret_win, today, window,
    risk_aversion=1.0, max_weight=1.0, turnover_penalty=0.0, previous_weights=None,
    use_shrinkage=True, **kwargs):
    """
    Mean-Variance Optimization selector: maximize mean/variance (Sharpe) of factor portfolio.
    Parameters:
    - metrics_df: DataFrame of per-factor metrics (from single_factor_metrics)
    - factors_win: DataFrame of factor exposures in the window (not used)
    - returns_win: Series of asset returns in the window (not used)
    - factor_ret_win: DataFrame of factor returns in the window
    - today: Current date
    - window: Lookback window size (list of dates)
    - risk_aversion: Risk aversion parameter
    - max_weight: Maximum weight per factor
    - turnover_penalty: Penalty for weight changes
    - previous_weights: Previous factor weights (Series or None)
    - use_shrinkage: Whether to apply Ledoit-Wolf shrinkage to covariance matrix
    """
    factor_names = metrics_df.index.tolist()
    n = len(factor_names)
    # Mean vector: always use mean of factor_ret_win
    mean = factor_ret_win[factor_names].mean()
    # Covariance matrix
    if use_shrinkage:
        cov = pd.DataFrame(
            ledoit_wolf_shrinkage(factor_ret_win[factor_names].values),
            index=factor_names, columns=factor_names
        )
    else:
        cov = factor_ret_win[factor_names].cov()

    w = cp.Variable(n)
    cov_matrix = 0.5 * (cov.values + cov.values.T)
    obj = mean.values @ w - risk_aversion * cp.quad_form(w, cov_matrix)
    # Turnover penalty
    if turnover_penalty > 0 and previous_weights is not None:
        prev = previous_weights.reindex(factor_names).fillna(0).values
        obj = obj - turnover_penalty * cp.norm1(w - prev)
    objective = cp.Maximize(obj)
    constraints = [cp.sum(w) == 1, w >= 0]
    if max_weight < 1.0:
        constraints.append(w <= max_weight)
    else:
        constraints.append(w <= 1)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        weights = w.value
        if weights is None:
            weights = np.zeros(n)
    except Exception:
        weights = np.zeros(n)
    vec = pd.Series(weights, index=factor_names, name=today)
    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec