import numpy as np
import pandas as pd
from portfolio_analyzer import PortfolioAnalyzer
from dataclasses import dataclass
from tqdm import tqdm
from scipy.optimize import minimize
import warnings

@dataclass
class SimulationSettings:
    # market data
    returns: pd.Series             # MultiIndex (date, symbol)
    cap_flag: pd.Series            # MultiIndex (date, symbol)
    investability_flag: pd.Series  # MultiIndex (date, symbol)
    factors_df: pd.DataFrame       # your factor DataFrame
    single_factor_returns: pd.DataFrame # date
    
    # simulation parameters (with defaults)
    method: str = 'equal'          # 'equal' | 'linear' | 'mvo' | 'barra' | 'pca_mvo'
    transaction_cost: bool = True
    max_weight: float = 0.01
    pct: float = 0.1
    min_universe: int = 1000
    contributor: bool = False
    output_summary: bool = False
    output_returns: bool = False
    plot: bool = True
    original_factors: pd.DataFrame = None
    
    # MVO-specific parameters
    mvo_window: int = 60           # look-back window for covariance estimation
    risk_aversion: float = 1.0      # risk aversion parameter (higher = more conservative)
    turnover_penalty: float = 5.0   # penalty for portfolio turnover (higher = less turnover)
    factor_weight: float = 0.5     # weight for composite factor (0.0 = pure MVO, 1.0 = pure factor)
    
    # PCA-MVO parameters
    pca_n_components: int = 10      # number of PCA components for PCA-MVO
    
    # Barra-style parameters
    factor_names: list = None       # list of factor names to use for Barra optimization (None = use all available)
    barra_window: int = 252         # look-back window for factor return estimation
    barra_risk_aversion: float = 1.0  # risk aversion for Barra optimization

class Simulation:
    """
    Runs a daily long/short simulation given a factor series and SimulationSettings.
    """
    def __init__(
        self,
        name: str,
        custom_feature: pd.Series,       # MultiIndex (date, symbol)
        settings: SimulationSettings
    ):
        self.name = name
        self.custom_feature     = custom_feature
        self.settings           = settings

        # unpack for convenience
        s = settings
        self.returns            = s.returns
        self.cap_flag           = s.cap_flag
        self.investability_flag = s.investability_flag
        self.factors_df         = s.factors_df
        self.method             = s.method
        self.transaction_cost   = s.transaction_cost
        self.max_weight         = s.max_weight
        self.pct                = s.pct
        self.min_universe       = s.min_universe
        self.contributor        = s.contributor
        self.output_summary     = s.output_summary
        self.output_returns     = s.output_returns
        self.plot               = s.plot
        self.single_factor_returns = s.single_factor_returns
        self.original_factors     = s.original_factors

        # MVO-specific parameters
        self.mvo_window = getattr(s, "mvo_window", 60)
        self.risk_aversion = getattr(s, "risk_aversion", 1.0)
        self.turnover_penalty = getattr(s, "turnover_penalty", 5.0)
        self.factor_weight = getattr(s, "factor_weight", 0.5)
        
        # Barra-specific parameters
        self.factor_names = getattr(s, "factor_names", None)
        self.barra_window = getattr(s, "barra_window", 252)
        self.barra_risk_aversion = getattr(s, "barra_risk_aversion", 1.0)
        # PCA-MVO parameters
        self.pca_n_components = getattr(s, "pca_n_components", 10)

    def run(self):
        self.factors_df[self.name] = self.custom_feature
        self.custom_feature = self.custom_feature * self.investability_flag
        weights, counts = self._daily_trade_list()
        result, top_longs, top_shorts = self._daily_portfolio_returns(weights)
        analyzer = PortfolioAnalyzer(result)

        if self.output_summary:
            metrics = self._calculate_metrics(weights, counts)
            summary_df = (
                pd.DataFrame.from_dict(analyzer.summary(), orient='index', columns=['Value'])
                .reset_index()
                .rename(columns={'index':'Metric'})
            )
            print(metrics.to_string(index=False))
            print(summary_df.to_string(index=False))
        if self.contributor:
            print('Top 10 long leg contributors:', top_longs)
            print('Top 10 short leg contributors:', top_shorts)
        if self.plot:
            analyzer.plot_full_performance(counts_df=counts)
        if self.output_returns:
            return result
        return None

    def _daily_trade_list(self):
        if self.method == 'equal':
            return self._equal_weight_daily_trade_list()
        elif self.method == 'linear':
            return self._flexible_weight_daily_trade_list()
        elif self.method == 'mvo':
            return self._optimal_weight_daily_trade_list()
        elif self.method == 'barra':
            return self._barra_style_daily_trade_list()
        elif self.method == 'pca_mvo':
            return self._pca_mvo_daily_trade_list()
        else:
            raise ValueError(f"Unknown method {self.method}")
        
    def _optimal_weight_daily_trade_list(self):
        """
        Mean-Variance Optimization combined with composite factor weighting.
        
        Process:
        1. Calculate composite factor weights for each date
        2. Apply mean-variance optimization on historical returns 
        3. Combine both weights using tunable parameter
        4. Process and return final weights
        """
        weights_list = []
        count_list = []
        prev_weights = pd.Series(dtype=float)
        prev_counts = {"long_count": np.nan, "short_count": np.nan}

        for date, group in tqdm(self.custom_feature.groupby(level='date'), desc="MVO Weight Simulation"):
            x = group.droplevel('date').dropna()
            if len(x) < self.min_universe:
                # carry previous if too small
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue

            try:
                # Get historical returns for MVO
                returns_data = self._get_historical_returns(date, x.index)
                
                if returns_data is None or returns_data.empty:
                    # Fall back to factor-only weights if no historical data
                    combined_weights = x.fillna(0)
                else:
                    # Calculate composite factor weights (inline)
                    factor_weights = x.fillna(0)
                    
                    # Calculate MVO weights
                    mvo_weights = self._calculate_mvo_weights(returns_data, prev_weights)
                    
                    # Combine weights
                    combined_weights = self._combine_weights(factor_weights, mvo_weights, x.index)
                
                # Process weights (normalize and apply constraints)
                weights = self._process_weights(combined_weights)
                
                # Count positions
                long_count = int((weights > 0).sum())
                short_count = int((weights < 0).sum())
                
                prev_weights = weights
                prev_counts = {"long_count": long_count, "short_count": short_count}

                # annotate date level
                weights.index = pd.MultiIndex.from_product(
                    [[date], weights.index],
                    names=["date", "symbol"]
                )
                weights_list.append(weights)
                count_list.append({"date": date,
                                   "long_count": long_count,
                                   "short_count": short_count})
                                   
            except Exception as e:
                warnings.warn(f"MVO optimization failed for date {date}: {str(e)}. Using previous weights.")
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})

        if not weights_list:
            # Return empty results if no weights were generated
            empty_weights = pd.Series(dtype=float)
            empty_counts = pd.DataFrame(columns=["long_count", "short_count"])
            return empty_weights, empty_counts

        all_w = pd.concat(weights_list).sort_index()
        shifted = all_w.groupby("symbol").shift(1)
        counts_df = pd.DataFrame(count_list).set_index("date")
        return shifted, counts_df

    def _get_historical_returns(self, current_date, symbols):
        """
        Get historical returns for mean-variance optimization (optimized version).
        
        Args:
            current_date: Current date for which to get historical data
            symbols: List of symbols to get returns for
            
        Returns:
            pd.DataFrame: Historical returns matrix (dates x symbols)
        """
        try:
            # Cache all_dates computation (only done once per simulation)
            if not hasattr(self, '_cached_dates'):
                self._cached_dates = self.returns.index.get_level_values('date').unique().sort_values()
                # Pre-compute date to index mapping for fast lookups
                self._date_to_idx = {date: idx for idx, date in enumerate(self._cached_dates)}
            
            date_idx = self._date_to_idx.get(current_date)
            if date_idx is None or date_idx < self.mvo_window:
                return None
                
            # Get historical window indices
            start_idx = max(0, date_idx - self.mvo_window)
            historical_dates = self._cached_dates[start_idx:date_idx]
            
            # Use more efficient indexing with loc and boolean masks
            date_mask = self.returns.index.get_level_values('date').isin(historical_dates)
            symbol_mask = self.returns.index.get_level_values('symbol').isin(symbols)
            
            # Extract returns efficiently 
            returns_subset = self.returns.loc[date_mask & symbol_mask]
            
            if returns_subset.empty:
                return None
            
            # Faster pivot using unstack with better memory handling
            returns_matrix = returns_subset.unstack(level='symbol', fill_value=0.0)
            
            # Vectorized symbol filtering (much faster than loops)
            if returns_matrix.empty:
                return None
                
            # Keep only symbols with sufficient data (vectorized)
            min_observations = min(10, len(historical_dates) // 2)
            non_zero_counts = (returns_matrix != 0).sum(axis=0)
            valid_mask = non_zero_counts >= min_observations
            
            if valid_mask.sum() < 2:
                return None
                
            valid_symbols = returns_matrix.columns[valid_mask]
            result = returns_matrix[valid_symbols]
            
            # Ensure we have the minimum required data
            if result.shape[0] < min(10, self.mvo_window // 2):
                return None
                
            return result
            
        except Exception as e:
            warnings.warn(f"Error getting historical returns: {str(e)}")
            return None

    def _calculate_mvo_weights(self, returns_data, prev_weights):
        """
        Calculate mean-variance optimal weights (optimized version).
        
        Args:
            returns_data: pd.DataFrame of historical returns (dates x symbols)
            prev_weights: pd.Series of previous weights for turnover penalty
            
        Returns:
            pd.Series: MVO optimal weights
        """
        try:
            n_assets = len(returns_data.columns)
            if n_assets < 2:
                return pd.Series(0.0, index=returns_data.columns)
            
            # Convert to numpy for faster computation
            returns_matrix = returns_data.values
            expected_returns = np.mean(returns_matrix, axis=0)
            
            # Calculate covariance matrix (faster)
            cov_matrix = np.cov(returns_matrix.T)
            
            # Fast regularization: simple diagonal loading (much faster than eigenvalue computation)
            reg_param = max(1e-8, 0.001 * np.trace(cov_matrix) / n_assets)
            cov_matrix += reg_param * np.eye(n_assets)
            
            # Pre-align previous weights to avoid repeated operations in objective
            aligned_prev = np.zeros(n_assets)
            if prev_weights is not None and not prev_weights.empty:
                for i, symbol in enumerate(returns_data.columns):
                    if symbol in prev_weights.index:
                        aligned_prev[i] = prev_weights[symbol]
            
            # Try analytical solution first for speed (when no turnover penalty)
            if self.turnover_penalty == 0 or prev_weights is None or prev_weights.empty:
                # Analytical mean-variance solution for market-neutral portfolio
                try:
                    # Solve: minimize w'Σw subject to w'1 = 0 and incorporate expected returns
                    ones = np.ones(n_assets)
                    inv_cov = np.linalg.inv(cov_matrix)
                    
                    # Market neutral analytical solution with return consideration
                    A = ones.T @ inv_cov @ ones
                    B = expected_returns.T @ inv_cov @ ones  
                    C = expected_returns.T @ inv_cov @ expected_returns
                    
                    if A > 1e-10:  # Avoid division by zero
                        # Optimal weights for market neutral portfolio
                        lambda_mult = B / A
                        weights = (inv_cov @ expected_returns - lambda_mult * inv_cov @ ones) / self.risk_aversion
                        
                        # Apply position limits
                        max_single_weight = min(0.1, 2.0 / n_assets)
                        weights = np.clip(weights, -max_single_weight, max_single_weight)
                        
                        # Renormalize to market neutral
                        if np.sum(weights) != 0:
                            weights = weights - np.sum(weights) / n_assets
                        
                        return pd.Series(weights, index=returns_data.columns)
                        
                except (np.linalg.LinAlgError, ValueError):
                    pass  # Fall back to numerical optimization
            
            # Numerical optimization (faster version)
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                
                # Mean-variance objective (negative because we minimize)
                objective_value = -portfolio_return + 0.5 * self.risk_aversion * portfolio_variance
                
                # Add turnover penalty (pre-aligned weights)
                if self.turnover_penalty > 0:
                    turnover = np.sum(np.abs(weights - aligned_prev))
                    objective_value += self.turnover_penalty * turnover
                
                return objective_value
            
            # Faster optimizer: L-BFGS-B (generally faster than SLSQP for unconstrained problems)
            max_single_weight = min(0.1, 2.0 / n_assets)
            bounds = [(-max_single_weight, max_single_weight) for _ in range(n_assets)]
            
            # Better initial guess: equal long-short based on expected returns
            ranked_returns = np.argsort(-expected_returns)  # Descending order
            x0 = np.zeros(n_assets)
            n_long = max(1, n_assets // 4)
            n_short = max(1, n_assets // 4)
            x0[ranked_returns[:n_long]] = 1.0 / n_long
            x0[ranked_returns[-n_short:]] = -1.0 / n_short
            
            # Solve optimization with L-BFGS-B
            result = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-6}  # Reduced tolerance for speed
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=returns_data.columns)
            else:
                # Fallback to equal long-short if optimization fails
                warnings.warn(f"MVO optimization failed: {result.message}. Using fallback weights.")
                optimal_weights = pd.Series(0.0, index=returns_data.columns)
                
                # Simple long-short based on expected returns
                if len(expected_returns) > 0:
                    ranked_returns = expected_returns.rank(ascending=False)
                    n_long = max(1, len(expected_returns) // 4)
                    n_short = max(1, len(expected_returns) // 4)
                    
                    optimal_weights[ranked_returns <= n_long] = 1.0 / n_long
                    optimal_weights[ranked_returns > len(expected_returns) - n_short] = -1.0 / n_short
            
            return optimal_weights
            
        except Exception as e:
            warnings.warn(f"Error in MVO calculation: {str(e)}")
            return pd.Series(0.0, index=returns_data.columns)

    def _combine_weights(self, factor_weights, optimization_weights, symbols):
        """
        Combine factor weights and optimization weights using tunable parameter (optimized).
        
        Args:
            factor_weights: pd.Series of factor-based weights
            optimization_weights: pd.Series of optimization weights (MVO, Barra, PCA-MVO, etc.)
            symbols: Index of symbols to ensure alignment
            
        Returns:
            pd.Series: Combined weights
        """
        # Use reindex for faster alignment (avoids loops)
        aligned_factor = factor_weights.reindex(symbols, fill_value=0.0)
        aligned_optimization = optimization_weights.reindex(symbols, fill_value=0.0)
        
        # Vectorized normalization (faster than pandas operations)
        factor_values = aligned_factor.values
        factor_std = np.std(factor_values)
        
        if factor_std > 1e-8:
            factor_mean = np.mean(factor_values)
            factor_values = (factor_values - factor_mean) / factor_std
            factor_values = np.clip(factor_values, -3, 3)  # Clip outliers
            aligned_factor = pd.Series(factor_values, index=symbols)
        
        # Combine using tunable parameter (vectorized)
        combined_weights = (self.factor_weight * aligned_factor + 
                          (1 - self.factor_weight) * aligned_optimization)
        
        return combined_weights

    def _calculate_pca_mvo_weights(self, returns_data, prev_weights):
        """
        Calculate PCA-based mean-variance optimal weights.
        
        This method applies Principal Component Analysis to reduce the dimensionality
        of the covariance matrix, performs optimization in the reduced space,
        then maps the solution back to the original asset space.
        
        Args:
            returns_data: pd.DataFrame of historical returns (dates x symbols)
            prev_weights: pd.Series of previous weights for turnover penalty
            
        Returns:
            pd.Series: PCA-MVO optimal weights
        """
        try:
            n_assets = len(returns_data.columns)
            if n_assets < 2:
                return pd.Series(0.0, index=returns_data.columns)
            
            # If we have fewer assets than PCA components, fall back to regular MVO
            if n_assets <= self.pca_n_components:
                warnings.warn(f"Number of assets ({n_assets}) <= PCA components ({self.pca_n_components}). Using regular MVO.")
                return self._calculate_mvo_weights(returns_data, prev_weights)
            
            # Convert to numpy for faster computation
            returns_matrix = returns_data.values
            expected_returns = np.mean(returns_matrix, axis=0)
            
            # Calculate covariance matrix
            cov_matrix = np.cov(returns_matrix.T)
            
            # Apply PCA to the covariance matrix (eigenvalue decomposition)
            # Use faster eigh with subset selection
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues and eigenvectors in descending order (vectorized)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            # Select top k components
            n_components = min(self.pca_n_components, len(eigenvalues))
            selected_eigenvalues = eigenvalues[:n_components]
            selected_eigenvectors = eigenvectors[:, :n_components]
            
            # Create reduced covariance matrix in PCA space
            # Σ_reduced = Λ (diagonal matrix of selected eigenvalues)
            reduced_cov = np.diag(selected_eigenvalues)
            
            # Project expected returns into PCA space (vectorized)
            # μ_reduced = V^T * μ
            reduced_returns = selected_eigenvectors.T @ expected_returns
            
            # Regularization in reduced space
            reg_param = max(1e-8, 0.001 * np.trace(reduced_cov) / n_components)
            reduced_cov += reg_param * np.eye(n_components)
            
            # Pre-align previous weights for turnover penalty (vectorized)
            aligned_prev = np.zeros(n_assets)
            if prev_weights is not None and not prev_weights.empty:
                # Vectorized alignment
                common_symbols = returns_data.columns.intersection(prev_weights.index)
                if len(common_symbols) > 0:
                    symbol_to_idx = {symbol: i for i, symbol in enumerate(returns_data.columns)}
                    for symbol in common_symbols:
                        aligned_prev[symbol_to_idx[symbol]] = prev_weights[symbol]
            
            # Project previous weights into PCA space (vectorized)
            reduced_prev_weights = selected_eigenvectors.T @ aligned_prev
            
            # Try analytical solution first (when no turnover penalty)
            if self.turnover_penalty == 0 or prev_weights is None or prev_weights.empty:
                try:
                    # Analytical solution in PCA space
                    ones_reduced = np.ones(n_components)
                    inv_cov_reduced = np.linalg.inv(reduced_cov)
                    
                    # Market neutral analytical solution
                    A = ones_reduced.T @ inv_cov_reduced @ ones_reduced
                    B = reduced_returns.T @ inv_cov_reduced @ ones_reduced
                    
                    if A > 1e-10:
                        lambda_mult = B / A
                        weights_reduced = (inv_cov_reduced @ reduced_returns - 
                                         lambda_mult * inv_cov_reduced @ ones_reduced) / self.risk_aversion
                        
                        # Map back to original space: w = V * w_reduced
                        weights_original = selected_eigenvectors @ weights_reduced
                        
                        # Apply position limits
                        max_single_weight = min(0.1, 2.0 / n_assets)
                        weights_original = np.clip(weights_original, -max_single_weight, max_single_weight)
                        
                        # Renormalize to market neutral
                        if np.sum(weights_original) != 0:
                            weights_original = weights_original - np.sum(weights_original) / n_assets
                        
                        return pd.Series(weights_original, index=returns_data.columns)
                        
                except (np.linalg.LinAlgError, ValueError):
                    pass  # Fall back to numerical optimization
            
            # Numerical optimization in PCA space (optimized)
            def objective_pca(weights_reduced):
                # Portfolio return and variance in PCA space
                portfolio_return = np.dot(weights_reduced, reduced_returns)
                portfolio_variance = np.dot(weights_reduced, np.dot(reduced_cov, weights_reduced))
                
                # Mean-variance objective
                objective_value = -portfolio_return + 0.5 * self.risk_aversion * portfolio_variance
                
                # Add turnover penalty (optimized - avoid repeated matrix multiplication)
                if self.turnover_penalty > 0:
                    # Pre-compute the difference in reduced space
                    turnover_reduced = np.sum(np.abs(weights_reduced - reduced_prev_weights))
                    # Scale by approximate factor (this is an approximation but much faster)
                    objective_value += self.turnover_penalty * turnover_reduced * (n_assets / n_components)
                
                return objective_value
            
            # Optimization bounds in reduced space (more relaxed since we have fewer variables)
            max_reduced_weight = min(1.0, 4.0 / n_components)
            bounds_reduced = [(-max_reduced_weight, max_reduced_weight) for _ in range(n_components)]
            
            # Initial guess in PCA space
            x0_reduced = np.zeros(n_components)
            if len(reduced_returns) > 0:
                ranked_returns = np.argsort(-reduced_returns)
                n_long = max(1, n_components // 4)
                n_short = max(1, n_components // 4)
                x0_reduced[ranked_returns[:n_long]] = 1.0 / n_long
                x0_reduced[ranked_returns[-n_short:]] = -1.0 / n_short
            
            # Solve optimization in PCA space
            result = minimize(
                objective_pca,
                x0_reduced,
                method='L-BFGS-B',
                bounds=bounds_reduced,
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            
            if result.success:
                optimal_weights_original = selected_eigenvectors @ result.x
                
                # Apply position limits in original space
                max_single_weight = min(0.1, 2.0 / n_assets)
                optimal_weights_original = np.clip(optimal_weights_original, -max_single_weight, max_single_weight)
                
                # Renormalize to market neutral
                if np.sum(optimal_weights_original) != 0:
                    optimal_weights_original = optimal_weights_original - np.sum(optimal_weights_original) / n_assets
                
                optimal_weights = pd.Series(optimal_weights_original, index=returns_data.columns)
            else:
                # Fallback to equal long-short if optimization fails (same as original MVO)
                warnings.warn(f"PCA-MVO optimization failed: {result.message}. Using fallback weights.")
                optimal_weights = pd.Series(0.0, index=returns_data.columns)
                
                # Simple long-short based on expected returns
                if len(expected_returns) > 0:
                    ranked_returns = pd.Series(expected_returns, index=returns_data.columns).rank(ascending=False)
                    n_long = max(1, len(expected_returns) // 4)
                    n_short = max(1, len(expected_returns) // 4)
                    
                    optimal_weights[ranked_returns <= n_long] = 1.0 / n_long
                    optimal_weights[ranked_returns > len(expected_returns) - n_short] = -1.0 / n_short
            
            return optimal_weights
            
        except Exception as e:
            warnings.warn(f"Error in PCA-MVO calculation: {str(e)}")
            return pd.Series(0.0, index=returns_data.columns)

    def _barra_style_daily_trade_list(self):
        """
        Barra-style multi-factor risk model portfolio construction.
        
        Process:
        1. Calculate expected asset returns via: R̂_i = Σ_k β_ik · f_k
        2. Compute full asset covariance matrix: Σ = BFB^T + D
        3. Solve mean-variance optimization: max x^T R̂ - (γ/2) x^T Σ x
        4. Apply constraints: full investment (Σx_i = 1) and optionally long-only
        """
        weights_list = []
        count_list = []
        prev_weights = pd.Series(dtype=float)
        prev_counts = {"long_count": np.nan, "short_count": np.nan}

        for date, group in tqdm(self.custom_feature.groupby(level='date'), desc="Barra-style Risk Model Simulation"):
            x = group.droplevel('date').dropna()
            if len(x) < self.min_universe:
                # carry previous if too small
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue

            try:
                # Get historical returns for Barra
                returns_data = self._get_historical_returns(date, x.index)
                
                if returns_data is None or returns_data.empty:
                    # Fall back to factor-only weights if no historical data
                    combined_weights = x.fillna(0)
                else:
                    # Calculate composite factor weights (inline)
                    factor_weights = x.fillna(0)
                    
                    # Calculate Barra weights
                    barra_weights = self._calculate_barra_weights(date, x.index)
                    
                    if barra_weights is None or barra_weights.empty:
                        # Fall back to factor-only weights if Barra calculation fails
                        combined_weights = factor_weights
                    else:
                        # Combine weights
                        combined_weights = self._combine_weights(factor_weights, barra_weights, x.index)
                
                # Process weights (normalize and apply constraints)
                weights = self._process_weights(combined_weights)
                
                # Count positions
                long_count = int((weights > 0).sum())
                short_count = int((weights < 0).sum())
                
                prev_weights = weights
                prev_counts = {"long_count": long_count, "short_count": short_count}

                # annotate date level
                weights.index = pd.MultiIndex.from_product(
                    [[date], weights.index],
                    names=["date", "symbol"]
                )
                weights_list.append(weights)
                count_list.append({"date": date,
                                   "long_count": long_count,
                                   "short_count": short_count})
                                   
            except Exception as e:
                warnings.warn(f"Barra optimization failed for date {date}: {str(e)}. Using factor-based fallback.")
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                else:
                    # Use factor-based fallback
                    factor_weights = self._process_weights(x.fillna(0))
                    factor_weights.index = pd.MultiIndex.from_product(
                        [[date], factor_weights.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(factor_weights)
                    count_list.append({"date": date, "long_count": int((factor_weights > 0).sum()), "short_count": int((factor_weights < 0).sum())})

        if not weights_list:
            # Return empty results if no weights were generated
            empty_weights = pd.Series(dtype=float)
            empty_counts = pd.DataFrame(columns=["long_count", "short_count"])
            return empty_weights, empty_counts

        all_w = pd.concat(weights_list).sort_index()
        shifted = all_w.groupby("symbol").shift(1)
        counts_df = pd.DataFrame(count_list).set_index("date")
        return shifted, counts_df

    def _calculate_barra_weights(self, current_date, symbols):
        """
        Calculate Barra-style portfolio weights using multi-factor risk model.
        
        Steps:
        1. Estimate factor exposures (B matrix) via regression of asset returns on factor returns
        2. Calculate expected asset returns: R̂_i = Σ_k β_ik · f_k  
        3. Calculate asset covariance matrix: Σ = BFB^T + D
        4. Solve mean-variance optimization: max x^T R̂ - (γ/2) x^T Σ x
        """
        try:
            # Check if we have factor returns data
            if self.single_factor_returns is None or self.single_factor_returns.empty:
                return None
                
            # Get factor returns data up to current date
            factor_data = self.single_factor_returns.loc[self.single_factor_returns.index < current_date]
            if factor_data.empty:
                return None
                
            # Use lookback window for regression
            window_size = min(self.barra_window, len(factor_data))
            recent_factor_returns = factor_data.tail(window_size)
            
            # Get asset returns for the same period (reuse existing function)
            historical_returns = self._get_historical_returns(current_date, symbols)
            if historical_returns is None or historical_returns.empty:
                return None
            
            # Align the time periods between factor returns and asset returns
            common_dates = recent_factor_returns.index.intersection(historical_returns.index)
            if len(common_dates) < 20:  # Need minimum observations for regression
                return None
                
            factor_ret_aligned = recent_factor_returns.loc[common_dates]
            asset_ret_aligned = historical_returns.loc[common_dates]
            
            # Get available symbols and factors
            available_symbols = asset_ret_aligned.columns.intersection(symbols)
            available_factors = factor_ret_aligned.columns.dropna()
            
            if len(available_symbols) < 2 or len(available_factors) == 0:
                return None
            
            # Step 1: Estimate factor exposures through regression
            # For each asset, regress returns on factor returns to get betas
            factor_exposures = pd.DataFrame(index=available_symbols, columns=available_factors)
            specific_risks = []
            
            for symbol in available_symbols:
                asset_returns = asset_ret_aligned[symbol].dropna()
                
                # Align dates for this specific asset
                asset_dates = factor_ret_aligned.index.intersection(asset_returns.index)
                if len(asset_dates) < 10:  # Need minimum observations
                    factor_exposures.loc[symbol] = 0.0
                    specific_risks.append(0.01)
                    continue
                
                y = asset_returns.loc[asset_dates].values
                X = factor_ret_aligned.loc[asset_dates].values
                
                # Add constant term for regression
                X_with_const = np.column_stack([np.ones(len(X)), X])
                
                try:
                    # OLS regression: y = α + β₁f₁ + β₂f₂ + ... + ε
                    betas = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                    
                    # Store factor exposures (exclude intercept)
                    factor_exposures.loc[symbol] = betas[1:]
                    
                    # Calculate specific risk from regression residuals
                    y_pred = X_with_const @ betas
                    residuals = y - y_pred
                    specific_risk = max(np.var(residuals, ddof=len(betas)), 1e-6)
                    specific_risks.append(specific_risk)
                    
                except (np.linalg.LinAlgError, ValueError):
                    # Fallback if regression fails
                    factor_exposures.loc[symbol] = 0.0
                    specific_risks.append(0.01)
            
            factor_exposures = factor_exposures.fillna(0.0).astype(float)
            
            # Step 2: Calculate expected returns using recent factor returns
            recent_mean_factor_returns = recent_factor_returns.mean()
            expected_returns = factor_exposures.dot(recent_mean_factor_returns)
            
            # Step 3: Calculate covariance matrix Σ = BFB^T + D
            B = factor_exposures.values  # Asset exposures matrix
            n_assets = B.shape[0]
            
            # Factor covariance matrix F
            F = np.cov(recent_factor_returns.T)
            if F.ndim == 0:
                F = np.array([[F]])
            elif F.ndim == 1:
                F = F.reshape(1, 1)
            
            # Systematic covariance: BFB^T
            systematic_cov = B @ F @ B.T
            
            # Specific risk matrix D (diagonal)
            D = np.diag(specific_risks)
            
            # Full covariance matrix: Σ = BFB^T + D + regularization
            Sigma = systematic_cov + D
            reg_param = max(1e-8, 0.001 * np.trace(Sigma) / n_assets)
            Sigma += reg_param * np.eye(n_assets)
            
            # Step 4: Solve optimization (reuse existing pattern)
            mu = expected_returns.values
            
            # Try analytical solution first
            try:
                ones = np.ones(n_assets)
                inv_Sigma = np.linalg.inv(Sigma)
                A = ones.T @ inv_Sigma @ ones
                B_scalar = mu.T @ inv_Sigma @ ones
                
                if A > 1e-10:
                    lambda_mult = B_scalar / A
                    weights = (inv_Sigma @ mu - lambda_mult * inv_Sigma @ ones) / self.barra_risk_aversion
                    
                    # Apply constraints
                    max_weight = min(0.1, 2.0 / n_assets)
                    weights = np.clip(weights, -max_weight, max_weight)
                    if np.sum(weights) != 0:
                        weights = weights - np.sum(weights) / n_assets
                    
                    return pd.Series(weights, index=available_symbols)
                    
            except (np.linalg.LinAlgError, ValueError):
                pass
            
            # Fallback to numerical optimization
            def objective(w):
                return -np.dot(w, mu) + 0.5 * self.barra_risk_aversion * np.dot(w, np.dot(Sigma, w))
            
            max_weight = min(0.1, 2.0 / n_assets)
            bounds = [(-max_weight, max_weight)] * n_assets
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w)}]
            
            # Initial guess based on expected returns
            x0 = np.zeros(n_assets)
            if len(mu) > 0:
                ranked = np.argsort(-mu)
                n_long = max(1, n_assets // 4)
                n_short = max(1, n_assets // 4)
                x0[ranked[:n_long]] = 1.0 / n_long
                x0[ranked[-n_short:]] = -1.0 / n_short
            
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                            constraints=constraints, options={'maxiter': 500, 'ftol': 1e-6})
            
            if result.success:
                return pd.Series(result.x, index=available_symbols)
            else:
                # Final fallback: simple long-short based on expected returns
                weights = pd.Series(0.0, index=available_symbols)
                ranked_returns = expected_returns.rank(ascending=False)
                n_long = max(1, len(available_symbols) // 4)
                n_short = max(1, len(available_symbols) // 4)
                weights[ranked_returns <= n_long] = 1.0 / n_long
                weights[ranked_returns > len(available_symbols) - n_short] = -1.0 / n_short
                return weights
                
        except Exception as e:
            warnings.warn(f"Error in Barra weight calculation: {str(e)}")
            return None

    def _equal_weight_daily_trade_list(self):
        weights_list, count_list = [], []
        prev_weights = pd.Series(dtype=float)
        prev_long = prev_short = np.nan

        for date, group in tqdm(self.custom_feature.groupby(level="date"), desc="Equal Weight Simulation"):
            x = group.droplevel('date').dropna()
            if len(x) < self.min_universe:
                # carry previous if too small
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    # Use consistent count storage
                    prev_counts = {"long_count": prev_long, "short_count": prev_short}
                    count_list.append({"date": date, **prev_counts})
                continue

            n = len(x)
            k = max(int(np.floor(n * self.pct)), 1)
            ranked = x.rank(method="first", ascending=False)

            long_mask = (ranked <= k)
            short_mask = (ranked > n - k)

            longs = long_mask.astype(float)
            shorts = short_mask.astype(float)
            
            # Create raw weights (no normalization here - _process_weights handles it)
            weights = longs - shorts
            long_count = int(long_mask.sum())
            short_count = int(short_mask.sum())

            # Process weights (normalize and apply constraints)
            weights = self._process_weights(weights)

            prev_weights = weights
            prev_long, prev_short = long_count, short_count

            # annotate date level
            weights.index = pd.MultiIndex.from_product(
                [[date], weights.index],
                names=["date", "symbol"]
            )
            weights_list.append(weights)
            count_list.append({"date": date,
                               "long_count": long_count,
                               "short_count": short_count})

        all_w = pd.concat(weights_list).sort_index()
        shifted = all_w.groupby("symbol").shift(1)
        counts_df = pd.DataFrame(count_list).set_index("date")
        return shifted, counts_df

    def _flexible_weight_daily_trade_list(self):
        weights_list = []
        count_list = []
        prev_weights = pd.Series(dtype=float)
        prev_counts = {"long_count": np.nan, "short_count": np.nan}

        for date, group in tqdm(self.custom_feature.groupby(level='date'), desc="Flexible Weight Simulation"):
            x = group.droplevel('date').dropna()
            if len(x) < self.min_universe:
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue

            w = pd.Series(0.0, index=x.index)
            pos, neg = x[x>0], x[x<0]
            
            # Create raw weights (no normalization here - _process_weights handles it)
            if not pos.empty:
                w[pos.index] = pos
            if not neg.empty:
                w[neg.index] = neg

            long_count  = int((w>0).sum())
            short_count = int((w<0).sum())
            
            # Process weights (normalize and apply constraints)
            w = self._process_weights(w)

            prev_weights = w
            prev_counts = {"long_count": long_count, "short_count": short_count}

            w.index = pd.MultiIndex.from_product(
                [[date], w.index],
                names=["date", "symbol"]
            )
            weights_list.append(w)
            count_list.append({"date": date,
                               "long_count": long_count,
                               "short_count": short_count})

        all_w = pd.concat(weights_list).sort_index()
        shifted = all_w.groupby("symbol").shift(1)
        counts_df = pd.DataFrame(count_list).set_index("date")
        return shifted, counts_df

    def _pca_mvo_daily_trade_list(self):
        """
        PCA-based Mean-Variance Optimization combined with composite factor weighting.
        
        Process:
        1. Calculate composite factor weights for each date
        2. Apply PCA-based mean-variance optimization on historical returns 
        3. Combine both weights using tunable parameter
        4. Process and return final weights
        """
        weights_list = []
        count_list = []
        prev_weights = pd.Series(dtype=float)
        prev_counts = {"long_count": np.nan, "short_count": np.nan}

        for date, group in tqdm(self.custom_feature.groupby(level='date'), desc="PCA-MVO Weight Simulation"):
            x = group.droplevel('date').dropna()
            if len(x) < self.min_universe:
                # carry previous if too small
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue

            try:
                # Get historical returns for PCA-MVO
                returns_data = self._get_historical_returns(date, x.index)
                
                if returns_data is None or returns_data.empty:
                    # Fall back to factor-only weights if no historical data
                    combined_weights = x.fillna(0)
                else:
                    # Calculate composite factor weights (inline)
                    factor_weights = x.fillna(0)
                    
                    # Calculate PCA-MVO weights
                    pca_mvo_weights = self._calculate_pca_mvo_weights(returns_data, prev_weights)
                    
                    # Combine weights
                    combined_weights = self._combine_weights(factor_weights, pca_mvo_weights, x.index)
                
                # Process weights (normalize and apply constraints)
                weights = self._process_weights(combined_weights)
                
                # Count positions
                long_count = int((weights > 0).sum())
                short_count = int((weights < 0).sum())
                
                prev_weights = weights
                prev_counts = {"long_count": long_count, "short_count": short_count}

                # annotate date level
                weights.index = pd.MultiIndex.from_product(
                    [[date], weights.index],
                    names=["date", "symbol"]
                )
                weights_list.append(weights)
                count_list.append({"date": date,
                                   "long_count": long_count,
                                   "short_count": short_count})
                                   
            except Exception as e:
                warnings.warn(f"PCA-MVO optimization failed for date {date}: {str(e)}. Using previous weights.")
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})

        if not weights_list:
            # Return empty results if no weights were generated
            empty_weights = pd.Series(dtype=float)
            empty_counts = pd.DataFrame(columns=["long_count", "short_count"])
            return empty_weights, empty_counts

        all_w = pd.concat(weights_list).sort_index()
        shifted = all_w.groupby("symbol").shift(1)
        counts_df = pd.DataFrame(count_list).set_index("date")
        return shifted, counts_df

    def _process_weights(self, weights: pd.Series, max_weight: float = None) -> pd.Series:
        """
        Consistent weight processing across all methods.
        
        Process order:
        1. Normalize weights for market neutrality (long leg = 1, short leg = -1)
        2. Apply max weight constraints and redistribute
        
        Args:
            weights: pd.Series of raw weights
            max_weight: maximum absolute weight (uses self.max_weight if None)
            
        Returns:
            pd.Series of processed weights
        """
        if max_weight is None:
            max_weight = self.max_weight
            
        w = weights.copy()
        
        # Step 1: Normalize for market neutrality
        w = self._normalize_weights(w)
        
        # Step 2: Apply max weight constraints and redistribute
        w = self._cap_and_redistribute(w, max_weight)
        
        return w
    
    def _normalize_weights(self, weights):
        """
        Normalize weights to ensure market neutrality.
        Long and short legs are normalized separately.
        """
        w = weights.copy()
        
        # Separate long and short positions
        w_pos = w.clip(lower=0)
        w_neg = w.clip(upper=0)
        
        # Normalize long leg to sum to 1
        if w_pos.sum() > 0:
            w_pos /= w_pos.sum()
        
        # Normalize short leg to sum to -1
        if w_neg.sum() < 0:
            w_neg /= -w_neg.sum()
        
        # Combine
        w_normalized = w_pos + w_neg
        
        return w_normalized
    
    @staticmethod
    def _cap_and_redistribute(weights: pd.Series, max_weight: float,
                              max_iter: int = 10, tol: float = 1e-6) -> pd.Series:
        """
        Apply max weight constraints and redistribute excess weight while maintaining market neutrality.
        
        Args:
            weights: pd.Series of weights (should be normalized: long leg = 1, short leg = -1)
            max_weight: maximum absolute weight allowed
            max_iter: maximum iterations for redistribution
            tol: tolerance for convergence
            
        Returns:
            pd.Series of constrained weights
        """
        w = weights.copy()
        
        for _ in range(max_iter):
            # Apply max weight constraint
            capped = w.clip(lower=-max_weight, upper=max_weight)
            
            # Calculate remaining weight to redistribute
            long_excess = 1 - capped[capped > 0].sum()
            short_excess = -1 - capped[capped < 0].sum()
            
            # Find uncapped positions that can receive additional weight
            uncapped_long = capped[(w > 0) & (capped < max_weight)]
            uncapped_short = capped[(w < 0) & (capped > -max_weight)]
            
            # Check convergence
            if (abs(long_excess) < tol and abs(short_excess) < tol) or \
               (uncapped_long.empty and uncapped_short.empty):
                break
            
            # Redistribute long excess
            if not uncapped_long.empty and abs(long_excess) > tol:
                # Distribute proportionally to uncapped long positions
                redistribution = long_excess * (uncapped_long / uncapped_long.sum())
                capped.loc[redistribution.index] += redistribution
            
            # Redistribute short excess
            if not uncapped_short.empty and abs(short_excess) > tol:
                # Distribute proportionally to uncapped short positions
                redistribution = short_excess * (uncapped_short / uncapped_short.sum())
                capped.loc[redistribution.index] += redistribution
            
            w = capped
        
        # Final clip to ensure constraints are satisfied
        return w.clip(lower=-max_weight, upper=max_weight)

    def _daily_portfolio_returns(self, weights: pd.Series):
        # align
        df = pd.concat([
            weights.rename("weight"),
            self.returns.rename("ret"),
            self.cap_flag.rename("cap_flag")
        ], axis=1).dropna()
        df["pnl"] = df["weight"] * df["ret"]

        # simple daily return
        daily_ret = df.groupby(level="date")["pnl"].sum()
        w_df = weights.unstack().fillna(0)
        r_df = self.returns.unstack().fillna(0)
        longs  = w_df.clip(lower=0)
        shorts = w_df.clip(upper=0).abs()

        long_ret_raw  = (longs * r_df).sum(axis=1)
        short_ret_raw = -(shorts * r_df).sum(axis=1)

        lt  = longs.diff().abs().sum(axis=1)
        st  = shorts.diff().abs().sum(axis=1)
        rate_map = {1: 0.0025, 2: 0.0015, 3: 0.0010}
        cost_rates = self.cap_flag.unstack().fillna(0).astype(int).replace(rate_map)
        # wide legs
        if self.transaction_cost:
            l_cost = (longs.diff().abs() * cost_rates).sum(axis=1)
            s_cost = (shorts.diff().abs() * cost_rates).sum(axis=1)
            long_ret  = long_ret_raw  - l_cost
            short_ret = short_ret_raw - s_cost
        else:
            long_ret, short_ret = long_ret_raw, short_ret_raw
            l_cost = s_cost = pd.Series(0.0, index=daily_ret.index)

        net = long_ret + short_ret

        result = pd.concat([
            net.rename("log_return"),
            long_ret.rename("long_return"),
            short_ret.rename("short_return"),
            lt.rename("long_turnover"),
            st.rename("short_turnover"),
            (lt+st).rename("turnover")
        ], axis=1).reset_index().sort_values("date", ascending=False).reset_index(drop=True)

        if self.contributor:
            longs_pnl  = (longs * r_df).sum() - (longs.diff().abs() * cost_rates).sum()
            shorts_pnl = -(shorts * r_df).sum() - (shorts.diff().abs() * cost_rates).sum()
            return result, longs_pnl.nlargest(10), shorts_pnl.nlargest(10)

        return result, None, None

    def _calculate_metrics(self, weights: pd.Series, counts: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([
            self.custom_feature.rename("alpha"),
            self.returns.rename("ret")
        ], axis=1).dropna()
        daily_ic = df.groupby(level="date").apply(lambda x: x["alpha"].corr(x["ret"]))
        ic_mean, ic_std = daily_ic.mean(), daily_ic.std()
        ir = ic_mean/ic_std if ic_std else np.nan

        w_df = weights.unstack().fillna(0)
        longs  = w_df.clip(lower=0)
        shorts = w_df.clip(upper=0).abs()
        turnover_series = (longs.diff().abs() + shorts.diff().abs()).sum(axis=1)

        metrics = pd.DataFrame({
            "IC (%)": [ic_mean*100],
            "IC_IR (%)": [ir*100],
            "IC_Std (%)": [ic_std*100],
            "Avg Turnover (%)": [turnover_series.mean() * 100]
        })
        return round(metrics, 2)