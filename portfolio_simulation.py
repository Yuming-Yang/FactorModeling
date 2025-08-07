import numpy as np
import pandas as pd
from portfolio_analyzer import PortfolioAnalyzer
from dataclasses import dataclass
from tqdm import tqdm
import warnings
from scipy.optimize import minimize
import cvxpy as cp

@dataclass
class SimulationSettings:
    # market data
    returns: pd.Series             # MultiIndex (date, symbol)
    cap_flag: pd.Series            # MultiIndex (date, symbol)
    investability_flag: pd.Series  # MultiIndex (date, symbol)
    factors_df: pd.DataFrame       # your factor DataFrame
    # simulation parameters (with defaults)
    method: str = 'equal'          # 'equal' | 'linear' | 'mvo' | 'mvo_turnover'
    transaction_cost: bool = True
    max_weight: float = 0.03
    pct: float = 0.1
    min_universe: int = 1000
    contributor: bool = False
    output_summary: bool = False
    output_returns: bool = False
    plot: bool = True
    # MVO-specific parameters
    lookback_period: int = 60      # days to look back for variance calculation
    use_cvxpy: bool = True         # use CVXPY instead of scipy for faster optimization
    mvo_solver: str = 'OSQP'       # CVXPY solver: 'OSQP' (fastest), 'ECOS', 'SCS'
    shrinkage_intensity: float = 0.1  # shrinkage intensity for covariance regularization
    turnover_penalty: float = 0.1     # penalty weight for turnover in MVO with turnover optimization

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
        self.lookback_period    = s.lookback_period
        self.use_cvxpy          = s.use_cvxpy
        self.mvo_solver         = s.mvo_solver
        self.shrinkage_intensity = s.shrinkage_intensity
        self.turnover_penalty = s.turnover_penalty

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
        """Generate daily portfolio weights using the specified method."""
        weights_list, count_list = [], []

        for date, group in tqdm(self.custom_feature.groupby(level="date"), 
                               desc=f"{self.method.title()} Weight Simulation"):
            x = group.droplevel('date')
            
            # Separate positive and negative signals
            pos = x[x > 0]
            neg = x[x < 0]

            # If one leg is missing, stay flat
            if pos.empty or neg.empty:
                zero_w = pd.Series(0.0, index=x.index)
                zero_w.index = pd.MultiIndex.from_product(
                    [[date], zero_w.index], names=["date", "symbol"]
                )
                weights_list.append(zero_w)
                count_list.append({"date": date, "long_count": 0, "short_count": 0})
                continue

            # For MVO methods, check if we have enough assets for optimization
            if self.method in ['mvo', 'mvo_turnover'] and len(x) < 2:
                zero_w = pd.Series(0.0, index=x.index)
                zero_w.index = pd.MultiIndex.from_product(
                    [[date], zero_w.index], names=["date", "symbol"]
                )
                weights_list.append(zero_w)
                count_list.append({"date": date, "long_count": 0, "short_count": 0})
                continue

            # Calculate weights based on method
            if self.method == 'equal':
                weights, long_count, short_count = self._calculate_equal_weights(pos, neg, x.index)
            elif self.method == 'linear':
                weights, long_count, short_count = self._calculate_linear_weights(pos, neg, x.index)
            elif self.method == 'mvo':
                weights, long_count, short_count = self._calculate_mvo_weights(x, pos, neg, date)
            elif self.method == 'mvo_turnover':
                # Get previous weights for turnover penalty
                prev_weights = self._get_previous_weights(weights_list, x.index)
                weights, long_count, short_count = self._calculate_mvo_turnover_weights(x, pos, neg, date, prev_weights)
            else:
                raise ValueError(f"Unknown method {self.method}")

            # Add date to index
            weights.index = pd.MultiIndex.from_product(
                [[date], weights.index], names=["date", "symbol"]
            )
            
            weights_list.append(weights)
            count_list.append({"date": date, "long_count": long_count, "short_count": short_count})

        # Common post-processing
        all_w = pd.concat(weights_list).sort_index()
        shifted = all_w.groupby("symbol").shift(1)
        counts_df = pd.DataFrame(count_list).set_index("date")
        return shifted, counts_df

    def _calculate_equal_weights(self, pos, neg, symbols):
        """Calculate equal weights for positive and negative signals."""
        k_long = max(int(np.floor(len(pos) * self.pct)), 1)
        k_short = max(int(np.floor(len(neg) * self.pct)), 1)

        top_pos = pos.sort_values(ascending=False).iloc[:k_long]
        top_neg = neg.sort_values().iloc[:k_short]

        weights = pd.Series(0.0, index=symbols)
        weights[top_pos.index] = 1.0
        weights[top_neg.index] = -1.0

        weights = self._normalize_legs(weights)
        
        return weights, k_long, k_short

    def _calculate_linear_weights(self, pos, neg, symbols):
        """Calculate linear weights proportional to factor values."""
        weights = pd.Series(0.0, index=symbols)
        weights[pos.index] = pos
        weights[neg.index] = neg

        weights = self._normalize_legs(weights)
        weights = self._cap_and_redistribute(weights, self.max_weight)
        
        return weights, len(pos), len(neg)

    def _calculate_mvo_weights(self, composite_factor, pos, neg, current_date):
        """Calculate mean variance optimization weights with fast optimizer."""
        # Get covariance matrix
        cov_matrix = self._calculate_covariance_matrix(composite_factor.index, current_date)
        
        if cov_matrix is None or cov_matrix.shape[0] < 2:
            # Fallback to equal weights if covariance calculation fails
            return self._calculate_equal_weights(pos, neg, composite_factor.index)
        
        # Apply shrinkage to improve conditioning
        cov_matrix = self._apply_shrinkage(cov_matrix)
        
        # Use factor values as expected returns
        expected_returns = composite_factor
        
        # Perform mean variance optimization with fast CVXPY solver
        if self.use_cvxpy:
            optimal_weights = self._solve_mvo_cvxpy(expected_returns, cov_matrix, pos, neg)
        else:
            optimal_weights = self._solve_mvo(expected_returns, cov_matrix)
        
        return optimal_weights, len(pos), len(neg)

    def _get_previous_weights(self, weights_list, current_symbols):
        """Get previous day's weights for the current symbols, defaulting to zero if not available."""
        if not weights_list:
            # First day: no previous weights
            return pd.Series(0.0, index=current_symbols)
        
        # Get the most recent weights (last element in weights_list)
        prev_weights_with_date = weights_list[-1]
        
        # Remove date level to get just symbol weights
        prev_weights = prev_weights_with_date.droplevel('date')
        
        # Align with current symbols, filling missing symbols with 0
        aligned_weights = pd.Series(0.0, index=current_symbols)
        
        # Update with previous weights where available
        common_symbols = prev_weights.index.intersection(current_symbols)
        aligned_weights.loc[common_symbols] = prev_weights.loc[common_symbols]
        
        return aligned_weights

    def _calculate_mvo_turnover_weights(self, composite_factor, pos, neg, current_date, prev_weights):
        """Calculate mean variance optimization weights with turnover penalty."""
        # Get covariance matrix (same as regular MVO)
        cov_matrix = self._calculate_covariance_matrix(composite_factor.index, current_date)
        
        if cov_matrix is None or cov_matrix.shape[0] < 2:
            # Fallback to equal weights if covariance calculation fails
            return self._calculate_equal_weights(pos, neg, composite_factor.index)
        
        # Apply shrinkage to improve conditioning (same as regular MVO)
        cov_matrix = self._apply_shrinkage(cov_matrix)
        
        # Use factor values as expected returns (same as regular MVO)
        expected_returns = composite_factor
        
        # Perform mean variance optimization with turnover penalty
        if self.use_cvxpy:
            optimal_weights = self._solve_mvo_turnover_cvxpy(expected_returns, cov_matrix, pos, neg, prev_weights)
        else:
            optimal_weights = self._solve_mvo_turnover(expected_returns, cov_matrix, prev_weights)
        
        return optimal_weights, len(pos), len(neg)

    @staticmethod
    def _normalize_legs(weights: pd.Series) -> pd.Series:
        """Normalize long and short legs to sum to 1 and -1 respectively."""
        w_pos = weights.clip(lower=0)
        w_neg = weights.clip(upper=0)

        if w_pos.sum() > 0:
            w_pos /= w_pos.sum()
        
        if w_neg.sum() < 0:
            w_neg /= -w_neg.sum()
            
        return w_pos + w_neg

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

    def _calculate_covariance_matrix(self, symbols, current_date):
        """Calculate covariance matrix from historical returns that matches the symbols index exactly."""
        try:
            # Get historical returns for the symbols
            hist_returns = self.returns.loc[self.returns.index.get_level_values('symbol').isin(symbols)]
            
            # Filter to dates before current_date
            hist_returns = hist_returns.loc[hist_returns.index.get_level_values('date') < current_date]
            
            if len(hist_returns) == 0:
                return None
            
            # Get the last lookback_period days
            dates = hist_returns.index.get_level_values('date').unique().sort_values()
            if len(dates) < self.lookback_period:
                start_date = dates[0]
            else:
                start_date = dates[-self.lookback_period]
            
            hist_returns = hist_returns.loc[hist_returns.index.get_level_values('date') >= start_date]
            
            # Pivot to get returns matrix (dates x symbols)
            returns_matrix = hist_returns.unstack('symbol').fillna(0)
            
            # Ensure we include ALL symbols from the composite factor index
            # If a symbol is missing from returns data, fill with zeros
            missing_symbols = [s for s in symbols if s not in returns_matrix.columns]
            for symbol in missing_symbols:
                returns_matrix[symbol] = 0.0
            
            # Reorder columns to match the original symbols order
            returns_matrix = returns_matrix[list(symbols)]
            
            # Calculate covariance matrix
            cov_matrix = returns_matrix.cov()
            
            # Add small regularization to diagonal for numerical stability
            regularization = 1e-6
            np.fill_diagonal(cov_matrix.values, cov_matrix.values.diagonal() + regularization)
            
            return cov_matrix
            
        except Exception as e:
            warnings.warn(f"Covariance calculation failed: {e}")
            return None

    def _apply_shrinkage(self, cov_matrix):
        """Apply shrinkage to covariance matrix for better conditioning."""
        if self.shrinkage_intensity <= 0:
            return cov_matrix
        
        # Identity matrix scaled by average diagonal
        avg_var = np.mean(np.diag(cov_matrix.values))
        identity = np.eye(cov_matrix.shape[0]) * avg_var
        
        # Shrink towards identity matrix
        shrunk_cov = (1 - self.shrinkage_intensity) * cov_matrix.values + \
                     self.shrinkage_intensity * identity
        
        return pd.DataFrame(shrunk_cov, index=cov_matrix.index, columns=cov_matrix.columns)

    def _solve_mvo_cvxpy(self, expected_returns, cov_matrix, pos, neg):
        """
        Fast CVXPY-based mean variance optimization solver.
        Follows the same logic as _solve_mvo but uses CVXPY for faster optimization.
        """
        n = len(expected_returns)
        
        # Separate into positive and negative expected returns (same as original)
        pos_mask = expected_returns > 0
        neg_mask = expected_returns < 0
        
        # Initial guess: equal weights for long/short legs (same as original)
        x0 = np.zeros(n)
        x0[pos_mask] = 1.0 / pos_mask.sum()
        x0[neg_mask] = -1.0 / neg_mask.sum()
        
        # Create optimization variable
        w = cp.Variable(n)
        
        # Objective: minimize portfolio variance (same as original objective)
        # Make sure covariance matrix is positive semidefinite
        cov_np = cov_matrix.values
        cov_np = 0.5 * (cov_np + cov_np.T)  # Ensure symmetry
        objective = cp.Minimize(cp.quad_form(w, cov_np))
        
        # Constraints (exactly same as original)
        constraints = [
            # Long leg (positive signals) sums to +1
            cp.sum(w[pos_mask.values]) == 1.0,
            # Short leg (negative signals) sums to -1
            cp.sum(w[neg_mask.values]) == -1.0
        ]
        
        # Bounds: respect sign constraints and max weight limits (same as original)
        for i in range(n):
            if pos_mask.iloc[i]:
                # Positive signals: weight between 0 and max_weight
                constraints.append(w[i] >= 0)
                constraints.append(w[i] <= self.max_weight)
            elif neg_mask.iloc[i]:
                # Negative signals: weight between -max_weight and 0
                constraints.append(w[i] <= 0)
                constraints.append(w[i] >= -self.max_weight)
            else:
                # Zero signals: weight must be 0
                constraints.append(w[i] == 0)
        
        # Create and solve problem
        prob = cp.Problem(objective, constraints)
        
        try:
            # Use fast OSQP solver with relaxed settings for speed
            prob.solve(
                solver=getattr(cp, self.mvo_solver),
                verbose=False,
                eps_abs=1e-4,  # Relaxed absolute tolerance
                eps_rel=1e-4,  # Relaxed relative tolerance
                max_iter=2000,  # Limit iterations
                adaptive_rho=True,  # Adaptive penalty parameter
                polish=True,  # Polish solution
                warm_start=True  # Warm start
            )
            
            if prob.status == cp.OPTIMAL:
                optimal_weights = pd.Series(w.value, index=expected_returns.index)
                
                # Verify constraints are satisfied (same as original)
                long_sum = optimal_weights[pos_mask].sum()
                short_sum = optimal_weights[neg_mask].sum()
                total_sum = optimal_weights.sum()
                
                # Small tolerance for numerical precision
                if abs(long_sum - 1.0) > 1e-6 or abs(short_sum + 1.0) > 1e-6:
                    warnings.warn(f"CVXPY MVO constraints not satisfied: long_sum={long_sum:.6f}, short_sum={short_sum:.6f}, total_sum={total_sum:.6f}")
                
            else:
                # Fallback to initial guess if optimization fails (same as original)
                optimal_weights = pd.Series(x0, index=expected_returns.index)
                warnings.warn(f"CVXPY optimization failed with status: {prob.status}")
                
        except Exception as e:
            warnings.warn(f"CVXPY optimization failed: {e}")
            # Fallback to initial guess (same as original)
            optimal_weights = pd.Series(x0, index=expected_returns.index)
        
        return optimal_weights

    def _solve_mvo_turnover_cvxpy(self, expected_returns, cov_matrix, pos, neg, prev_weights):
        """
        Fast CVXPY-based mean variance optimization solver with turnover penalty.
        Follows the same logic as _solve_mvo_cvxpy but adds turnover penalty to the objective.
        
        Objective: minimize portfolio_variance + turnover_penalty * turnover
        where turnover = sum(|w_i - prev_w_i|)
        """
        n = len(expected_returns)
        
        # Separate into positive and negative expected returns (same as original)
        pos_mask = expected_returns > 0
        neg_mask = expected_returns < 0
        
        # Initial guess: equal weights for long/short legs (same as original)
        x0 = np.zeros(n)
        x0[pos_mask] = 1.0 / pos_mask.sum()
        x0[neg_mask] = -1.0 / neg_mask.sum()
        
        # Create optimization variable
        w = cp.Variable(n)
        
        # Objective: minimize portfolio variance + turnover penalty
        # Make sure covariance matrix is positive semidefinite
        cov_np = cov_matrix.values
        cov_np = 0.5 * (cov_np + cov_np.T)  # Ensure symmetry
        
        # Portfolio variance term (same as original)
        variance_term = cp.quad_form(w, cov_np)
        
        # Turnover penalty term: sum of absolute differences from previous weights
        prev_weights_np = prev_weights.values
        turnover_term = cp.sum(cp.abs(w - prev_weights_np))
        
        # Combined objective
        objective = cp.Minimize(variance_term + self.turnover_penalty * turnover_term)
        
        # Constraints (exactly same as original)
        constraints = [
            # Long leg (positive signals) sums to +1
            cp.sum(w[pos_mask.values]) == 1.0,
            # Short leg (negative signals) sums to -1
            cp.sum(w[neg_mask.values]) == -1.0
        ]
        
        # Bounds: respect sign constraints and max weight limits (same as original)
        for i in range(n):
            if pos_mask.iloc[i]:
                # Positive signals: weight between 0 and max_weight
                constraints.append(w[i] >= 0)
                constraints.append(w[i] <= self.max_weight)
            elif neg_mask.iloc[i]:
                # Negative signals: weight between -max_weight and 0
                constraints.append(w[i] <= 0)
                constraints.append(w[i] >= -self.max_weight)
            else:
                # Zero signals: weight must be 0
                constraints.append(w[i] == 0)
        
        # Create and solve problem
        prob = cp.Problem(objective, constraints)
        
        try:
            # Use fast OSQP solver with relaxed settings for speed
            prob.solve(
                solver=getattr(cp, self.mvo_solver),
                verbose=False,
                eps_abs=1e-4,  # Relaxed absolute tolerance
                eps_rel=1e-4,  # Relaxed relative tolerance
                max_iter=2000,  # Limit iterations
                adaptive_rho=True,  # Adaptive penalty parameter
                polish=True,  # Polish solution
                warm_start=True  # Warm start
            )
            
            if prob.status == cp.OPTIMAL:
                optimal_weights = pd.Series(w.value, index=expected_returns.index)
                
                # Verify constraints are satisfied (same as original)
                long_sum = optimal_weights[pos_mask].sum()
                short_sum = optimal_weights[neg_mask].sum()
                total_sum = optimal_weights.sum()
                
                # Small tolerance for numerical precision
                if abs(long_sum - 1.0) > 1e-6 or abs(short_sum + 1.0) > 1e-6:
                    warnings.warn(f"CVXPY MVO+Turnover constraints not satisfied: long_sum={long_sum:.6f}, short_sum={short_sum:.6f}, total_sum={total_sum:.6f}")
                
            else:
                # Fallback to initial guess if optimization fails (same as original)
                optimal_weights = pd.Series(x0, index=expected_returns.index)
                warnings.warn(f"CVXPY MVO+Turnover optimization failed with status: {prob.status}")
                
        except Exception as e:
            warnings.warn(f"CVXPY MVO+Turnover optimization failed: {e}")
            # Fallback to initial guess (same as original)
            optimal_weights = pd.Series(x0, index=expected_returns.index)
        
        return optimal_weights

    def _solve_mvo(self, expected_returns, cov_matrix):
        """
        Solve mean variance optimization problem with proper constraints:
        - Long leg sums to +1
        - Short leg sums to -1  
        - Total portfolio weight sums to 0 (market neutral)
        - Individual weights respect max_weight bounds
        """
        n = len(expected_returns)
        
        # Separate into positive and negative expected returns
        pos_mask = expected_returns > 0
        neg_mask = expected_returns < 0
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix.values, weights))
        
        # Initial guess: equal weights for long/short legs
        x0 = np.zeros(n)
        x0[pos_mask] = 1.0 / pos_mask.sum()
        x0[neg_mask] = -1.0 / neg_mask.sum()
        
        # Constraints
        constraints = [
            # Long leg (positive signals) sums to +1
            {'type': 'eq', 'fun': lambda w: w[pos_mask].sum() - 1.0},
            # Short leg (negative signals) sums to -1
            {'type': 'eq', 'fun': lambda w: w[neg_mask].sum() + 1.0}
        ]
        
        # Bounds: respect sign constraints and max weight limits
        bounds = []
        for i in range(n):
            if pos_mask.iloc[i]:
                # Positive signals: weight between 0 and max_weight
                bounds.append((0, self.max_weight))
            elif neg_mask.iloc[i]:
                # Negative signals: weight between -max_weight and 0
                bounds.append((-self.max_weight, 0))
            else:
                # Zero signals: weight must be 0
                bounds.append((0, 0))
        
        try:
            # Solve optimization
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=expected_returns.index)
                
                # Verify constraints are satisfied (for debugging)
                long_sum = optimal_weights[pos_mask].sum()
                short_sum = optimal_weights[neg_mask].sum()
                total_sum = optimal_weights.sum()
                
                # Small tolerance for numerical precision
                if abs(long_sum - 1.0) > 1e-6 or abs(short_sum + 1.0) > 1e-6:
                    warnings.warn(f"MVO constraints not satisfied: long_sum={long_sum:.6f}, short_sum={short_sum:.6f}, total_sum={total_sum:.6f}")
                
            else:
                # Fallback to initial guess if optimization fails
                optimal_weights = pd.Series(x0, index=expected_returns.index)
                warnings.warn(f"MVO optimization failed: {result.message}")
                
        except Exception as e:
            warnings.warn(f"MVO optimization failed: {e}")
            # Fallback to initial guess
            optimal_weights = pd.Series(x0, index=expected_returns.index)
        
        return optimal_weights

    def _solve_mvo_turnover(self, expected_returns, cov_matrix, prev_weights):
        """
        Solve mean variance optimization problem with turnover penalty using scipy.
        Follows the same logic as _solve_mvo but adds turnover penalty to the objective.
        
        Objective: minimize portfolio_variance + turnover_penalty * turnover
        where turnover = sum(|w_i - prev_w_i|)
        """
        n = len(expected_returns)
        
        # Separate into positive and negative expected returns
        pos_mask = expected_returns > 0
        neg_mask = expected_returns < 0
        
        # Objective function: minimize portfolio variance + turnover penalty
        def objective(weights):
            # Portfolio variance term
            variance_term = np.dot(weights, np.dot(cov_matrix.values, weights))
            
            # Turnover penalty term: sum of absolute differences from previous weights
            turnover_term = np.sum(np.abs(weights - prev_weights.values))
            
            return variance_term + self.turnover_penalty * turnover_term
        
        # Initial guess: equal weights for long/short legs (same as original)
        x0 = np.zeros(n)
        x0[pos_mask] = 1.0 / pos_mask.sum()
        x0[neg_mask] = -1.0 / neg_mask.sum()
        
        # Constraints (exactly same as original)
        constraints = [
            # Long leg (positive signals) sums to +1
            {'type': 'eq', 'fun': lambda w: w[pos_mask].sum() - 1.0},
            # Short leg (negative signals) sums to -1
            {'type': 'eq', 'fun': lambda w: w[neg_mask].sum() + 1.0}
        ]
        
        # Bounds: respect sign constraints and max weight limits (same as original)
        bounds = []
        for i in range(n):
            if pos_mask.iloc[i]:
                # Positive signals: weight between 0 and max_weight
                bounds.append((0, self.max_weight))
            elif neg_mask.iloc[i]:
                # Negative signals: weight between -max_weight and 0
                bounds.append((-self.max_weight, 0))
            else:
                # Zero signals: weight must be 0
                bounds.append((0, 0))
        
        try:
            # Solve optimization
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=expected_returns.index)
                
                # Verify constraints are satisfied (for debugging)
                long_sum = optimal_weights[pos_mask].sum()
                short_sum = optimal_weights[neg_mask].sum()
                total_sum = optimal_weights.sum()
                
                # Small tolerance for numerical precision
                if abs(long_sum - 1.0) > 1e-6 or abs(short_sum + 1.0) > 1e-6:
                    warnings.warn(f"MVO+Turnover constraints not satisfied: long_sum={long_sum:.6f}, short_sum={short_sum:.6f}, total_sum={total_sum:.6f}")
                
            else:
                # Fallback to initial guess if optimization fails
                optimal_weights = pd.Series(x0, index=expected_returns.index)
                warnings.warn(f"MVO+Turnover optimization failed: {result.message}")
                
        except Exception as e:
            warnings.warn(f"MVO+Turnover optimization failed: {e}")
            # Fallback to initial guess
            optimal_weights = pd.Series(x0, index=expected_returns.index)
        
        return optimal_weights

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