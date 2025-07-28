import numpy as np
import pandas as pd
from portfolio_analyzer import PortfolioAnalyzer
from dataclasses import dataclass
from tqdm import tqdm
import scipy.linalg

@dataclass
class SimulationSettings:
    # market data
    returns: pd.Series             # MultiIndex (date, symbol)
    cap_flag: pd.Series            # MultiIndex (date, symbol)
    investability_flag: pd.Series  # MultiIndex (date, symbol)
    factors_df: pd.DataFrame       # your factor DataFrame
    
    # simulation parameters (with defaults)
    method: str = 'equal'          # 'equal' | 'linear' | 'mvo'
    transaction_cost: bool = True
    max_weight: float = 0.01
    pct: float = 0.1
    min_universe: int = 1000
    contributor: bool = False
    output_summary: bool = False
    output_returns: bool = False
    plot: bool = True
    
    # MVO-specific parameters
    mvo_window: int = 60           # look-back window for covariance estimation
    risk_aversion: float = 0.1     # risk aversion parameter
    turnover_penalty: float = 0.1  # penalty for portfolio turnover
    covariance_shrinkage: float = 0.1  # covariance shrinkage intensity [0-1]

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
        else:
            raise ValueError(f"Unknown method {self.method}")
        
    def _optimal_weight_daily_trade_list(self):
        """
        Mean-variance optimization with turnover penalty.
        
        For each date:
        1. Estimate expected returns and covariance using rolling window
        2. Solve mean-variance optimization with turnover penalty
        3. Normalize long and short legs separately
        4. Apply constraints (max weight, etc.)
        
        Returns:
            weights: pd.Series with MultiIndex (date, symbol) - one-day lagged
            counts_df: pd.DataFrame with date index and columns ['long_count', 'short_count']
        """
        # Get MVO parameters from settings with defaults
        window = getattr(self.settings, "mvo_window", 60)  # look-back window
        risk_aversion = getattr(self.settings, "risk_aversion", 1.0)  # risk aversion parameter
        turnover_penalty = getattr(self.settings, "turnover_penalty", 0.1)  # turnover penalty strength
        shrinkage = getattr(self.settings, "covariance_shrinkage", 0.1)  # covariance shrinkage
        
        weights_list = []
        count_list = []
        prev_weights = pd.Series(dtype=float)
        prev_counts = {"long_count": np.nan, "short_count": np.nan}
        
        # Get returns data in wide format for efficient computation
        returns_wide = self.returns.unstack().sort_index()
        
        # Create date to position mapping for efficient slicing
        date_to_pos = {d: pos for pos, d in enumerate(returns_wide.index)}
        
        # Iterate through dates with progress bar
        dates_iter = tqdm(self.custom_feature.index.get_level_values("date").unique(), desc="MVO optimization")
        
        for date in dates_iter:
            # Get current factor values for this date (already filtered by investability_flag in run())
            current_factors = self.custom_feature.xs(date, level="date").dropna()
            
            # Check minimum universe size
            if len(current_factors) < self.min_universe:
                # Carry forward previous weights if available
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue
            
            # Get historical returns for estimation
            pos = date_to_pos.get(date)
            if pos is None or pos < window:
                # Not enough history, carry forward
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue
            
            # Get historical data for current investable symbols
            start_pos = max(0, pos - window)
            hist_returns = returns_wide.iloc[start_pos:pos]
            
            # Filter to current investable symbols (same as current_factors)
            hist_returns = hist_returns[current_factors.index].dropna(axis=1, how="all")
            
            if len(hist_returns.columns) < self.min_universe:
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue
            
            # Estimate expected returns and covariance
            mu = hist_returns.mean()  # Expected returns
            cov = hist_returns.cov()  # Covariance matrix
            
            # Align factor values with return estimates
            common_symbols = current_factors.index.intersection(mu.index)
            if len(common_symbols) < self.min_universe:
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue
            
            # Use factor values as expected returns (alpha)
            alpha = current_factors.reindex(common_symbols)
            mu_aligned = mu.reindex(common_symbols)
            
            # Combine alpha with historical mean (can be adjusted)
            expected_returns = alpha  # Pure factor-based approach
            # Alternative: expected_returns = 0.5 * alpha + 0.5 * mu_aligned
            
            # Get covariance matrix for common symbols only
            cov_common = cov.loc[common_symbols, common_symbols]
            
            # Apply covariance shrinkage for stability
            if shrinkage > 0:
                diag_cov = np.diag(np.diag(cov_common.values))
                cov_shrink = (1.0 - shrinkage) * cov_common.values + shrinkage * diag_cov
            else:
                cov_shrink = cov_common.values
            
            # Solve mean-variance optimization
            try:
                # Use factor values as expected returns
                w_optimal = self._solve_mvo_optimization(
                    expected_returns, 
                    cov_shrink, 
                    common_symbols,
                    risk_aversion,
                    turnover_penalty,
                    prev_weights
                )
            except Exception as e:
                # Fallback to simple approach if optimization fails
                print(f"Optimization failed for {date}: {e}")
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product(
                        [[date], carried.index],
                        names=["date", "symbol"]
                    )
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue
            
            # Apply weight constraints and normalize
            w_optimal = self._cap_and_redistribute(w_optimal, self.max_weight)
            
            # Normalize weights to ensure market neutrality
            w_optimal = self._normalize_weights(w_optimal)
            
            # Count positions
            long_count = int((w_optimal > 0).sum())
            short_count = int((w_optimal < 0).sum())
            
            # Store results
            prev_weights = w_optimal
            prev_counts = {"long_count": long_count, "short_count": short_count}
            
            # Add date level to weights
            w_optimal.index = pd.MultiIndex.from_product(
                [[date], w_optimal.index],
                names=["date", "symbol"]
            )
            weights_list.append(w_optimal)
            count_list.append({
                "date": date,
                "long_count": long_count,
                "short_count": short_count
            })
        
        # Combine all results
        if not weights_list:
            return pd.Series(dtype=float), pd.DataFrame(columns=["long_count", "short_count"])
        
        all_weights = pd.concat(weights_list).sort_index()
        shifted_weights = all_weights.groupby("symbol").shift(1)  # One-day lag
        counts_df = pd.DataFrame(count_list).set_index("date")
        
        return shifted_weights, counts_df
    
    def _solve_mvo_optimization(self, expected_returns, covariance, symbols, 
                               risk_aversion, turnover_penalty, prev_weights):
        """
        Solve mean-variance optimization with turnover penalty.
        
        Objective function: maximize μ'w - (λ/2)w'Σw - (γ/2)||w - w_prev||²
        
        Args:
            expected_returns: pd.Series of expected returns
            covariance: np.array of covariance matrix
            symbols: list of symbols
            risk_aversion: float, risk aversion parameter (λ)
            turnover_penalty: float, penalty for turnover (γ)
            prev_weights: pd.Series of previous weights
            
        Returns:
            pd.Series of optimal weights
        """
        n_assets = len(symbols)
        
        # If no previous weights, solve simple MVO
        if prev_weights.empty:
            # Simple mean-variance optimization: w ∝ Σ⁻¹ μ
            # The solution is: w = (1/λ) * Σ⁻¹ μ
            try:
                # Use scipy.linalg.solve for efficiency
                w_raw = scipy.linalg.solve(covariance, expected_returns.values, assume_a='sym')
            except Exception:
                # Fallback to pseudo-inverse if matrix is singular
                w_raw = np.linalg.pinv(covariance).dot(expected_returns.values)
            
            # Scale by risk aversion (higher risk aversion = smaller weights)
            w_raw = w_raw / risk_aversion
            
            w_optimal = pd.Series(w_raw, index=symbols)
            
        else:
            # With turnover penalty
            # Align previous weights with current symbols
            prev_aligned = prev_weights.reindex(symbols).fillna(0.0)
            
            # Solve: w = argmax μ'w - (λ/2)w'Σw - (γ/2)||w - w_prev||²
            # This gives: w = (λΣ + γI)⁻¹(μ + γw_prev)
            
            gamma = turnover_penalty
            identity = np.eye(n_assets)
            
            try:
                # Solve (λΣ + γI)w = μ + γw_prev
                rhs = expected_returns.values + gamma * prev_aligned.values
                lhs = risk_aversion * covariance + gamma * identity
                w_raw = scipy.linalg.solve(lhs, rhs, assume_a='sym')
            except Exception:
                # Fallback to pseudo-inverse
                lhs = risk_aversion * covariance + gamma * identity
                w_raw = np.linalg.pinv(lhs).dot(rhs)
            
            w_optimal = pd.Series(w_raw, index=symbols)
        
        return w_optimal
    
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
    


    def _equal_weight_daily_trade_list(self):
        weights_list, count_list = [], []
        prev_weights = pd.Series(dtype=float)
        prev_long = prev_short = np.nan

        for date, group in self.custom_feature.groupby(level="date"):
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
                    count_list.append({"date": date,
                                       "long_count": prev_long,
                                       "short_count": prev_short})
                continue

            n = len(x)
            k = max(int(np.floor(n * self.pct)), 1)
            ranked = x.rank(method="first", ascending=False)

            long_mask = (ranked <= k)
            short_mask = (ranked > n - k)

            longs = long_mask.astype(float)
            shorts = short_mask.astype(float)
            if longs.sum() > 0:
                longs /= longs.sum()
            if shorts.sum() > 0:
                shorts /= shorts.sum()

            weights = longs - shorts
            long_count = int(long_mask.sum())
            short_count = int(short_mask.sum())

            # cap and redistribute
            weights = self._cap_and_redistribute(weights, self.max_weight)

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

        for date, group in self.custom_feature.groupby(level='date'):
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
            if not pos.empty:
                w[pos.index] = pos / pos.sum()
            if not neg.empty:
                w[neg.index] = neg / -neg.sum()

            long_count  = int((w>0).sum())
            short_count = int((w<0).sum())
            w = self._cap_and_redistribute(w, self.max_weight)

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

        all_w = pd.concat(weights_list)
        shifted = all_w.groupby("symbol").shift(1)
        counts_df = pd.DataFrame(count_list).set_index("date")
        return shifted, counts_df

    @staticmethod
    def _cap_and_redistribute(weights: pd.Series, max_weight: float,
                              max_iter: int = 10, tol: float = 1e-6) -> pd.Series:
        w = weights.copy()
        for _ in range(max_iter):
            capped = w.clip(lower=-max_weight, upper=max_weight)

            long_left  = 1 - capped[capped>0].sum()
            short_left = -1 - capped[capped<0].sum()

            uncapped_l = capped[(w>0)&(capped<max_weight)]
            uncapped_s = capped[(w<0)&(capped>-max_weight)]

            if (abs(long_left)<tol and abs(short_left)<tol) or \
               (uncapped_l.empty and uncapped_s.empty):
                break

            if not uncapped_l.empty and abs(long_left)>tol:
                adj = uncapped_l / uncapped_l.sum()
                capped.loc[adj.index] += long_left * adj

            if not uncapped_s.empty and abs(short_left)>tol:
                adj = uncapped_s / uncapped_s.sum()
                capped.loc[adj.index] += short_left * adj

            w = capped

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