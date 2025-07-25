import numpy as np
import pandas as pd
from portfolio_analyzer import PortfolioAnalyzer
from dataclasses import dataclass
from tqdm import tqdm  # type: ignore
from tqdm.notebook import tqdm  # Use notebook-friendly progress bar
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
        Mean-variance optimisation with a turnover penalty.
        For each date we:
        1.  Estimate expected returns (mean) and covariance using a rolling
            window of past daily returns (no look-ahead).
        2.  Form the unconstrained mean–variance portfolio `w ∝ Σ⁻¹ μ`.
        3.  Normalise long and short legs separately so that gross exposure
            equals 1 on each side (market-neutral).
        4.  Apply an explicit turnover penalty by shrinking the new weights
            towards the previous day’s portfolio.
        5.  (No capping/redistribution in this version)
        The output structure (a lagged weight Series and a counts DataFrame)
        matches that of the other weighting methods, so the remainder of the
        simulation pipeline remains unchanged.
        """
        window          = getattr(self.settings, "mvo_window", 60)        # look-back window in trading days
        shrink          = getattr(self.settings, "mvo_shrink", 0.10)      # covariance shrinkage intensity [0-1]
        turnover_lambda = getattr(self.settings, "turnover_lambda", 0.20)  # 0 = ignore prev weights, 1 = keep prev
        returns_wide = self.returns.unstack().sort_index()
        weights_list: list[pd.Series] = []
        count_list:   list[dict] = []
        prev_weights = pd.Series(dtype=float)
        prev_counts  = {"long_count": np.nan, "short_count": np.nan}
        dates = self.custom_feature.index.get_level_values("date").unique()
        # Precompute investable symbols for each date
        investable_dict = {
            date: self.investability_flag.xs(date, level="date").dropna()
                [lambda x: x > 0].index
            for date in dates if date in self.investability_flag.index.get_level_values("date")
        }
        # Always use tqdm for progress
        dates_iter = tqdm(dates, desc="MVO weighting")
        date_to_pos = {d: pos for pos, d in enumerate(returns_wide.index)}
        for i, date in enumerate(dates_iter):
            pos = date_to_pos.get(date)
            if pos is None or pos == 0:
                continue
            start_pos = max(0, pos - window)
            hist = returns_wide.iloc[start_pos:pos]
            if len(hist) < 2:
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product([[date], carried.index],
                                                               names=["date", "symbol"])
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue
            # Use precomputed investable symbols
            investable = investable_dict.get(date, None)
            if investable is None or len(investable) == 0:
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product([[date], carried.index],
                                                               names=["date", "symbol"])
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue
            hist = hist[investable].dropna(axis=1, how="all")
            if len(hist.columns) < self.min_universe:
                if not prev_weights.empty:
                    carried = prev_weights.copy()
                    carried.index = pd.MultiIndex.from_product([[date], carried.index],
                                                               names=["date", "symbol"])
                    weights_list.append(carried)
                    count_list.append({"date": date, **prev_counts})
                continue
            mu  = hist.mean()
            cov = hist.cov()
            diag = np.diag(np.diag(cov.values))
            cov_shrink = (1.0 - shrink) * cov.values + shrink * diag
            # Use scipy.linalg.solve for w = inv_cov @ mu, fallback to pinv if needed
            try:
                w_raw = scipy.linalg.solve(cov_shrink, mu.values, assume_a='sym')
            except Exception:
                w_raw = np.linalg.pinv(cov_shrink).dot(mu.values)
            w_raw = pd.Series(w_raw, index=mu.index)
            w_pos = w_raw.clip(lower=0)
            w_neg = w_raw.clip(upper=0)
            if w_pos.sum() > 0:
                w_pos /= w_pos.sum()
            if w_neg.sum() < 0:
                w_neg /= -w_neg.sum()
            w = w_pos + w_neg  # w_neg already negative
            # Turnover penalty
            if not prev_weights.empty and 0 < turnover_lambda < 1:
                aligned_prev = prev_weights.reindex(w.index).fillna(0.0)
                w = (1.0 - turnover_lambda) * w + turnover_lambda * aligned_prev
                long_sum  = w[w > 0].sum()
                short_sum = -w[w < 0].sum()
                if long_sum > 0:
                    w[w > 0] /= long_sum
                if short_sum > 0:
                    w[w < 0] /= short_sum
            # SKIP cap and redistribute
            long_count  = int((w > 0).sum())
            short_count = int((w < 0).sum())
            prev_weights = w
            prev_counts  = {"long_count": long_count, "short_count": short_count}
            w.index = pd.MultiIndex.from_product([[date], w.index], names=["date", "symbol"])
            weights_list.append(w)
            count_list.append({"date": date,
                               "long_count": long_count,
                               "short_count": short_count})
        if not weights_list:
            return pd.Series(dtype=float), pd.DataFrame(columns=["long_count", "short_count"])
        all_w = pd.concat(weights_list).sort_index()
        shifted = all_w.groupby("symbol").shift(1)  # one-day lag – no look-ahead
        counts_df = pd.DataFrame(count_list).set_index("date")
        return shifted, counts_df

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