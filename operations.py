import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

# ========== Time-Series Operations ==========
def ts_sum(series: pd.Series, window: int) -> pd.Series:
    return series.groupby(level="symbol").transform(lambda x: x.rolling(window).sum())


def ts_mean(series: pd.Series, window: int) -> pd.Series:
    return series.groupby(level="symbol").transform(lambda x: x.rolling(window).mean())


def ts_std(series: pd.Series, window: int) -> pd.Series:
    return series.groupby(level="symbol").transform(lambda x: x.rolling(window).std())


def ts_zscore(series: pd.Series, window: int) -> pd.Series:
    return series.groupby(level="symbol").transform(
        lambda x: (x - x.rolling(window).mean()) / x.rolling(window).std().replace(0, np.nan)
    )

def ts_rank(series: pd.Series, window: int) -> pd.Series:
    def _fractional_rank(x: pd.Series) -> float:
        # x is a length‐`window` slice; rank returns 0–1
        return x.rank(pct=True).iloc[-1]
    return (
        series
        .groupby(level="symbol")
        .transform(lambda x: x.rolling(window, min_periods=window)
                             .apply(_fractional_rank, raw=False))
    )

def ts_diff(series: pd.Series, window: int) -> pd.Series:
    return series.groupby(level="symbol").transform(lambda x: x.diff(window))

def ts_delay(series: pd.Series, window: int) -> pd.Series:
    return series.groupby(level="symbol").transform(lambda x: x.shift(window))

def ts_decay(series: pd.Series, window: int) -> pd.Series:
    if window < 1:
        return series

    def decay_func(w):
        weights = np.arange(1, len(w) + 1)
        return np.dot(w, weights) / weights.sum()

    return series.groupby(level="symbol").transform(lambda x: x.rolling(window=window, min_periods=window).apply(decay_func, raw=True))

def ts_backfill(series: pd.Series) -> pd.Series:
    return series.groupby(level="symbol").transform(lambda x: x.ffill())

# ========== Math ==========
def cs_rank(series: pd.Series, method="average") -> pd.Series:
    """
    Cross-sectional rank normalized to [0, 1] per date.
    """
    def normalize_rank(x):
        r = x.rank(method=method)
        return (r - 1) / (len(r) - 1) if len(r) > 1 else 0.5  # single element edge case

    return series.groupby(level="date").transform(normalize_rank)

def cs_winsor(series: pd.Series, limits=(0.01, 0.99)) -> pd.Series:
    return series.groupby(level="date").transform(
        lambda x: x.clip(lower=x.quantile(limits[0]), upper=x.quantile(limits[1]))
        if x.notna().sum() >= 5 else x
    )

def cs_filter_center(series: pd.Series, center=(0.3, 0.7)) -> pd.Series:
    def filter_func(x):
        lower_val = x.quantile(center[0])
        upper_val = x.quantile(center[1])
        return x.where((x < lower_val) | (x > upper_val), 0)
    return series.groupby(level="date").transform(filter_func)

def cs_zscore(series: pd.Series) -> pd.Series:
    return series.groupby(level="date").transform(lambda x: (x - x.mean()) / x.std(ddof=0))

def cs_bool(condition: pd.Series, true_value: float, false_value: float) -> pd.Series:
    return pd.Series(
        np.where(condition, true_value, false_value),
        index=condition.index
    )
def cs_mean(series: pd.Series) -> pd.Series:
    return series.groupby(level="date").transform(lambda x: x.mean())

def sign(series: pd.Series) -> pd.Series:
    return np.sign(series)

def power(series: pd.Series, exp: float) -> pd.Series:
    return np.power(series, exp)

def log(series: pd.Series) -> pd.Series:
    return np.log(series)

def abs_(series: pd.Series) -> pd.Series:
    return np.abs(series)

def clip(series: pd.Series, lower, upper) -> pd.Series:
    return series.clip(lower, upper)

# ========== Group Operations ==========
def bucket(series: pd.Series, bin_range=(0.2, 1.0, 0.2)) -> pd.Series:
    low, up, step = bin_range
    bin_edges = np.arange(low, up + 1e-8, step)
    bin_labels = [f"group{i+1}" for i in range(len(bin_edges) - 1)]
    def _bucket_one_day(x):
        return pd.cut(x, bins=bin_edges, labels=bin_labels, include_lowest=True)
    return series.groupby(level="date", group_keys=False).apply(_bucket_one_day)

def group_mean(series: pd.Series, group: pd.Series) -> pd.Series:
    """
    Compute the group mean within each date and group, skipping NaNs.
    """
    # align into a DataFrame for grouping
    df = pd.DataFrame({"val": series, "group": group})
    # for each (date, group) compute mean of 'val' ignoring NaNs
    out = df.groupby(
        [series.index.get_level_values("date"), "group"]
    )["val"].transform(lambda x: x.mean(skipna=True))
    return out

def group_neutralize(series: pd.Series, group: pd.Series) -> pd.Series:
    """
    Subtract the group mean (ignoring NaNs) within each date and group.
    """
    df = pd.DataFrame({"val": series, "group": group})
    # compute mean skipping NaNs
    def demean(x):
        mu = x.mean(skipna=True)
        return x - mu
    out = df.groupby([series.index.get_level_values("date"), "group"])['val'].transform(demean)
    return out


def group_normalize(series: pd.Series, group: pd.Series) -> pd.Series:
    """
    Z-score normalize the series within each date and group, skipping NaNs.
    """
    df = pd.DataFrame({"val": series, "group": group})
    def zscore(x):
        mu = x.mean(skipna=True)
        sigma = x.std(skipna=True, ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0, index=x.index)
        return (x - mu) / sigma
    out = df.groupby([series.index.get_level_values("date"), "group"])['val'].transform(zscore)
    return out


def group_rank_normalized(series: pd.Series, group: pd.Series, method="average") -> pd.Series:
    """
    Compute cross-sectional rank within each date and group, normalized to [0,1], skipping NaNs.
    """
    df = pd.DataFrame({"val": series, "group": group})
    def normalize_rank(x):
        valid = x.dropna()
        if len(valid) <= 1:
            return pd.Series(0.5, index=x.index)
        r = valid.rank(method=method)
        norm = (r - 1) / (len(r) - 1)
        # reindex to original
        out = pd.Series(np.nan, index=x.index)
        out.loc[valid.index] = norm
        return out
    out = df.groupby([series.index.get_level_values("date"), "group"])['val'].transform(normalize_rank)
    return out


def market_neutralize(series: pd.Series) -> pd.Series:
    """
    Z-score normalize across the full cross-section for each date, skipping NaNs.
    """
    def zscore(x):
        mu = x.mean(skipna=True)
        sigma = x.std(skipna=True, ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0, index=x.index)
        return (x - mu) / sigma
    out = series.groupby(level="date").transform(zscore)
    return out


def ts_regression_fast(y: pd.Series, x: pd.Series, window: int, lag: int = 0, rettype: int = 2) -> pd.Series:
    """
    Fast rolling regression of y ~ x by symbol, via closed‐form cov/var.

    Parameters:
      y, x : pd.Series with MultiIndex [date, symbol]
      window : int
      lag : int
      rettype: 0=resid,1=alpha,2=beta,3=fitted,6=R2, etc.

    Returns:
      pd.Series same index
    """
    # align & shift
    x = x.shift(lag)
    df = pd.DataFrame({'y': y, 'x': x}).dropna()

    out = []

    for sym, g in df.groupby(level='symbol'):
        g2 = g.droplevel('symbol')
        # rolling co‐stats
        # E[x], E[y], E[x^2], E[xy], E[y^2]
        mx = g2['x'].rolling(window).mean()
        my = g2['y'].rolling(window).mean()
        ex2 = g2['x'].rolling(window).mean().mul(g2['x'])
        # Actually need rolling mean of x^2:
        ex2 = g2['x'].pow(2).rolling(window).mean()
        exy = (g2['x'].mul(g2['y'])).rolling(window).mean()

        cov_xy = exy - mx.mul(my)
        var_x  = ex2 - mx.pow(2)

        beta    = cov_xy.div(var_x)
        alpha   = my - beta.mul(mx)
        fitted  = alpha + beta.mul(g2['x'])
        resid   = g2['y'] - fitted

        # R² = corr²
        # or R² = (cov_xy)**2 / (var_x * var_y)
        var_y = (g2['y'].pow(2).rolling(window).mean() - my.pow(2))
        r2   = cov_xy.pow(2).div(var_x.mul(var_y))

        # pick your output
        if rettype == 0:
            vals = resid
        elif rettype == 1:
            vals = alpha
        elif rettype == 2:
            vals = beta
        elif rettype == 3:
            vals = fitted
        elif rettype == 6:
            vals = r2
        else:
            raise ValueError("rettype not implemented")

        # re‐attach symbol level
        vals.index = pd.MultiIndex.from_product([[sym], vals.index], names=['symbol','date'])
        out.append(vals.dropna())

    return pd.concat(out).sort_index().swaplevel().sort_index()

def cs_regression(
    y: pd.Series,
    x: pd.Series,
    rettype: str = 'resid'
) -> pd.Series:

    df = pd.concat([y.rename('y'), x.rename('x')], axis=1)
    out = []

    for date, grp in df.groupby(level='date'):
        # drop the date level for regression
        data = grp.droplevel('date').dropna()
        symbols = grp.index.get_level_values('symbol')

        if len(data) < 2:
            # too few points → all NaN for this date
            vals = pd.Series(np.nan, index=symbols)
        else:
            mx = data['x'].mean()
            my = data['y'].mean()
            cov_xy = ((data['x']-mx)*(data['y']-my)).mean()
            var_x  = ((data['x']-mx)**2).mean()
            beta   = cov_xy/var_x
            alpha  = my - beta*mx
            fitted = alpha + beta*data['x']
            resid  = data['y'] - fitted
            var_y  = ((data['y']-my)**2).mean()
            r2     = cov_xy**2/(var_x*var_y)

            if rettype == 'resid':
                vals = resid
            elif rettype == 'beta':
                vals = pd.Series(beta, index=data.index)
            elif rettype == 'alpha':
                vals = pd.Series(alpha, index=data.index)
            elif rettype == 'fitted':
                vals = fitted
            elif rettype == 'r2':
                vals = pd.Series(r2, index=data.index)
            else:
                raise ValueError(f"ERROR: rettype={rettype}")

            # ensure every symbol is represented
            vals = vals.reindex(symbols)

        # restore MultiIndex for this date
        mi = pd.MultiIndex.from_product(
            [[date], symbols],
            names=['date','symbol']
        )
        vals.index = mi
        out.append(vals)

    result = pd.concat(out).sort_index()

    # **key**: reindex back onto exactly the y.index (fills any missing with NaN)
    return result.reindex(y.index)