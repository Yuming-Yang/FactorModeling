import os
import glob
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.cm as cm
# from tqdm import tqdm  # Commented out to avoid progress bar issues in Jupyter
import matplotlib.dates as mdates
from portfolio_analyzer import *
from portfolio_simulation import *

def plot_factor_distributions(
    factors_df: pd.DataFrame,
    exclude: list = None,
    bins: int = 50,
    ncols: int = 3,
    figsize: tuple = (15, 5)
):

    exclude = exclude or []
    cols = [c for c in factors_df.columns if c not in exclude]
    n = len(cols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    axes = axes.flatten()
    
    for ax, col in zip(axes, cols):
        data = factors_df[col].dropna()
        ax.hist(data, bins=bins, density=True, alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
    
    # turn off any unused subplots
    for ax in axes[n:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_quantile_backtests_log(com_factors_df: pd.DataFrame,
                                returns: pd.Series,
                                n_groups: int = 5,
                                ncols: int = 2,
                                figsize: tuple = (20, 6)):
    """
    Pure-log version quantile backtest + plot:
      • Input `returns` must be log-returns: ln(1 + r).
      • For each factor column in com_factors_df (MultiIndex [date, symbol]):
        1) Assign cross-sectional quantile labels 1..n_groups (1 = top).
        2) Shift labels by 1 day per symbol.
        3) Compute average log-return per (date, group).
        4) Cumulate in log-space, then convert back to simple returns via expm1().
        5) Plot each bucket’s cumulative P&L and the Long1–ShortN spread (in black).
      • Arranges subplots in a grid with `ncols` per row.
    """
    def quantile_backtest_log(feature: pd.Series) -> pd.DataFrame:
        # 1) raw 0..n_groups-1 labels (0=bottom)
        lbl0 = (feature
                .groupby(level="date")
                .transform(lambda x: pd.qcut(x.rank(method="first"),
                                             n_groups,
                                             labels=False,
                                             duplicates="drop")))
        # 2) invert to 1=top … n_groups=bottom
        q = (n_groups - lbl0).astype("Int64")
        # 3) shift per symbol
        q1 = q.groupby(level="symbol").shift(1)
        # 4) build DF, drop any missing
        df = (pd.DataFrame({
                  "log_ret": returns,
                  "group":    q1
              })
              .dropna(subset=["group", "log_ret"]))
        # 5) avg log-ret per (date, group)
        grp_log = (df
                   .reset_index()
                   .groupby(["date", "group"])["log_ret"]
                   .mean()
                   .unstack(level="group")
                   .sort_index())
        # ensure all groups present
        return grp_log.reindex(columns=range(1, n_groups + 1))

    factors = com_factors_df.columns.tolist()
    n = len(factors)
    nrows = math.ceil(n / ncols)
    
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize[0], figsize[1] * nrows),
                             squeeze=False)

    for idx, fac in enumerate(factors):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        
        # run log-backtest
        grp_log = quantile_backtest_log(com_factors_df[fac])
        # cumulative simple P&L
        cum = np.expm1(grp_log.cumsum())
        # build LS spread
        ls_log = grp_log[1] - grp_log[n_groups]
        cum[f"DN_L1-S{n_groups}"] = np.expm1(ls_log.cumsum())
        
        # plot: buckets in default colors, LS in black
        for line in cum.columns:
            if str(line).startswith("DN_L1-S"):
                ax.plot(cum.index, cum[line],
                        label=line,
                        color="black",
                        linewidth=2)
            else:
                ax.plot(cum.index, cum[line], label=line)
        
        ax.set_title(fac)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend(loc="upper left", fontsize="small")
        ax.grid(True)

    # remove any empty subplots (when n is odd)
    total = nrows * ncols
    for empty_idx in range(n, total):
        r, c = divmod(empty_idx, ncols)
        fig.delaxes(axes[r][c])

    plt.tight_layout()
    plt.show()


def composite_factor_calculation(factors_df: pd.DataFrame,selected_factors: list,method: str = 'zscore'):
    """
    1) Preprocess suffix‐based factors per date:
       - '_eq': bottom/top 10% → -1/1, middle → 0
       - '_flx': winsorize at 2%/98%, then scale to [-1,1]
       - '_long': winsorize at 2%/98%, then scale to [0,1]
       - '_short': winsorize at 2%/98%, then scale to [–1,0]
    2) Group factors by prefix before underscore
    3) Build group‐mean proxies
    4) Cross‐sectionally transform proxies per date using `method`
    5) Combine proxies and neutralize (demean) composite per date

    Returns:
        composite: pd.Series of composite factor
    """
    # Use all selected_factors, no ICIR filter
    pos = selected_factors
    adj = factors_df[pos].copy()

    # 2) guarded preprocessing for _eq, _flx, _long, _short
    def preprocess(group):
        for suffix, (q_low, q_high, scale_fn) in {
            '_eq':   (10, 90, lambda s, l, h: np.where(s<=l, -1, np.where(s>=h, 1, 0))),
            '_flx':  ( 2, 98, lambda s, l, h: ((np.clip(s, l, h)-l)/(h-l))*2 - 1),
            '_long': ( 2, 98, lambda s, l, h: (np.clip(s, l, h)-l)/(h-l)         ),
            '_short':( 2, 98, lambda s, l, h: (np.clip(s, l, h)-h)/(h-l)         )
        }.items():
            cols = [c for c in group.columns if c.endswith(suffix)]
            for c in cols:
                arr = group[c].values
                clean = arr[~np.isnan(arr)]
                if clean.size == 0:
                    group[c] = 0
                    continue
                low, high = np.nanpercentile(clean, [q_low, q_high])
                if high == low:
                    group[c] = 0
                else:
                    group[c] = scale_fn(arr, low, high)
        return group

    adj = adj.groupby(level='date', group_keys=False).apply(preprocess)

    # 3) group fac
    prefix_to_facs = defaultdict(list)
    for fac in pos:
        prefix = fac.split('_', 1)[0]
        prefix_to_facs[prefix].append(fac)

    # 4) build group proxies
    proxies_df = pd.DataFrame({
        f'group_{prefix}': adj[facs].mean(axis=1)
        for prefix, facs in prefix_to_facs.items()
    }, index=factors_df.index)

    # 5) safe cross-sectional transform
    if method == 'zscore':

        def safe_zcol(x: pd.Series) -> pd.Series:
            mu = x.mean()
            sigma = x.std(ddof=0)
            if sigma == 0 or np.isnan(sigma):
                return pd.Series(0, index=x.index)
            return (x - mu) / sigma

        # This ensures safe_zcol sees *one column* at a time:
        proxies_norm = proxies_df.groupby(level='date').transform(safe_zcol)
        composite = proxies_norm.mean(axis=1)

    elif method == 'rank':
        proxy_ranks = proxies_df.groupby(level='date').transform(
            lambda x: (stats.rankdata(x) - 1) / (len(x) - 1)
        )
        composite = proxy_ranks.sum(axis=1)

    else:
        raise ValueError("method must be 'zscore' or 'rank'")

    # 5) Demean per date
    composite = composite.groupby(level='date').transform(lambda x: x - x.mean())
    composite.name = 'composite_factor'
    return composite

def weighted_composite_factor(
    factors_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    method: str = 'zscore'
) -> pd.Series:
    """
    Compute a composite factor using daily factor weights (selection_df),
    following the same preprocessing and logic as composite_factor_calculation.

    Parameters
    ----------
    factors_df : pd.DataFrame
        MultiIndex (date, symbol) -> factor values. Columns = factor names.
    selection_df : pd.DataFrame
        Index = date; columns = factor names; values = weights (sum to 1 per date).
    method : {'zscore', 'rank'}
        How to cross-sectionally transform group proxies per date.

    Returns
    -------
    pd.Series
        MultiIndex (date, symbol) composite factor, name='composite_factor'.
        Only includes dates that are present in selection_df.
    """
    suffix_map = {
        '_eq':   (10, 90, lambda s, lo, hi: np.where(s <= lo, -1, np.where(s >= hi, 1, 0))),
        '_flx':  ( 2, 98, lambda s, lo, hi: ((np.clip(s, lo, hi) - lo) / (hi - lo)) * 2 - 1),
        '_long': ( 2, 98, lambda s, lo, hi: (np.clip(s, lo, hi) - lo) / (hi - lo)),
        '_short':( 2, 98, lambda s, lo, hi: (np.clip(s, lo, hi) - hi) / (hi - lo)),
    }

    def preprocess_suffix(df_row: pd.DataFrame) -> pd.DataFrame:
        out = df_row.copy()
        for suffix, (q_lo, q_hi, fn) in suffix_map.items():
            cols = [c for c in out.columns if c.endswith(suffix)]
            if not cols:
                continue
            vals = out[cols].values
            clean = vals[~np.isnan(vals)]
            if clean.size == 0:
                out[cols] = 0
                continue
            lo, hi = np.nanpercentile(clean, [q_lo, q_hi])
            if lo == hi:
                out[cols] = 0
            else:
                for c in cols:
                    out[c] = fn(out[c].values, lo, hi)
        return out

    def safe_z(x: pd.Series) -> pd.Series:
        mu, sigma = x.mean(), x.std(ddof=0)
        return (x - mu) / sigma if sigma and not np.isnan(sigma) else pd.Series(0, index=x.index)

    def safe_rank(x: pd.Series) -> pd.Series:
        r = stats.rankdata(x, method='average')
        return pd.Series((r - 1) / (len(r) - 1), index=x.index)

    pieces = []
    for date, weights in selection_df.iterrows():
        # Only keep factors with nonzero weight
        today_factors = weights[weights > 0].index.tolist()
        syms = factors_df.loc[date].index if date in factors_df.index.get_level_values('date') else []
        if not today_factors or len(syms) == 0:
            idx = pd.MultiIndex.from_product([[date], syms], names=['date', 'symbol'])
            pieces.append(pd.Series(np.nan, index=idx, name='composite_factor'))
            continue

        today_df = factors_df.loc[date, today_factors]
        adj = preprocess_suffix(today_df)

        # group by prefix -> proxies
        prefix_groups = defaultdict(list)
        for fac in today_factors:
            prefix = fac.split('_', 1)[0]
            prefix_groups[prefix].append(fac)

        proxies = pd.DataFrame({
            f'group_{p}': adj[facs].mean(axis=1)
            for p, facs in prefix_groups.items()
        }, index=adj.index)

        if proxies.empty:
            idx = pd.MultiIndex.from_product([[date], syms], names=['date', 'symbol'])
            pieces.append(pd.Series(np.nan, index=idx, name='composite_factor'))
            continue

        # Compute group weights: sum weights of factors in each group
        group_weights = {}
        for p, facs in prefix_groups.items():
            group_weights[f'group_{p}'] = weights[facs].sum()
        # Normalize group weights to sum to 1 (if not all zero)
        gw_sum = sum(group_weights.values())
        if gw_sum > 0:
            for k in group_weights:
                group_weights[k] /= gw_sum
        else:
            # fallback: equal weights
            n = len(group_weights)
            group_weights = {k: 1/n for k in group_weights}

        # Cross-sectional transform
        if method == 'zscore':
            normed = proxies.apply(safe_z, axis=0)
        elif method == 'rank':
            normed = proxies.apply(safe_rank, axis=0)
        else:
            raise ValueError("method must be 'zscore' or 'rank'")

        # Weighted sum of group proxies
        comp = sum(normed[col] * group_weights[col] for col in proxies.columns)
        # Demean per date
        comp = comp - comp.mean()
        comp.name = 'composite_factor'

        idx = pd.MultiIndex.from_product([[date], comp.index], names=['date', 'symbol'])
        pieces.append(pd.Series(comp.values, index=idx, name='composite_factor'))

    result = pd.concat(pieces)
    result.index.names = ['date', 'symbol']
    # Ensure the output index matches the full factors_df index
    result = result.reindex(factors_df.index).fillna(0)
    return result