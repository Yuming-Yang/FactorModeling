import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from scipy import stats
from factor_selection_methods import (
    icir_top_selector, 
    mvo_selector, 
    factor_momentum_selector
)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

FACTOR_SELECTION_METHODS = {
    'icir_top': icir_top_selector,
    'mvo': mvo_selector,
    'momentum': factor_momentum_selector,
}

def single_factor_metrics(factors_df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """
    Run single‐factor analysis on a MultiIndexed DataFrame.
    Returns a DataFrame of metrics sorted by rank_IC_IR descending.
    Clusters factors by their name prefix instead of hierarchical clustering.
    """
    factor_cols = [c for c in factors_df.columns]
    shifted = factors_df[factor_cols].groupby(level='symbol').shift(1)
    metrics = []

    for fac in factor_cols:
        pair = pd.concat([shifted[fac], returns], axis=1, keys=['f','r']).dropna()
        ic_list, rank_list, beta_list = [], [], []
        for _, grp in pair.groupby(level='date'):
            f = grp['f'].values
            r = grp['r'].values
            if len(f) < 3:
                continue
            ic_list.append(stats.pearsonr(f, r)[0])
            rank_list.append(stats.pearsonr(stats.rankdata(f), r)[0])
            denom = np.dot(f, f)
            if denom > 0:
                beta_list.append(np.dot(f, r) / denom)

        ic_arr   = np.array(ic_list);   ic_arr   = ic_arr[~np.isnan(ic_arr)]
        rank_arr = np.array(rank_list); rank_arr = rank_arr[~np.isnan(rank_arr)]
        beta_arr = np.array(beta_list); beta_arr = beta_arr[~np.isnan(beta_arr)]

        ic_mean   = ic_arr.mean() if ic_arr.size else np.nan
        ic_ir     = ic_mean / ic_arr.std(ddof=1) if ic_arr.size > 1 else np.nan
        rank_mean = rank_arr.mean() if rank_arr.size else np.nan
        rank_ir   = rank_mean / rank_arr.std(ddof=1) if rank_arr.size > 1 else np.nan
        tstat, pval = stats.ttest_1samp(beta_arr, 0) if beta_arr.size > 1 else (np.nan, np.nan)
        pct_pos    = np.mean(beta_arr > 0) if beta_arr.size else np.nan

        metrics.append({
            'factor': fac,
            'IC':                    ic_mean,
            'IC_IR':                 ic_ir,
            'rank_IC':               rank_mean,
            'rank_IC_IR':            rank_ir,
            'factor_return_tstat':   tstat,
            'factor_return_pvalue':  pval,
            'pct_pos_factor_return': pct_pos,
        })

    metrics_df = pd.DataFrame(metrics).set_index('factor')
    return metrics_df.sort_values('rank_IC_IR', ascending=False)

# --- Main Classes ---
class FactorSelector:
    """
    Selects factors over time based on a specified method and lookback window.
    """
    def __init__(self, factors_df: pd.DataFrame, returns: pd.Series, factor_ret_df: pd.DataFrame, window: int, method: str, method_kwargs: dict = None):
        logger.info(f"Initializing FactorSelector with method='{method}' and window={window}...")
        self.factors = factors_df
        self.factor_cols = [c for c in factors_df.columns]
        self.factors = factors_df.groupby(level='symbol').shift(1)
        self.returns = returns
        self.factor_ret_df = factor_ret_df
        self.window = window
        self.method = method
        self.method_kwargs = method_kwargs or {}
        self.factor_selection = None
        self.dates = sorted(list(set(self.factors.index.get_level_values("date")).intersection(set(self.factor_ret_df.index))))
        logger.info("FactorSelector initialized.")

    def prepare_selection(self) -> pd.DataFrame:
        """
        Iterates through time to select factors based on the chosen rolling window methodology.
        """
        if self.factor_selection is not None:
            logger.info("Factor selection already prepared. Returning cached result.")
            return self.factor_selection

        logger.info("Executing rolling factor selection...")
        proc_dates = self.dates[self.window : -1]
        
        all_vecs = []
        found_nonzero_mvo = False if self.method == 'mvo' else True  # Track if we've found non-zero weights for MVO
        
        for today in tqdm(proc_dates, desc="Selecting Factors"):
            idx = self.dates.index(today)
            window_dates = self.dates[max(0, idx - self.window): idx]
            
            # Slice data for the lookback window
            factor_ret_win = self.factor_ret_df.loc[window_dates].copy()
            factors_win = self.factors.loc[window_dates].copy()
            returns_win = self.returns.loc[window_dates].copy()
            
            # Get metrics for the window
            metrics_df = single_factor_metrics(factors_win, returns_win)
            
            # Dispatch to the appropriate selection function
            selector_func = FACTOR_SELECTION_METHODS.get(self.method)
            if selector_func is None:
                raise ValueError(f"Unknown factor selection method: {self.method}")
            vec = selector_func(metrics_df, factors_win, returns_win, factor_ret_win, today, window_dates, **self.method_kwargs)

            
            vec.name = today
            all_vecs.append(vec)

        # Combine daily selections into a single DataFrame
        selection_df = pd.concat(all_vecs, axis=1).T if all_vecs else pd.DataFrame()
        if not selection_df.empty:
            selection_df.index.name = "date"
            selection_df.columns.name = "factor"
            # Normalize weights to sum to 1 each day
            selection_df = selection_df.div(selection_df.sum(axis=1), axis=0).fillna(0)

        self.factor_selection = selection_df
        return selection_df