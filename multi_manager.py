import pandas as pd
import numpy as np
import logging
from portfolio_simulation import Simulation, SimulationSettings

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def compute_manager_weights(factor_series, settings, name="manager"):
    """
    Given a factor (custom_feature) Series (MultiIndex: date, symbol),
    compute daily portfolio weights and counts using the Simulation class.
    Returns (weights, counts).
    """
    # Create a shallow copy of settings to avoid mutation
    if isinstance(settings, SimulationSettings):
        sim_settings = settings
    else:
        sim_settings = SimulationSettings(**settings)
    sim = Simulation(name=name, custom_feature=factor_series, settings=sim_settings)
    # Use the internal method to get weights and counts
    weights, counts = sim._daily_trade_list()
    return weights, counts


def compute_multimanager_weights(factors_df, factor_weights, settings):
    """
    For each manager (factor), compute daily weights and counts, then combine them using factor_weights.
    Returns (final_weights, final_counts):
      - final_weights: Series (MultiIndex: date, symbol) of final portfolio weights
      - final_counts: DataFrame (index: date, columns: long_count, short_count) of weighted average counts
    """
    manager_weight_dict = {}
    manager_count_dict = {}
    for fac in factor_weights.columns:
        if fac not in factors_df.columns:
            logger.warning(f"Factor {fac} not in factors_df, skipping.")
            continue
        fac_series = factors_df[fac].dropna()
        mgr_w, mgr_counts = compute_manager_weights(fac_series, settings, name=fac)
        manager_weight_dict[fac] = mgr_w
        manager_count_dict[fac] = mgr_counts

    all_dates = factor_weights.index
    all_symbols = factors_df.index.get_level_values("symbol").unique()
    combined_weights = []
    combined_counts = []
    for date in all_dates:
        daily_weights = pd.Series(0.0, index=all_symbols)
        long_count = 0.0
        short_count = 0.0
        for fac, fac_w in factor_weights.loc[date].items():
            if fac_w == 0 or fac not in manager_weight_dict:
                continue
            mgr_w = manager_weight_dict[fac]
            mgr_counts = manager_count_dict[fac]
            try:
                mgr_w_today = mgr_w.xs(date, level="date")
                mgr_counts_today = mgr_counts.loc[date]
            except (KeyError, IndexError):
                continue
            daily_weights = daily_weights.add(mgr_w_today * fac_w, fill_value=0)
            long_count += fac_w * mgr_counts_today["long_count"]
            short_count += fac_w * mgr_counts_today["short_count"]
        daily_weights.index = pd.MultiIndex.from_product([[date], daily_weights.index], names=["date", "symbol"])
        combined_weights.append(daily_weights)
        combined_counts.append({"date": date, "long_count": long_count, "short_count": short_count})
    if combined_weights:
        final_weights = pd.concat(combined_weights)
        final_weights = final_weights[final_weights != 0]
        final_counts = pd.DataFrame(combined_counts).set_index("date")
    else:
        final_weights = pd.Series(dtype=float)
        final_counts = pd.DataFrame(columns=["long_count", "short_count"])
    return final_weights, final_counts


def run_multimanager_backtest(factors_df, returns, cap_flag, factor_weights, settings):
    """
    Run the multimanager backtest and return the result DataFrame (like daily_portfolio_returns),
    plus top contributors and counts DataFrame.
    """
    logger.info("Computing multimanager portfolio weights and counts...")
    weights, counts = compute_multimanager_weights(factors_df, factor_weights, settings)
    logger.info("Running backtest...")
    # Prepare a Simulation object for the combined weights
    if isinstance(settings, SimulationSettings):
        sim_settings = settings
    else:
        sim_settings = SimulationSettings(**settings)
    # We need to pass the combined weights as the custom_feature, but use a special method to run returns
    sim = Simulation(name="multimanager", custom_feature=weights, settings=sim_settings)
    result, top_longs, top_shorts = sim._daily_portfolio_returns(weights)
    return result, top_longs, top_shorts, counts