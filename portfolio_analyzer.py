import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

# This analyzer is for after-cost clean portfolio 
class PortfolioAnalyzer:
    def __init__(self, df: pd.DataFrame, trading_days_per_year: int = 252):
        # Clean and set up data
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.sort_values('date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.trading_days = trading_days_per_year
        self.df['return'] = np.exp(self.df['log_return']) - 1  # Convert log return to simple return
        self.df['cumulative_return'] = (1 + self.df['return']).cumprod() - 1  # Start from 0%

    def average_return(self):
        return self.df['return'].mean()

    def daily_volatility(self):
        return self.df['return'].std()

    def yearly_volatility(self):
        return self.daily_volatility() * np.sqrt(self.trading_days)

    def annualized_return(self):
        total_days = (self.df['date'].iloc[-1] - self.df['date'].iloc[0]).days
        total_years = total_days / 365.25
        final_value = self.df['cumulative_return'].iloc[-1] + 1
        return final_value**(1 / total_years) - 1

    def sharpe_ratio(self, risk_free_rate=0):
        excess_return = self.df['return'] - risk_free_rate / self.trading_days
        return (excess_return.mean() / excess_return.std()) * np.sqrt(self.trading_days)

    def sortino_ratio(self, risk_free_rate=0):
        excess_return = self.df['return'] - risk_free_rate / self.trading_days
        downside_std = excess_return[excess_return < 0].std()
        return (excess_return.mean() / downside_std) * np.sqrt(self.trading_days)

    def max_drawdown(self):
        cum = self.df['cumulative_return'] + 1
        peak = cum.cummax()
        drawdown = cum / peak - 1
        return drawdown.min()

    def max_drawdown_curve(self):
        cum = self.df['cumulative_return'] + 1
        peak = cum.cummax()
        return cum / peak - 1

    def max_daily_return(self):
        return self.df['return'].max()

    def min_daily_return(self):
        return self.df['return'].min()

    def monthly_return(self):
        monthly = self.df.set_index('date')['return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        return monthly

    def yearly_return(self):
        yearly = self.df.set_index('date')['return'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
        return yearly

    def summary(self):
        return {
        'Average Daily Return': f"{round(self.average_return() * 100, 2)}%",
        'Annualized Return': f"{round(self.annualized_return() * 100, 2)}%",
        'Yearly Volatility': f"{round(self.yearly_volatility() * 100, 2)}%",
        'Max Daily Return': f"{round(self.max_daily_return() * 100, 2)}%",
        'Sharpe Ratio': round(self.sharpe_ratio(), 2),
        'Sortino Ratio': round(self.sortino_ratio(), 2),
        'Max Drawdown': f"{round(self.max_drawdown() * 100, 2)}%",
        'Min Daily Return': f"{round(self.min_daily_return() * 100, 2)}%",

    }

    def plot_full_performance(self, counts_df: pd.DataFrame = None):
        # 1) compute all series

        self.df['log_return_ma120'] = self.df['log_return'].rolling(120).mean()
        self.df['log_return_ma252'] = self.df['log_return'].rolling(252).mean()
        if 'long_return' in self.df.columns:
            self.df['cumulative_long_return'] = self.df['long_return'].fillna(0).cumsum().apply(np.exp) - 1
        if 'short_return' in self.df.columns:
            self.df['cumulative_short_return'] = self.df['short_return'].fillna(0).cumsum().apply(np.exp) - 1

        monthly_return_df = (
            self.df.set_index('date')['return']
            .resample('M')
            .apply(lambda x: (1 + x).prod() - 1)
            .dropna()
        )

        has_turnover = 'turnover' in self.df.columns
        has_counts = counts_df is not None and not counts_df.empty
        has_leg_turnover = 'long_turnover' in self.df.columns or 'short_turnover' in self.df.columns

        # 2) build GridSpec
        n_rows = 4 + int(has_turnover) + int(has_counts)
        height_ratios = [0.6, 2, 0.8, 0.8]
        if has_turnover: height_ratios.append(0.8)
        if has_counts:   height_ratios.append(0.8)

        fig = plt.figure(figsize=(14, 4 * n_rows))
        gs  = GridSpec(n_rows, 1, height_ratios=height_ratios, hspace=0.3)

        # 3) 
        ax_txt = fig.add_subplot(gs[0, :])
        ax_txt.axis('off')
        stats = self.summary()
        items = list(stats.items())
        mid   = len(items) // 2
        left  = items[:mid]
        right = items[mid:]

        # build 3x4 table_data
        table_data = [
            [lm, lv, rm, rv]
            for (lm, lv), (rm, rv) in zip(left, right)
        ]
        col_labels = ['Metric', 'Value', 'Metric', 'Value']

        tbl = ax_txt.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            colLoc='center',
            loc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1, 1.5)

        ax_main = fig.add_subplot(gs[1, :])
        ax_ret = ax_main.twinx()

        ax_main.plot(self.df['date'], self.df['cumulative_return'], label='Total', color='black')
        ax_main.plot(self.df['date'], self.max_drawdown_curve(), label='Max Drawdown Curve', color='red', linestyle='--')
        
        if 'cumulative_long_return' in self.df.columns:
            ax_main.plot(self.df['date'], self.df['cumulative_long_return'], label='Long Leg', color='green', linestyle=':')
        if 'cumulative_short_return' in self.df.columns:
            ax_main.plot(self.df['date'], self.df['cumulative_short_return'], label='Short Leg', color='orange', linestyle='-.')

        ax_main.set_ylabel('Cumulative Return')
        ax_main.set_title('Cumulative Return (Total / Long / Short) with Monthly Bars')
        ax_main.legend(loc='upper left')
        ax_main.grid(True)
        ax_main.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        bar_colors = ['green' if r >= 0 else 'red' for r in monthly_return_df]
        ax_ret.bar(monthly_return_df.index, monthly_return_df.values, width=20, color=bar_colors, alpha=0.4)
        ax_ret.set_ylabel('Monthly Return', color='gray')
        ax_ret.tick_params(axis='y', labelcolor='gray')
        ax_ret.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        ax_ma = fig.add_subplot(gs[2, :], sharex=ax_main)

        # Filled‐area curves for 120d and 250d rolling MA of daily returns
        ax_ma.fill_between(
            self.df['date'],
            self.df['log_return_ma120'],
            color='darkred',
            alpha=0.5,
            label='120d MA'
        )
        ax_ma.fill_between(
            self.df['date'],
            self.df['log_return_ma252'],
            color='navy',
            alpha=0.5,
            label='250d MA'
        )

        ax_ma.set_ylabel('MA(Return)')
        ax_ma.set_title('Rolling MA of Daily Returns')
        ax_ma.legend(loc='upper left')
        ax_ma.grid(True)

        ax_ma.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax_ma.xaxis.set_major_locator(mdates.YearLocator())
        ax_ma.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # 5) Turnover chart (optional)
        current_row = 3
        if has_turnover:
            ax_turnover = fig.add_subplot(gs[current_row, :], sharex=ax_main)
            avg_turnover = self.df['turnover'].mean()
            mask = self.df['turnover'] > 1.5
            if mask.any():
                first_day = mask.idxmax()   # label of the first True
                self.df.loc[first_day, ['turnover','long_turnover','short_turnover']] = 0
            ax_turnover.plot(self.df['date'], self.df['turnover'], color='purple', linewidth=1.2, label='Total Turnover')
            
            if has_leg_turnover:
                if 'long_turnover' in self.df.columns:
                    ax_turnover.plot(self.df['date'], self.df['long_turnover'], color='green', linestyle='--', label='Long Turnover')
                if 'short_turnover' in self.df.columns:
                    ax_turnover.plot(self.df['date'], self.df['short_turnover'], color='red', linestyle='--', label='Short Turnover')

            ax_turnover.axhline(avg_turnover, color='gray', linestyle=':', linewidth=1.2, label=f'Avg: {avg_turnover:.2%}')
            ax_turnover.set_ylabel('Turnover')
            ax_turnover.set_title('Portfolio Turnover (Total / Long / Short)')
            ax_turnover.legend(loc='upper right')
            ax_turnover.grid(True)
            ax_turnover.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            current_row += 1

        if has_counts:
            ax_counts = fig.add_subplot(gs[current_row, :], sharex=ax_main)
            ax_counts.plot(counts_df.index, counts_df['long_count'], label='Long Count', color='green')
            ax_counts.plot(counts_df.index, counts_df['short_count'], label='Short Count', color='red')
            ax_counts.set_title("Number of Symbols in Long and Short Legs Over Time")
            ax_counts.set_xlabel("Date")
            ax_counts.set_ylabel("Count")
            ax_counts.legend()
            ax_counts.grid(True)
            current_row += 1
        
        ax3 = fig.add_subplot(gs[current_row, :], sharex=ax_main)
        # compute both windows
        for w, col in zip([120, 252], ['darkred', 'navy']):
            # use log_return to compute Sharpe:
            sr = (self.df['log_return']
                .rolling(w)
                .mean() /
                self.df['log_return']
                .rolling(w)
                .std()
                * np.sqrt(252))
            ax3.plot(self.df['date'],
                    sr,
                    label=f'{w}d Sharpe',
                    color=col,
                    linewidth=1.5)

        # horizontal lines at integer Sharpe levels
        yticks = ax3.get_yticks()
        for y in yticks:
            if abs(y - round(y)) < 1e-6:
                ax3.axhline(y,
                            color='gray',
                            linestyle='--',
                            linewidth=0.5,
                            alpha=0.3)

        ax3.set_title('Rolling Sharpe Ratios')
        ax3.set_ylabel('Sharpe')
        ax3.set_xlabel('Date')
        ax3.legend(loc='upper left', fontsize='small')
        ax3.grid(True)

        plt.show()