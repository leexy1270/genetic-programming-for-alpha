"""
factor_analysis.py — Factor Performance Analysis Module

Usage:
    from factor_analysis import FactorAnalyzer
    analyzer = FactorAnalyzer(factor_values, forward_returns, trade_dates, stock_codes)
    analyzer.report()  # print summary
    analyzer.plot_all()  # generate analysis charts
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import os

os.makedirs("result", exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = ['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c', '#1f77b4']


class FactorAnalyzer:
    """Comprehensive factor performance analyzer."""

    def __init__(self, factor_values, forward_returns, trade_dates=None, stock_codes=None,
                 factor_name="Factor"):
        """
        Parameters
        ----------
        factor_values : np.ndarray, shape (n_stocks, n_dates) or pd.DataFrame
        forward_returns : np.ndarray, shape (n_stocks, n_dates) or pd.DataFrame
        trade_dates : list-like, optional
        stock_codes : list-like, optional
        factor_name : str
        """
        if isinstance(factor_values, pd.DataFrame):
            self.fv = factor_values.values
            if trade_dates is None:
                trade_dates = factor_values.index
            if stock_codes is None:
                stock_codes = factor_values.columns
        else:
            self.fv = factor_values

        if isinstance(forward_returns, pd.DataFrame):
            self.fwd = forward_returns.values
        else:
            self.fwd = forward_returns

        self.dates = trade_dates
        self.codes = stock_codes
        self.name = factor_name
        self.n_stocks, self.n_dates = self.fv.shape

    # ===== IC Analysis =====

    def compute_ic_series(self):
        """Compute cross-sectional Rank IC for each date."""
        ic_list = []
        for t in range(self.n_dates):
            fv_cross = self.fv[:, t]
            fwd_cross = self.fwd[:, t]
            mask = ~(np.isnan(fv_cross) | np.isnan(fwd_cross) | np.isinf(fv_cross) | np.isinf(fwd_cross))
            if mask.sum() < 10:
                ic_list.append(np.nan)
                continue
            ic = stats.spearmanr(fv_cross[mask], fwd_cross[mask])[0]
            ic_list.append(ic if not np.isnan(ic) else np.nan)
        return np.array(ic_list)

    def ic_summary(self):
        """IC statistics: mean, std, ICIR, positive ratio, t-stat."""
        ic = self.compute_ic_series()
        valid = ic[~np.isnan(ic)]
        if len(valid) == 0:
            return {'IC_Mean': np.nan, 'IC_Std': np.nan, 'ICIR': np.nan,
                    'Pos_Ratio': np.nan, 't_stat': np.nan, 'N': 0}
        return {
            'IC_Mean': np.mean(valid),
            'IC_Std': np.std(valid, ddof=1),
            'ICIR': np.mean(valid) / np.std(valid, ddof=1) if np.std(valid, ddof=1) > 1e-12 else 0,
            'Pos_Ratio': np.mean(valid > 0),
            't_stat': np.mean(valid) / (np.std(valid, ddof=1) / np.sqrt(len(valid))),
            'N': len(valid),
        }

    # ===== Quantile Analysis =====

    def quintile_analysis(self):
        """Quintile returns (Q0=lowest factor, Q4=highest)."""
        records = []
        for t in range(self.n_dates):
            fv_cross = self.fv[:, t]
            fwd_cross = self.fwd[:, t]
            mask = ~(np.isnan(fv_cross) | np.isnan(fwd_cross) | np.isinf(fv_cross) | np.isinf(fwd_cross))
            if mask.sum() < 15:
                continue
            try:
                groups = pd.qcut(fv_cross[mask], 5, labels=False, duplicates='drop')
                for q in range(5):
                    q_ret = fwd_cross[mask][groups == q]
                    if len(q_ret) > 0:
                        records.append({'date': self.dates[t] if self.dates is not None else t,
                                        'quintile': q, 'mean_ret': q_ret.mean(),
                                        'n_stocks': len(q_ret)})
            except Exception:
                continue
        return pd.DataFrame(records)

    def long_short_returns(self, top_quantile=0.2, bottom_quantile=0.2):
        """Long-short portfolio returns (top vs bottom quantile)."""
        ls_returns = []
        for t in range(self.n_dates):
            fv_cross = self.fv[:, t]
            fwd_cross = self.fwd[:, t]
            mask = ~(np.isnan(fv_cross) | np.isnan(fwd_cross) | np.isinf(fv_cross) | np.isinf(fwd_cross))
            if mask.sum() < 15:
                continue
            n_top = max(1, int(mask.sum() * top_quantile))
            n_bot = max(1, int(mask.sum() * bottom_quantile))
            order = np.argsort(fv_cross[mask])
            top_ret = fwd_cross[mask][order[-n_top:]].mean()
            bot_ret = fwd_cross[mask][order[:n_bot]].mean()
            ls_returns.append({'date': self.dates[t] if self.dates is not None else t,
                               'long': top_ret, 'short': bot_ret, 'ls': top_ret - bot_ret})
        return pd.DataFrame(ls_returns)

    # ===== Turnover Analysis =====

    def factor_turnover(self, top_quantile=0.2):
        """Average daily turnover of top quantile."""
        turnovers = []
        prev_top = None
        for t in range(self.n_dates):
            fv_cross = self.fv[:, t]
            mask = ~(np.isnan(fv_cross) | np.isinf(fv_cross))
            if mask.sum() < 15:
                continue
            n_top = max(1, int(mask.sum() * top_quantile))
            order = np.argsort(fv_cross[mask])
            top_stocks = set(np.where(mask)[0][order[-n_top:]])
            if prev_top is not None and len(prev_top) > 0:
                overlap = len(top_stocks & prev_top)
                turnover = 1.0 - overlap / len(top_stocks)
                turnovers.append(turnover)
            prev_top = top_stocks
        return np.mean(turnovers) if turnovers else np.nan

    # ===== Decay Analysis =====

    def decay_analysis(self, max_lag=20):
        """Factor autocorrelation decay (factor persistence)."""
        # Compute daily factor mean across stocks
        factor_mean = np.nanmean(self.fv, axis=0)
        valid = ~np.isnan(factor_mean)
        autocorr = [1.0]
        for lag in range(1, max_lag + 1):
            if len(factor_mean[valid]) <= lag:
                autocorr.append(np.nan)
                continue
            ac = np.corrcoef(factor_mean[valid][:-lag], factor_mean[valid][lag:])[0, 1]
            autocorr.append(ac)
        return np.array(autocorr)

    # ===== Reporting =====

    def report(self):
        """Print comprehensive factor analysis report."""
        print(f"\n{'='*60}")
        print(f"  Factor Analysis Report: {self.name}")
        print(f"{'='*60}")

        # IC summary
        ic_sum = self.ic_summary()
        print(f"\n  --- IC Statistics ---")
        print(f"  IC Mean:     {ic_sum['IC_Mean']:.6f}")
        print(f"  IC Std:      {ic_sum['IC_Std']:.6f}")
        print(f"  ICIR:        {ic_sum['ICIR']:.4f}")
        print(f"  Pos Ratio:   {ic_sum['Pos_Ratio']:.2%}")
        print(f"  t-stat:      {ic_sum['t_stat']:.2f}")
        print(f"  N (days):    {ic_sum['N']}")

        # Quintile
        qdf = self.quintile_analysis()
        if not qdf.empty:
            q_means = qdf.groupby('quintile')['mean_ret'].mean()
            print(f"\n  --- Quintile Returns ---")
            for q in range(5):
                if q in q_means.index:
                    print(f"  Q{q}: {q_means[q]:.6f}")
            if 4 in q_means.index and 0 in q_means.index:
                spread = q_means[4] - q_means[0]
                print(f"  Q4-Q0 Spread: {spread:.6f}")

        # Long-Short
        ls = self.long_short_returns()
        if not ls.empty:
            ls_ir = ls['ls'].mean() / ls['ls'].std() if ls['ls'].std() > 1e-12 else 0
            print(f"\n  --- Long-Short Portfolio ---")
            print(f"  LS Mean Ret: {ls['ls'].mean():.6f}")
            print(f"  LS Std:      {ls['ls'].std():.6f}")
            print(f"  LS IR:       {ls_ir:.4f}")
            print(f"  Win Rate:    {np.mean(ls['ls'] > 0):.2%}")

        # Turnover
        to = self.factor_turnover()
        print(f"\n  --- Turnover ---")
        print(f"  Avg Turnover: {to:.2%}" if not np.isnan(to) else f"  Avg Turnover: N/A")

        return ic_sum

    # ===== Plotting =====

    def plot_ic_series(self, savepath="result/analysis_ic_series.png"):
        """Plot IC time series with cumulative."""
        ic = self.compute_ic_series()
        valid_mask = ~np.isnan(ic)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Top: IC scatter + rolling mean
        ax = axes[0]
        ax.axhline(y=0, color='black', lw=0.5)
        ax.plot(np.arange(len(ic))[valid_mask], ic[valid_mask], 'o', ms=2, alpha=0.5,
                color=COLORS[0], label='Daily IC')
        if np.sum(valid_mask) > 60:
            roll_mean = pd.Series(ic[valid_mask]).rolling(60).mean().values
            ax.plot(np.arange(len(ic))[valid_mask], roll_mean, lw=2, color=COLORS[1],
                    label='60D Rolling Mean')
        ax.set_ylabel('Rank IC')
        ax.set_title(f'{self.name} — IC Time Series')
        ax.legend()

        # Bottom: Cumulative IC
        ax = axes[1]
        cum_ic = np.nancumsum(np.nan_to_num(ic, nan=0.0))
        ax.plot(cum_ic, lw=1.5, color=COLORS[3])
        ax.fill_between(range(len(cum_ic)), 0, cum_ic, alpha=0.2, color=COLORS[3])
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Cumulative IC')
        ax.set_title('Cumulative IC')

        plt.tight_layout()
        fig.savefig(savepath)
        plt.close(fig)
        print(f"[Analysis] {savepath}")

    def plot_quintile_returns(self, savepath="result/analysis_quintiles.png"):
        """Plot quintile mean returns."""
        qdf = self.quintile_analysis()
        if qdf.empty:
            return
        q_means = qdf.groupby('quintile')['mean_ret'].mean()

        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 5))
        bars = ax.bar(q_means.index, q_means.values, color=colors, edgecolor='black', lw=0.5)
        for bar, val in zip(bars, q_means.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.5f}', ha='center', va='bottom' if val > 0 else 'top',
                    fontsize=8, fontweight='bold')
        ax.axhline(y=0, color='black', lw=0.5)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f'Q{q} (Low)' if q == 0 else f'Q{q}' if q < 4 else f'Q{q} (High)' for q in range(5)])
        ax.set_ylabel('Mean Forward Return')
        ax.set_title(f'{self.name} — Quintile Returns')
        plt.tight_layout()
        fig.savefig(savepath)
        plt.close(fig)
        print(f"[Analysis] {savepath}")

    def plot_decay(self, savepath="result/analysis_decay.png"):
        """Plot factor autocorrelation decay."""
        ac = self.decay_analysis(max_lag=30)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(ac)), ac, color=COLORS[1], edgecolor='white')
        ax.axhline(y=0, color='black', lw=0.5)
        ax.axhline(y=0.5, color='gray', lw=0.5, ls='--')
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'{self.name} — Factor Autocorrelation Decay')
        plt.tight_layout()
        fig.savefig(savepath)
        plt.close(fig)
        print(f"[Analysis] {savepath}")

    def plot_all(self):
        """Generate all analysis charts."""
        self.plot_ic_series()
        self.plot_quintile_returns()
        self.plot_decay()


def analyze_hof_factors(hof, pset, data, feature_cols, trade_dates, stock_codes):
    """Analyze all Hall of Fame factors and return summary DataFrame."""
    from deap import gp
    ret_idx = feature_cols.index('RETURN') if 'RETURN' in feature_cols else 4
    # Compute 5-day forward returns
    daily_ret = data[:, :, ret_idx] / 100.0
    log_ret = np.log(1.0 + daily_ret)
    n_dates = data.shape[1]
    fwd5 = np.full((data.shape[0], n_dates), np.nan)
    for t in range(n_dates - 5):
        fwd5[:, t] = np.exp(np.sum(log_ret[:, t+1:t+6], axis=1)) - 1.0

    results = []
    for rank_i, ind in enumerate(hof):
        try:
            func = gp.compile(ind, pset)
            args_3d = [data[:, :, j] for j in range(data.shape[2])]
            fv = func(*args_3d)
            fv = np.atleast_1d(np.squeeze(fv))
            if fv.ndim != 2:
                continue
            analyzer = FactorAnalyzer(fv, fwd5, trade_dates, stock_codes, f"HoF_Rank{rank_i+1}")
            ic_sum = analyzer.ic_summary()
            ls = analyzer.long_short_returns()
            ls_mean = ls['ls'].mean() if not ls.empty else np.nan
            to = analyzer.factor_turnover()
            results.append({'Rank': rank_i+1, 'Expression': str(ind), 'Fitness': ind.fitness.values[0],
                            'IC_Mean': ic_sum['IC_Mean'], 'ICIR': ic_sum['ICIR'],
                            'LS_Ret': ls_mean, 'Turnover': to})
        except Exception as e:
            results.append({'Rank': rank_i+1, 'Expression': str(ind), 'Fitness': ind.fitness.values[0],
                            'IC_Mean': np.nan, 'ICIR': np.nan, 'LS_Ret': np.nan, 'Turnover': np.nan})

    summary = pd.DataFrame(results)
    summary.to_csv("result\\hof_analysis_summary.csv", index=False)
    print(f"[Analysis] result/hof_analysis_summary.csv")
    return summary
