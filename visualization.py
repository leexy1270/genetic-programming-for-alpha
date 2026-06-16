"""
visualization.py — Publication-quality GP evolution visualization

Usage:
    from visualization import plot_evolution_report
    plot_evolution_report(stats_log, hof, pset, data, stock_codes, trade_dates, feature_cols)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import os
from deap import gp

# ---- Style Configuration ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
os.makedirs("result", exist_ok=True)


def plot_evolution_convergence(stats_log, savepath="result/fig1_convergence.png"):
    """Fig 1: Fitness convergence (max, avg, min, std bands)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    gens = np.array(stats_log["gen"])
    max_f = np.array(stats_log["max_fitness"])
    avg_f = np.array(stats_log["avg_fitness"])
    min_f = np.array(stats_log["min_fitness"])
    sz = np.array(stats_log["avg_size"])

    # Left: Fitness
    ax = axes[0]
    ax.fill_between(gens, min_f, max_f, alpha=0.15, color=COLORS[0], label='Min-Max Range')
    ax.plot(gens, max_f, color=COLORS[0], lw=2, label='Max Fitness')
    ax.plot(gens, avg_f, color=COLORS[1], lw=2, ls='--', label='Avg Fitness')
    ax.axhline(y=0, color='black', lw=0.5, ls=':')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (ICIR)')
    ax.set_title('Fitness Convergence')
    ax.legend(frameon=True, fancybox=True)

    # Right: Tree size
    ax = axes[1]
    ax.fill_between(gens, 0, sz, alpha=0.2, color=COLORS[2])
    ax.plot(gens, sz, color=COLORS[2], lw=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg Tree Size (nodes)')
    ax.set_title('Bloat Control')

    fig.suptitle('GP Evolution — Factor Discovery', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualization] {savepath}")
    return fig


def plot_fitness_complexity(stats_log, savepath="result/fig2_pareto.png"):
    """Fig 2: Fitness-Complexity trade-off (Pareto front proxy)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    gens = np.array(stats_log["gen"])
    max_f = np.array(stats_log["max_fitness"])
    sz = np.array(stats_log["avg_size"])

    sc = ax.scatter(sz, max_f, c=gens, cmap='viridis', s=60, alpha=0.85, edgecolors='white', linewidth=0.5)
    ax.plot(sz, max_f, color='gray', alpha=0.3, lw=1)
    ax.set_xlabel('Avg Tree Size (nodes)')
    ax.set_ylabel('Max Fitness (ICIR)')
    ax.set_title('Fitness–Complexity Trade-off')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Generation')

    fig.tight_layout()
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualization] {savepath}")
    return fig


def plot_hof_quintiles(hof, pset, data, feature_cols, trade_dates, stock_codes,
                       forward_return=5, savepath="result/fig3_quintiles.png"):
    """Fig 3: Top-3 HoF factors — quintile return distribution."""
    n_show = min(3, len(hof))
    fig, axes = plt.subplots(1, n_show, figsize=(5*n_show, 4.5))
    if n_show == 1:
        axes = [axes]

    ret_idx = feature_cols.index('RETURN') if 'RETURN' in feature_cols else 4
    # compute forward returns
    n_dates = data.shape[1]
    daily_ret = data[:, :, ret_idx] / 100.0
    log_ret = np.log(1.0 + daily_ret)
    fwd_ret = np.full((data.shape[0], n_dates), np.nan, dtype=np.float64)
    for t in range(n_dates - forward_return):
        fwd_ret[:, t] = np.exp(np.sum(log_ret[:, t+1:t+1+forward_return], axis=1)) - 1.0

    for rank_i in range(n_show):
        ind = hof[rank_i]
        ax = axes[rank_i]
        try:
            func = gp.compile(ind, pset)
            args_3d = [data[:, :, j] for j in range(data.shape[2])]
            fv_raw = func(*args_3d)
            fv_raw = np.atleast_1d(np.squeeze(fv_raw))

            # Compute quintile returns
            quintile_returns = []
            for t in range(n_dates):
                cross = fv_raw[:, t] if fv_raw.ndim == 2 else fv_raw
                if fv_raw.ndim != 2:
                    continue
                fwd = fwd_ret[:, t]
                mask = ~(np.isnan(cross) | np.isnan(fwd) | np.isinf(cross) | np.isinf(fwd))
                if mask.sum() < 10:
                    continue
                fv_m = cross[mask]
                fwd_m = fwd[mask]
                try:
                    q5_idx = pd.qcut(fv_m, 5, labels=False, duplicates='drop')
                    for q in range(5):
                        if (q5_idx == q).sum() > 0:
                            quintile_returns.append({'quintile': q, 'ret': fwd_m[q5_idx == q].mean()})
                except Exception:
                    continue

            if quintile_returns:
                df_q = pd.DataFrame(quintile_returns)
                means = df_q.groupby('quintile')['ret'].mean()
                colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 5))
                bars = ax.bar(means.index, means.values, color=colors, edgecolor='black', linewidth=0.5)
                ax.axhline(y=0, color='black', lw=0.5)
                ax.set_xticks(range(5))
                ax.set_xticklabels([f'Q{q}' for q in range(5)])
                ax.set_xlabel('Factor Quintile')
                ax.set_ylabel(f'Mean Fwd {forward_return}D Return')
                ax.set_title(f'Rank {rank_i+1}: {str(ind)[:50]}...' if len(str(ind))>50 else f'Rank {rank_i+1}: {str(ind)}')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Rank {rank_i+1}')

    fig.suptitle('Hall of Fame — Factor Quintile Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualization] {savepath}")
    return fig


def plot_ic_decay(hof, pset, data, feature_cols, trade_dates, stock_codes,
                  forward_periods=[1, 3, 5, 10, 20], savepath="result/fig4_ic_decay.png"):
    """Fig 4: IC decay across different forward return periods."""
    n_show = min(3, len(hof))
    fig, axes = plt.subplots(1, n_show, figsize=(5*n_show, 4))
    if n_show == 1:
        axes = [axes]
    ret_idx = feature_cols.index('RETURN') if 'RETURN' in feature_cols else 4

    for rank_i in range(n_show):
        ind = hof[rank_i]
        ax = axes[rank_i]
        try:
            func = gp.compile(ind, pset)
            args_3d = [data[:, :, j] for j in range(data.shape[2])]
            fv_raw = func(*args_3d)
            fv_raw = np.atleast_1d(np.squeeze(fv_raw))
            if fv_raw.ndim != 2:
                continue

            ic_means = []
            for fwd_d in forward_periods:
                daily_ret = data[:, :, ret_idx] / 100.0
                log_ret = np.log(1.0 + daily_ret)
                n_dates = data.shape[1]
                fwd_ret = np.full((data.shape[0], n_dates), np.nan)
                for t in range(n_dates - fwd_d):
                    fwd_ret[:, t] = np.exp(np.sum(log_ret[:, t+1:t+1+fwd_d], axis=1)) - 1.0

                ic_list = []
                for t in range(n_dates):
                    fv_cross = fv_raw[:, t]
                    fwd_cross = fwd_ret[:, t]
                    mask = ~(np.isnan(fv_cross) | np.isnan(fwd_cross) | np.isinf(fv_cross) | np.isinf(fwd_cross))
                    if mask.sum() < 10:
                        continue
                    ic = np.corrcoef(fv_cross[mask], fwd_cross[mask])[0, 1]
                    if not np.isnan(ic):
                        ic_list.append(ic)
                ic_means.append(np.mean(np.abs(ic_list)) if ic_list else 0)

            ax.bar(range(len(forward_periods)), ic_means, color=COLORS[:len(forward_periods)], edgecolor='white')
            ax.set_xticks(range(len(forward_periods)))
            ax.set_xticklabels([f'{d}D' for d in forward_periods])
            ax.set_ylabel('|Mean IC|')
            ax.set_xlabel('Forward Return Period')
            ax.set_title(f'Rank {rank_i+1} IC Decay')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error', ha='center', va='center', transform=ax.transAxes)

    fig.suptitle('Factor IC Decay Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualization] {savepath}")
    return fig


def plot_evolution_report(stats_log, hof, pset, data, feature_cols, trade_dates, stock_codes):
    """Generate all evolution visualization figures."""
    print("\n[Visualization] Generating publication-quality figures...")
    plot_evolution_convergence(stats_log)
    plot_fitness_complexity(stats_log)
    plot_hof_quintiles(hof, pset, data, feature_cols, trade_dates, stock_codes)
    plot_ic_decay(hof, pset, data, feature_cols, trade_dates, stock_codes)
    print("[Visualization] All figures saved to result/\n")
