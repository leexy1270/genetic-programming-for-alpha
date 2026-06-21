"""
factor_analysis_standalone.py — 独立因子分析与可视化工具
====================================================================================
可直接输入GP因子表达式（DEAP格式），对因子进行全面评估并生成论文级图表。
数据加载方式与 main.py 保持一致（中证500成分股，7维基础特征+8维衍生特征）。

功能:
  1. 因子计算      — 从表达式字符串 → 编译 → 面板数据 (股票×日期)
  2. 数据预处理    — MAD去极值 → 截面中位数填充 → 截面Z-Score标准化
  3. IC分析        — 逐日截面Spearman Rank IC + 统计检验
  4. 分层回测      — 等量5分组(Q0~Q4)，检验分组收益单调性
  5. 多空组合      — 自动对齐IC方向，做多IC预测正向、做空IC预测负向
  6. 稳定性分析    — IC自相关衰减 + 因子换手率
  7. 可视化        — IC序列、分组收益、IC衰减、多空净值+回撤
  8. 综合报告      — 控制台 + Markdown

使用方式:
  # 命令行传入表达式
  python factor_analysis_standalone.py --expr "STD(NEQ(WMA(DECAYLINEAR(GAP, 230), 32), DBM(SIGN(RETURN), PRICE_POS)), 5)"

  # 从文件读取表达式（支持多因子）
  python factor_analysis_standalone.py --file result/best_factors.txt

  # 分析已有的HoF因子（从checkpoint加载）
  python factor_analysis_standalone.py --hof result/gp_checkpoint.pkl

  # 批量分析HoF所有因子并输出对比图表
  python factor_analysis_standalone.py --hof result/gp_checkpoint.pkl --compare

Author: GP Alpha Research
Date: 2026-06-20
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats as sp_stats
from deap import gp
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 全局绘图样式 — 论文级别
# ============================================================
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
    'figure.facecolor': 'white',
})

COLORS = ['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c', '#1f77b4']

os.makedirs("result", exist_ok=True)

# ============================================================
# 1. 数据加载（与 main.py 完全一致）
# ============================================================

def load_data() -> tuple:
    """
    加载3D数据 (自动检测股票/期货数据源)

    Returns
    -------
    data_3d : np.ndarray
    codes : list of str
    trade_dates : pd.DatetimeIndex
    feature_cols : list of str
    """
    from parameters import DATA_SOURCE

    if DATA_SOURCE == "futures":
        data_path = "data/futures_data_3d.npz"
        codes_key = "contract_codes"
        label = "contracts"
    else:
        data_path = "data/stock_data_3d.npz"
        codes_key = "stock_codes"
        label = "stocks"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    loaded = np.load(data_path, allow_pickle=True)
    data_3d = loaded["data_3d"]
    codes = loaded[codes_key].tolist()
    trade_dates = pd.DatetimeIndex(loaded["dates"])
    feature_cols = loaded["feature_cols"].tolist()

    print(f"[数据] 已加载: {data_3d.shape} | "
          f"{len(codes)} {label} × {len(trade_dates)} dates × {len(feature_cols)} features")
    print(f"[数据] 特征: {feature_cols}")
    print(f"[数据] 日期范围: {trade_dates[0].strftime('%Y-%m-%d')} ~ {trade_dates[-1].strftime('%Y-%m-%d')}")

    return data_3d, codes, trade_dates, feature_cols


# ============================================================
# 2. 表达式解析与因子计算
# ============================================================

def build_pset_from_features(feature_cols: list) -> gp.PrimitiveSetTyped:
    """基于特征列表构建原语集（与 main.py 共用 build_pset）。"""
    from build_pset import build_pset
    return build_pset(feature_cols)


def parse_and_compile(expr_str: str, pset: gp.PrimitiveSetTyped):
    """
    将DEAP表达式字符串解析为表达式树并编译。

    Parameters
    ----------
    expr_str : str  e.g. "STD(NEQ(...), 5)"
    pset : gp.PrimitiveSetTyped

    Returns
    -------
    tree : gp.PrimitiveTree
    func : callable
    """
    tree = gp.PrimitiveTree.from_string(expr_str, pset)
    func = gp.compile(tree, pset)
    return tree, func


def compute_factor_values(func, data_3d: np.ndarray) -> np.ndarray:
    """
    计算因子值面板。

    Parameters
    ----------
    func : callable  编译后的表达式函数
    data_3d : np.ndarray, shape (n_stocks, n_dates, n_features)

    Returns
    -------
    factor_values : np.ndarray, shape (n_stocks, n_dates)
    """
    n_features = data_3d.shape[2]
    args = [data_3d[:, :, j] for j in range(n_features)]

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        fv = func(*args)
        fv = np.atleast_1d(np.squeeze(fv))

    if fv.ndim != 2:
        raise ValueError(f"因子值形状异常: {fv.shape}，期望 2D (stocks × dates)")

    return fv.astype(np.float64)


# ============================================================
# 3. 数据预处理
# ============================================================

def preprocess_factor(factor: np.ndarray,
                      winsorize_std: float = 5.0,
                      fill_method: str = 'cross_median',
                      standardize: bool = True) -> np.ndarray:
    """
    标准预处理流水线:
    1. MAD去极值（逐日截面，默认5σ）
    2. 缺失值填充（截面中位数）
    3. 截面Z-Score标准化

    Parameters
    ----------
    factor : np.ndarray, shape (n_stocks, n_dates)
    winsorize_std : float  MAD倍数阈值
    fill_method : str  'cross_median' | 'zero'
    standardize : bool  是否执行截面标准化

    Returns
    -------
    cleaned : np.ndarray, same shape
    """
    cleaned = factor.copy()
    n_dates = cleaned.shape[1]

    for t in range(n_dates):
        cross = cleaned[:, t]
        mask = ~(np.isnan(cross) | np.isinf(cross))
        if mask.sum() < 3:
            continue

        # 3.1 MAD去极值
        median = np.median(cross[mask])
        mad = np.median(np.abs(cross[mask] - median)) * 1.4826  # 一致性校正
        if mad >= 1e-12:
            upper = median + winsorize_std * mad
            lower = median - winsorize_std * mad
            cleaned[mask, t] = np.clip(cross[mask], lower, upper)

        # 3.2 缺失值填充
        mask2 = np.isnan(cleaned[:, t]) | np.isinf(cleaned[:, t])
        if mask2.any():
            if fill_method == 'cross_median':
                valid_vals = cleaned[~mask2, t]
                if len(valid_vals) > 0:
                    cleaned[mask2, t] = np.nanmedian(valid_vals)
                else:
                    cleaned[mask2, t] = 0.0
            else:  # 'zero'
                cleaned[mask2, t] = 0.0

        # 3.3 截面Z-Score标准化
        if standardize:
            mask3 = ~(np.isnan(cleaned[:, t]) | np.isinf(cleaned[:, t]))
            if mask3.sum() >= 3:
                vals = cleaned[mask3, t]
                mu, sigma = vals.mean(), vals.std(ddof=1)
                if sigma >= 1e-12:
                    cleaned[mask3, t] = (vals - mu) / sigma
                else:
                    cleaned[mask3, t] = 0.0

    return cleaned


# ============================================================
# 4. 前向收益计算
# ============================================================

def compute_forward_returns(data_3d: np.ndarray,
                            feature_cols: list,
                            forward_period: int = 5) -> np.ndarray:
    """
    计算未来N日累计收益（对数收益累加，向量化）。
    与 evaluate.py 中 _compute_forward_returns 逻辑一致。

    Returns
    -------
    fwd_ret : np.ndarray, shape (n_stocks, n_dates)
        末尾 forward_period 天为 NaN（因为无法计算未来收益）。
    """
    n_stocks, n_dates, _ = data_3d.shape
    ret_idx = feature_cols.index('RETURN') if 'RETURN' in feature_cols else 4

    daily_ret = data_3d[:, :, ret_idx] / 100.0  # 百分比 → 小数
    fwd_ret = np.full((n_stocks, n_dates), np.nan, dtype=np.float64)

    if n_dates <= forward_period:
        return fwd_ret

    # 使用原始 main.py 中的逐日循环方法计算前向收益。
    # 此方法相比 cumsum 向量化方法能正确处理 NaN（NaN 不会传染到
    # 后续窗口），确保每只股票在其实际上市期间的前向收益计算正确。
    log_ret = np.log(1.0 + daily_ret)
    fwd_ret = np.full((n_stocks, n_dates), np.nan, dtype=np.float64)
    for t in range(n_dates - forward_period):
        fwd_ret[:, t] = np.exp(
            np.nansum(log_ret[:, t+1:t+1+forward_period], axis=1)
        ) - 1.0

    return fwd_ret


# ============================================================
# 5. IC 分析
# ============================================================

def compute_ic_series(factor: np.ndarray,
                      fwd_ret: np.ndarray,
                      forward_period: int = 5) -> np.ndarray:
    """
    计算逐日截面Spearman Rank IC。
    【修复】正确使用 forward_period 偏移，避免前视偏差。

    Parameters
    ----------
    factor : np.ndarray, shape (n_stocks, n_dates)
    fwd_ret : np.ndarray, shape (n_stocks, n_dates)
    forward_period : int

    Returns
    -------
    ic_series : np.ndarray, shape (n_dates - forward_period,)
    """
    n_dates = factor.shape[1]
    ic_list = []

    for t in range(n_dates - forward_period):
        fv_cross = factor[:, t]
        fwd_cross = fwd_ret[:, t]
        mask = ~(np.isnan(fv_cross) | np.isnan(fwd_cross) |
                 np.isinf(fv_cross) | np.isinf(fwd_cross))
        if mask.sum() < 10:
            ic_list.append(np.nan)
            continue

        ic = sp_stats.spearmanr(fv_cross[mask], fwd_cross[mask])[0]
        ic_list.append(ic if not np.isnan(ic) else np.nan)

    return np.array(ic_list)


def compute_ic_stats(ic_series: np.ndarray) -> dict:
    """
    计算IC序列的统计量。

    Returns
    -------
    dict with: ic_mean, ic_std, icir, pos_ratio, t_stat, n_obs
    """
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) < 10:
        return {
            'ic_mean': np.nan, 'ic_std': np.nan, 'icir': np.nan,
            'pos_ratio': np.nan, 't_stat': np.nan, 'n_obs': 0
        }

    mean_ic = np.mean(valid)
    std_ic = np.std(valid, ddof=1)

    return {
        'ic_mean': mean_ic,
        'ic_std': std_ic,
        'icir': mean_ic / std_ic if std_ic > 1e-12 else 0.0,
        'pos_ratio': np.mean(valid > 0),
        't_stat': mean_ic / (std_ic / np.sqrt(len(valid))) if std_ic > 1e-12 else 0.0,
        'n_obs': len(valid),
    }


# ============================================================
# 6. 分层回测
# ============================================================

def quintile_analysis(factor: np.ndarray,
                      fwd_ret: np.ndarray,
                      forward_period: int = 5,
                      n_groups: int = 5,
                      trade_dates=None) -> pd.DataFrame:
    """
    因子分层回测 — 按因子值等量分n_groups组，计算各组平均未来收益。

    【修复说明】
    原代码使用 pd.qcut(..., duplicates='drop') 或 np.quantile，当因子值存在
    大量重复时（如比较运算符输出0/1掩码），会产生少于n_groups个分组或空组。
    修复方案：使用 rank-based 分位法，保证每组股票数严格相等（±1）。

    Parameters
    ----------
    factor : np.ndarray, shape (n_stocks, n_dates)
    fwd_ret : np.ndarray, shape (n_stocks, n_dates)
    forward_period : int
    n_groups : int  分组数（默认5组）
    trade_dates : pd.DatetimeIndex, optional

    Returns
    -------
    DataFrame with columns: date, quintile, mean_ret, n_stocks
    """
    n_dates = factor.shape[1]
    records = []

    for t in range(n_dates - forward_period):
        fv_cross = factor[:, t]
        fwd_cross = fwd_ret[:, t]
        mask = ~(np.isnan(fv_cross) | np.isnan(fwd_cross) |
                 np.isinf(fv_cross) | np.isinf(fwd_cross))
        if mask.sum() < n_groups * 3:
            continue

        valid_fv = fv_cross[mask]
        valid_fwd = fwd_cross[mask]
        n_valid = len(valid_fv)

        # 【修复】Rank-based分位法: 每组严格等量（±1）
        # 步骤: 排序 → 按秩分桶，确保每桶大小 = floor(n_valid / n_groups)
        order = np.argsort(valid_fv)
        ranks = np.empty(n_valid, dtype=np.int64)
        ranks[order] = np.arange(n_valid)

        # 每组目标大小
        group_size = n_valid // n_groups
        # 前 remainder 组多分1只股票
        remainder = n_valid % n_groups

        group_bounds = [0]
        for g in range(n_groups):
            extra = 1 if g < remainder else 0
            group_bounds.append(group_bounds[-1] + group_size + extra)

        for g in range(n_groups):
            start, end = group_bounds[g], group_bounds[g + 1]
            # 取排序后该组对应的原始索引
            g_indices = order[start:end]
            if len(g_indices) > 0:
                records.append({
                    'date': trade_dates[t] if trade_dates is not None else t,
                    'quintile': g,
                    'mean_ret': valid_fwd[g_indices].mean(),
                    'n_stocks': len(g_indices),
                })

    return pd.DataFrame(records)


# ============================================================
# 7. 多空组合
# ============================================================

def long_short_returns(factor: np.ndarray,
                       fwd_ret: np.ndarray,
                       forward_period: int = 5,
                       top_quantile: float = 0.2,
                       bottom_quantile: float = 0.2,
                       ic_sign: float = 1.0,
                       trade_dates=None) -> pd.DataFrame:
    """
    多空组合回测。

    【修复说明】
    原代码无论IC方向如何，始终做多因子值最高的top_quantile、做空因子值最低的
    bottom_quantile（即 ls = top_ret - bot_ret）。
    当IC为负（高因子值→低未来收益）时，多空收益为负，不利于报告展示。

    修复方案：根据ic_sign自动对齐多空方向。
    - ic_sign > 0: 做多高因子值（Q4），做空低因子值（Q0）→ ls = Q4 - Q0
    - ic_sign < 0: 做多低因子值（Q0），做空高因子值（Q4）→ ls = Q0 - Q4

    始终保证 ls > 0 表示因子有效。

    Parameters
    ----------
    factor, fwd_ret, forward_period : 同 quintile_analysis
    top_quantile : float  top组比例（默认20%）
    bottom_quantile : float  bottom组比例（默认20%）
    ic_sign : float  IC均值符号（>0: 正相关, <0: 负相关）
    trade_dates : pd.DatetimeIndex, optional

    Returns
    -------
    DataFrame with columns: date, long_ret, short_ret, ls_ret
        ls_ret > 0 表示因子在有效方向上盈利。
    """
    n_dates = factor.shape[1]
    records = []

    for t in range(n_dates - forward_period):
        fv_cross = factor[:, t]
        fwd_cross = fwd_ret[:, t]
        mask = ~(np.isnan(fv_cross) | np.isnan(fwd_cross) |
                 np.isinf(fv_cross) | np.isinf(fwd_cross))
        if mask.sum() < 15:
            continue

        valid_fv = fv_cross[mask]
        valid_fwd = fwd_cross[mask]

        n_top = max(1, int(mask.sum() * top_quantile))
        n_bot = max(1, int(mask.sum() * bottom_quantile))

        order = np.argsort(valid_fv)
        # 因子值最高的n_top只 → 尾部
        top_ret = valid_fwd[order[-n_top:]].mean()
        # 因子值最低的n_bot只 → 头部
        bot_ret = valid_fwd[order[:n_bot]].mean()

        # 【修复】根据IC符号对齐多空方向
        if ic_sign >= 0:
            # 正IC: 高因子值 → 高收益, 做多高因子值
            ls_ret = top_ret - bot_ret
            long_ret = top_ret
            short_ret = bot_ret
        else:
            # 负IC: 低因子值 → 高收益, 做多低因子值
            ls_ret = bot_ret - top_ret
            long_ret = bot_ret
            short_ret = top_ret

        records.append({
            'date': trade_dates[t] if trade_dates is not None else t,
            'long_ret': long_ret,
            'short_ret': short_ret,
            'ls_ret': ls_ret,
        })

    return pd.DataFrame(records)


# ============================================================
# 8. 因子稳定性分析
# ============================================================

def factor_turnover(factor: np.ndarray,
                    top_quantile: float = 0.2) -> float:
    """
    计算Top组（因子值最高的top_quantile比例股票）的日均换手率。

    换手率 = 1 - (当日Top组 ∩ 前日Top组) / |Top组|
    取值范围 [0, 1]，越低越稳定。

    Returns
    -------
    avg_turnover : float
    """
    n_dates = factor.shape[1]
    turnovers = []
    prev_top = None

    for t in range(n_dates):
        fv_cross = factor[:, t]
        mask = ~(np.isnan(fv_cross) | np.isinf(fv_cross))
        if mask.sum() < 15:
            continue

        n_top = max(1, int(mask.sum() * top_quantile))
        order = np.argsort(fv_cross[mask])
        # 记录的是在原始数组中的全局索引
        top_indices = set(np.where(mask)[0][order[-n_top:]])

        if prev_top is not None and len(prev_top) > 0 and len(top_indices) > 0:
            overlap = len(top_indices & prev_top)
            turnover = 1.0 - overlap / len(top_indices)
            turnovers.append(turnover)

        prev_top = top_indices

    return np.mean(turnovers) if turnovers else np.nan


def ic_autocorrelation(ic_series: np.ndarray,
                       max_lag: int = 30) -> np.ndarray:
    """
    计算IC序列的自相关衰减（替代原decay_analysis中对因子均值的自相关）。

    【修复说明】
    原代码 compute_ic_series → decay_analysis 计算的是因子截面均值的自相关，
    该指标与因子截面有效性（换手率）无关——因子均值可能高度自相关，
    但截面排序完全随机（换手率=100%）。

    修复方案：计算IC序列本身的自相关。IC自相关衡量的是因子预测能力的持续性，
    高 IC 自相关 → 因子预测信号持续有效 → 换手需求低。

    Returns
    -------
    autocorr : np.ndarray, shape (max_lag + 1,)
        autocorr[0] = 1.0, autocorr[k] = corr(IC[t], IC[t-k])
    """
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) <= max_lag:
        return np.full(max_lag + 1, np.nan)

    autocorr = [1.0]
    for lag in range(1, max_lag + 1):
        ac = np.corrcoef(valid[:-lag], valid[lag:])[0, 1]
        autocorr.append(ac if not np.isnan(ac) else np.nan)

    return np.array(autocorr)


# ============================================================
# 9. 综合因子分析（串联上述所有步骤）
# ============================================================

class FactorAnalyzer:
    """
    独立因子分析器 — 串联因子计算、预处理、IC、分层、多空、稳定性全流程。

    Usage:
        analyzer = FactorAnalyzer(data_3d, feature_cols, trade_dates, stock_codes, pset)
        results = analyzer.run("STD(NEQ(WMA(...), 5))", name="MyFactor")
        analyzer.print_report(results)
        analyzer.plot_all(results, output_dir="result/myfactor")
    """

    def __init__(self,
                 data_3d: np.ndarray,
                 feature_cols: list,
                 trade_dates: pd.DatetimeIndex,
                 stock_codes: list,
                 pset: gp.PrimitiveSetTyped,
                 forward_period: int = 5):
        self.data = data_3d
        self.feature_cols = list(feature_cols)
        self.trade_dates = trade_dates
        self.stock_codes = list(stock_codes)
        self.pset = pset
        self.forward_period = forward_period
        self.n_stocks, self.n_dates, self.n_features = data_3d.shape

        # 预计算前向收益（所有因子共用）
        self.fwd_ret = compute_forward_returns(data_3d, feature_cols, forward_period)

    def run(self,
            expr_str: str,
            name: str = "Factor",
            preprocess: bool = True,
            winsorize_std: float = 5.0,
            fill_method: str = 'cross_median',
            standardize: bool = False) -> dict:
        """
        运行完整的因子分析流程。

        Parameters
        ----------
        expr_str : str  DEAP格式因子表达式
        name : str  因子名称（用于图表标题和文件命名）
        preprocess : bool  是否对因子值进行预处理
        winsorize_std, fill_method, standardize : 预处理参数

        Returns
        -------
        results : dict  包含所有分析结果的字典
        """
        print(f"\n{'='*60}")
        print(f"  分析因子: {name}")
        print(f"{'='*60}")

        # Step 1: 解析并计算因子
        print(f"\n[1/6] 解析表达式并计算因子值...")
        try:
            tree, func = parse_and_compile(expr_str, self.pset)
        except Exception as e:
            raise ValueError(f"表达式解析失败: {e}\n表达式: {expr_str}")

        fv_raw = compute_factor_values(func, self.data)
        print(f"  因子值形状: {fv_raw.shape} | 有效值比例: {(~np.isnan(fv_raw)).mean():.1%}")

        # Step 2: 预处理
        fv_processed = fv_raw.copy()
        if preprocess:
            print(f"\n[2/6] 数据预处理 (MAD={winsorize_std}σ, fill={fill_method}, "
                  f"standardize={standardize})...")
            fv_processed = preprocess_factor(fv_raw, winsorize_std, fill_method, standardize)
        else:
            print(f"\n[2/6] 跳过预处理（使用原始因子值）")

        # Step 3: IC分析
        print(f"\n[3/6] 计算截面Rank IC...")
        ic_series = compute_ic_series(fv_processed, self.fwd_ret, self.forward_period)
        ic_stats = compute_ic_stats(ic_series)
        ic_mean = ic_stats['ic_mean']
        print(f"  IC Mean={ic_mean:.6f}  Std={ic_stats['ic_std']:.6f}  "
              f"ICIR={ic_stats['icir']:.4f}  t={ic_stats['t_stat']:.2f}  "
              f"N={ic_stats['n_obs']}")

        # Step 4: 分层回测
        print(f"\n[4/6] 分层回测 (5组)...")
        qdf = quintile_analysis(fv_processed, self.fwd_ret, self.forward_period,
                                n_groups=5, trade_dates=self.trade_dates)
        q_means = {}
        if not qdf.empty:
            q_means = qdf.groupby('quintile')['mean_ret'].mean().to_dict()
            spread = q_means.get(4, np.nan) - q_means.get(0, np.nan)
            q_str = " | ".join([f"Q{q}: {q_means.get(q, np.nan):.6f}" for q in range(5)])
            print(f"  {q_str}")
            print(f"  Q4-Q0 Spread: {spread:.6f}")
        else:
            spread = np.nan
            print(f"  分组回测数据为空")

        # Step 5: 多空组合（IC方向对齐）
        print(f"\n[5/6] 多空组合回测 (Top/Bottom 20%, IC方向对齐)...")
        ic_sign = 1.0 if ic_mean >= 0 else -1.0 if not np.isnan(ic_mean) else 1.0
        ls_df = long_short_returns(fv_processed, self.fwd_ret, self.forward_period,
                                   top_quantile=0.2, bottom_quantile=0.2,
                                   ic_sign=ic_sign, trade_dates=self.trade_dates)
        if not ls_df.empty:
            ls_mean = ls_df['ls_ret'].mean()
            ls_std = ls_df['ls_ret'].std()
            ls_ir = ls_mean / ls_std if ls_std > 1e-12 else 0.0
            ls_win = np.mean(ls_df['ls_ret'] > 0)
            ls_annual_ret = ls_mean * 252
            ls_annual_vol = ls_std * np.sqrt(252)
            ls_sharpe = ls_annual_ret / ls_annual_vol if ls_annual_vol > 1e-12 else 0.0
            ls_cumret = (1.0 + ls_df['ls_ret'].values).cumprod() - 1.0
            ls_maxdd = _max_drawdown(ls_cumret)
            print(f"  日均LS收益: {ls_mean:.6f}  年化: {ls_annual_ret:.2%}")
            print(f"  LS IR (日频): {ls_ir:.4f}  夏普(年化): {ls_sharpe:.2f}")
            print(f"  胜率: {ls_win:.1%}  最大回撤: {ls_maxdd:.2%}")
            print(f"  IC方向: {'正向 (做多高因子值)' if ic_sign > 0 else '负向 (做多低因子值)'}")
        else:
            ls_mean = ls_std = ls_ir = ls_win = ls_annual_ret = ls_annual_vol = np.nan
            ls_sharpe = ls_maxdd = ls_cumret = np.nan
            print(f"  多空组合数据为空")

        # Step 6: 稳定性分析
        print(f"\n[6/6] 因子稳定性分析...")
        turnover = factor_turnover(fv_processed, top_quantile=0.2)
        ic_ac = ic_autocorrelation(ic_series, max_lag=30)
        ac_5d = ic_ac[5] if len(ic_ac) > 5 else np.nan
        ac_20d = ic_ac[20] if len(ic_ac) > 20 else np.nan
        print(f"  Top组日均换手率: {turnover:.2%}")
        print(f"  IC自相关 AC(5d)={ac_5d:.4f}  AC(20d)={ac_20d:.4f}")

        # 汇总
        results = {
            'name': name,
            'expr_str': expr_str,
            'tree': tree,
            'factor_raw': fv_raw,
            'factor_processed': fv_processed,
            'ic_series': ic_series,
            'ic_stats': ic_stats,
            'quintile_df': qdf,
            'q_means': q_means,
            'q_spread': spread,
            'ls_df': ls_df,
            'ls_mean': ls_mean,
            'ls_std': ls_std,
            'ls_ir': ls_ir,
            'ls_win': ls_win,
            'ls_annual_ret': ls_annual_ret,
            'ls_annual_vol': ls_annual_vol,
            'ls_sharpe': ls_sharpe,
            'ls_maxdd': ls_maxdd,
            'ls_cumret': ls_cumret,
            'ic_sign': ic_sign,
            'turnover': turnover,
            'ic_autocorr': ic_ac,
            'ac_5d': ac_5d,
            'ac_20d': ac_20d,
            'nodes': tree.height if hasattr(tree, 'height') else len(tree),
            'height': tree.height if hasattr(tree, 'height') else len(tree),
        }

        return results

    # ============================================================
    # 报告打印
    # ============================================================

    def print_report(self, results: dict):
        """打印格式化的因子分析报告（控制台）。"""
        ic = results['ic_stats']
        print(f"\n{'='*65}")
        print(f"  因子测试报告: {results['name']}")
        print(f"{'='*65}")
        print(f"\n  表达式:")
        expr = results['expr_str']
        if len(expr) > 120:
            print(f"    {expr[:120]}...")
        else:
            print(f"    {expr}")
        print(f"  节点数: {results['nodes']}  高度: {results['height']}")

        print(f"\n  {'─'*55}")
        print(f"  [IC 分析]  Spearman Rank IC | 前向收益: {self.forward_period}D")
        print(f"  {'─'*55}")
        print(f"  IC均值:     {ic['ic_mean']:>12.6f}")
        print(f"  IC标准差:   {ic['ic_std']:>12.6f}")
        print(f"  ICIR:       {ic['icir']:>12.4f}")
        print(f"  IC>0比例:   {ic['pos_ratio']:>11.1%}")
        print(f"  t统计量:    {ic['t_stat']:>12.2f}")
        print(f"  有效观测:   {ic['n_obs']:>12}")

        print(f"\n  {'─'*55}")
        print(f"  [分层回测]  5等量分组")
        print(f"  {'─'*55}")
        q_means = results['q_means']
        for q in range(5):
            val = q_means.get(q, np.nan)
            label = f"Q{q} ({'Low' if q == 0 else 'High' if q == 4 else f'Q{q}'})"
            print(f"  {label:<16} {val:>12.6f}")
        spread = results['q_spread']
        print(f"  {'Q4-Q0 Spread':<16} {spread:>12.6f}")

        print(f"\n  {'─'*55}")
        print(f"  [多空组合]  Top/Bottom 20% | 方向: {'正向' if results['ic_sign'] > 0 else '负向'}")
        print(f"  {'─'*55}")
        print(f"  日均收益:    {results['ls_mean']:>12.6f}")
        print(f"  年化收益:    {results['ls_annual_ret']:>11.2%}")
        print(f"  年化波动:    {results['ls_annual_vol']:>11.2%}")
        print(f"  LS IR(日频): {results['ls_ir']:>12.4f}")
        print(f"  夏普(年化):  {results['ls_sharpe']:>12.2f}")
        print(f"  日胜率:      {results['ls_win']:>11.1%}")
        print(f"  最大回撤:    {results['ls_maxdd']:>11.2%}")

        print(f"\n  {'─'*55}")
        print(f"  [稳定性]")
        print(f"  {'─'*55}")
        print(f"  换手率:      {results['turnover']:>11.2%}")
        print(f"  IC AC(5d):   {results['ac_5d']:>12.4f}")
        print(f"  IC AC(20d):  {results['ac_20d']:>12.4f}")

        print(f"\n  {'─'*55}")
        print(f"  [综合评价]")
        print(f"  {'─'*55}")

        # 自动生成评价
        comments = []
        if not np.isnan(ic['t_stat']) and abs(ic['t_stat']) > 2.58:
            comments.append("[+] IC统计极显著 (|t|>2.58, p<0.01)")
        elif not np.isnan(ic['t_stat']) and abs(ic['t_stat']) > 1.96:
            comments.append("[+] IC统计显著 (|t|>1.96, p<0.05)")
        else:
            comments.append("[!] IC未达统计显著水平")

        if not np.isnan(ic['icir']) and abs(ic['icir']) >= 0.3:
            comments.append("[+] ICIR达标 (>=0.3)，可作为独立因子")
        elif not np.isnan(ic['icir']):
            comments.append("[~] ICIR偏低 (<0.3)，建议作为多因子组成部分")

        if not np.isnan(results['turnover']):
            if results['turnover'] < 0.10:
                comments.append("[+] 换手率极低 (<10%)，交易成本友好")
            elif results['turnover'] < 0.30:
                comments.append("[~] 换手率中等 (10-30%)")
            else:
                comments.append("[!] 换手率偏高 (>30%)，交易成本需关注")

        if not np.isnan(results['ls_win']) and results['ls_win'] > 0.55:
            comments.append("[+] 多空胜率较高 (>55%)")

        for c in comments:
            print(f"  {c}")

        print(f"\n{'='*65}\n")

    # ============================================================
    # 可视化
    # ============================================================

    def plot_all(self, results: dict, output_dir: str = "result",
                 prefix: str = ""):
        """
        生成全部分析图表。

        输出文件:
        - {prefix}ic_series.png     IC时间序列 + 累计IC + 月度IC
        - {prefix}quintile_returns.png  五分组平均收益柱状图
        - {prefix}ic_decay.png      IC自相关衰减
        - {prefix}ls_performance.png  多空净值 + 回撤

        Parameters
        ----------
        results : dict  run()返回的结果字典
        output_dir : str  输出目录
        prefix : str  文件名前缀（如 "rank1_"）
        """
        os.makedirs(output_dir, exist_ok=True)
        name = results['name']

        self._plot_ic_series(results, output_dir, prefix)
        self._plot_quintile_returns(results, output_dir, prefix)
        self._plot_ic_decay(results, output_dir, prefix)
        self._plot_ls_performance(results, output_dir, prefix)

    def _plot_ic_series(self, results, output_dir, prefix):
        """IC时间序列图（3子图: 日IC分布, 累计IC, 月度IC热力）"""
        ic = results['ic_series']
        valid_mask = ~np.isnan(ic)
        valid_ic = ic[valid_mask]
        n_valid = len(valid_ic)

        fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                                 gridspec_kw={'height_ratios': [2, 1.5, 1.5]})

        # 子图1: 日IC散点 + 60日滚动均值
        ax = axes[0]
        ax.axhline(y=0, color='black', lw=0.5, ls='--')
        x_days = np.arange(len(ic))
        ax.scatter(x_days[valid_mask], valid_ic, s=4, alpha=0.4,
                   color=COLORS[0], label='Daily Rank IC')
        if n_valid > 60:
            roll_mean = pd.Series(valid_ic).rolling(60, min_periods=20).mean().values
            ax.plot(x_days[valid_mask], roll_mean, lw=2, color=COLORS[1],
                    label='60D Rolling Mean')
        ax.axhline(y=results['ic_stats']['ic_mean'], color=COLORS[4], lw=1.5,
                   ls=':', label=f"Mean={results['ic_stats']['ic_mean']:.4f}")
        ax.set_ylabel('Rank IC')
        ax.set_title(f'{results["name"]} — IC Time Series (ICIR={results["ic_stats"]["icir"]:.3f})')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, len(ic))

        # 子图2: 累计IC
        ax = axes[1]
        cum_ic = np.nancumsum(np.nan_to_num(ic, nan=0.0))
        ax.fill_between(range(len(cum_ic)), 0, cum_ic, alpha=0.15, color=COLORS[0])
        ax.plot(cum_ic, lw=1.5, color=COLORS[0])
        ax.axhline(y=0, color='black', lw=0.5, ls='--')
        ax.set_ylabel('Cumulative IC')
        ax.set_title('Cumulative IC')
        ax.set_xlim(0, len(ic))

        # 子图3: 月度IC均值柱状图
        ax = axes[2]
        if self.trade_dates is not None and n_valid > 20:
            ic_series_pd = pd.Series(ic, index=self.trade_dates[:len(ic)])
            monthly_ic = ic_series_pd.resample('ME').mean()
            monthly_valid = monthly_ic[~monthly_ic.isna()]
            colors_bar = ['#d62728' if v < 0 else '#2ca02c' for v in monthly_valid.values]
            ax.bar(range(len(monthly_valid)), monthly_valid.values, color=colors_bar,
                   edgecolor='white', width=0.8)
            ax.axhline(y=0, color='black', lw=0.5)
            # 标注x轴（每3个月标一个）
            tick_step = max(1, len(monthly_valid) // 12)
            ax.set_xticks(range(0, len(monthly_valid), tick_step))
            ax.set_xticklabels([monthly_valid.index[i].strftime('%Y-%m')
                                for i in range(0, len(monthly_valid), tick_step)],
                               rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Mean Monthly IC')
            ax.set_title('Monthly Mean IC')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)

        plt.tight_layout()
        savepath = f"{output_dir}/{prefix}ic_series.png"
        fig.savefig(savepath)
        plt.close(fig)
        print(f"[图表] {savepath}")

    def _plot_quintile_returns(self, results, output_dir, prefix):
        """五分组平均收益柱状图"""
        q_means = results['q_means']
        if not q_means:
            return

        fig, ax = plt.subplots(figsize=(7, 5))
        groups = sorted(q_means.keys())
        values = [q_means[g] for g in groups]
        colors_bar = plt.cm.RdYlGn(np.linspace(0.12, 0.88, len(groups)))

        bars = ax.bar(groups, values, color=colors_bar, edgecolor='black', lw=0.6, width=0.65)
        ax.axhline(y=0, color='black', lw=0.6)

        # 标注数值
        for bar, val in zip(bars, values):
            va = 'bottom' if val >= 0 else 'top'
            offset = max(abs(val) * 0.02, 0.00002)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + offset if val >= 0 else val - offset,
                    f'{val:.6f}', ha='center', va=va, fontsize=9, fontweight='bold')

        # 标注多空利差
        if len(groups) >= 2:
            spread = values[-1] - values[0]
            ax.text(0.5, 0.95,
                    f'Q{groups[-1]}−Q{groups[0]} Spread = {spread:.6f}',
                    ha='center', va='top', transform=ax.transAxes,
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.85))

        ax.set_xticks(groups)
        ax.set_xticklabels([f'Q{g}\n({"Low" if g == groups[0] else "High" if g == groups[-1] else ""})'
                            for g in groups])
        ax.set_ylabel(f'Mean Forward {self.forward_period}D Return')
        ax.set_title(f'{results["name"]} — Quintile Returns')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        savepath = f"{output_dir}/{prefix}quintile_returns.png"
        fig.savefig(savepath)
        plt.close(fig)
        print(f"[图表] {savepath}")

    def _plot_ic_decay(self, results, output_dir, prefix):
        """IC自相关衰减图"""
        ic_ac = results['ic_autocorr']
        max_lag = len(ic_ac) - 1

        fig, ax = plt.subplots(figsize=(8, 4.5))
        lags = range(max_lag + 1)
        colors_bar = [COLORS[0]] + [COLORS[1]] * max_lag
        ax.bar(lags, ic_ac, color=colors_bar, edgecolor='white', width=0.6)
        ax.axhline(y=0, color='black', lw=0.5)
        ax.axhline(y=0.1, color='gray', lw=0.5, ls='--', alpha=0.5)
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('IC Autocorrelation')
        ax.set_title(f'{results["name"]} — IC Autocorrelation Decay')
        ax.set_xticks(range(0, max_lag + 1, 5))
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        savepath = f"{output_dir}/{prefix}ic_decay.png"
        fig.savefig(savepath)
        plt.close(fig)
        print(f"[图表] {savepath}")

    def _plot_ls_performance(self, results, output_dir, prefix):
        """多空净值 + 回撤图"""
        ls_df = results['ls_df']
        if ls_df.empty:
            return

        ls_ret = ls_df['ls_ret'].values
        cumret = (1.0 + ls_ret).cumprod() - 1.0
        peak = np.maximum.accumulate(cumret)
        drawdown = (cumret - peak) / (1.0 + peak)

        # 日期轴
        if self.trade_dates is not None and 'date' in ls_df.columns:
            x_vals = ls_df['date'].values
        else:
            x_vals = np.arange(len(cumret))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                       gridspec_kw={'height_ratios': [2, 1]},
                                       sharex=True)

        # 上图: 多空累计净值
        ax1.fill_between(x_vals, 0, cumret, alpha=0.12, color=COLORS[4])
        ax1.plot(x_vals, cumret, lw=1.5, color=COLORS[4])
        ax1.axhline(y=0, color='black', lw=0.5, ls='--')
        ax1.set_ylabel('Cumulative LS Return')
        ax1.set_title(f'{results["name"]} — Long-Short Performance '
                      f'(Sharpe={results["ls_sharpe"]:.2f}, '
                      f'MaxDD={results["ls_maxdd"]:.1%})')
        ax1.grid(alpha=0.3)

        # 下图: 回撤
        ax2.fill_between(x_vals, 0, drawdown, alpha=0.35, color='#d62728')
        ax2.plot(x_vals, drawdown, lw=0.8, color='#d62728')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.grid(alpha=0.3)
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

        plt.tight_layout()
        savepath = f"{output_dir}/{prefix}ls_performance.png"
        fig.savefig(savepath)
        plt.close(fig)
        print(f"[图表] {savepath}")


# ============================================================
# 10. 多因子横向对比
# ============================================================

def multi_factor_compare(analyzer: FactorAnalyzer,
                         all_results: list,
                         output_dir: str = "result"):
    """
    多因子横向对比可视化（当有2个及以上因子时使用）。

    Parameters
    ----------
    analyzer : FactorAnalyzer
    all_results : list of dict  每个因子的 run() 结果
    output_dir : str
    """
    n = len(all_results)
    if n < 2:
        return

    print(f"\n[对比] 生成多因子横向对比图表...")

    # ---- 图1: 累计多空收益对比 + ICIR柱状图 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # 左: 累计多空净值
    ax = axes[0]
    for i, r in enumerate(all_results):
        ls_df = r['ls_df']
        if ls_df.empty:
            continue
        ls_ret = ls_df['ls_ret'].values
        cumret = (1.0 + ls_ret).cumprod() - 1.0
        if 'date' in ls_df.columns:
            x_vals = ls_df['date'].values
        else:
            x_vals = np.arange(len(cumret))
        ax.plot(x_vals, cumret, lw=1.5, alpha=0.85,
                label=f"{r['name']} (IR={r['ls_ir']:.3f})")
    ax.axhline(y=0, color='black', lw=0.5, ls='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative LS Return')
    ax.set_title(f'Multi-Factor — Cumulative Long-Short ({analyzer.forward_period}D Forward)')
    ax.legend(fontsize=7, frameon=True)
    ax.grid(alpha=0.3)

    # 右: ICIR柱状图
    ax = axes[1]
    names = [r['name'] for r in all_results]
    icirs = [abs(r['ic_stats']['icir']) for r in all_results]
    colors_bar = plt.cm.viridis(np.linspace(0.15, 0.9, len(names)))
    bars = ax.bar(range(len(names)), icirs, color=colors_bar, edgecolor='black', lw=0.5)
    for bar, val in zip(bars, icirs):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(y=0.3, color='gray', lw=0.5, ls='--', label='ICIR=0.3 threshold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('|ICIR|')
    ax.set_title('Factor |ICIR| Comparison')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Cross-Factor Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/cross_factor_comparison.png")
    plt.close(fig)
    print(f"[图表] {output_dir}/cross_factor_comparison.png")

    # ---- 图2: 因子多空收益相关性热力图 ----
    ls_curves = {}
    for r in all_results:
        if not r['ls_df'].empty:
            ls_curves[r['name']] = r['ls_df']['ls_ret'].values

    if len(ls_curves) >= 2:
        ls_series = {k: pd.Series(v) for k, v in ls_curves.items()}
        ls_df_all = pd.concat(ls_series, axis=1)
        corr_matrix = ls_df_all.corr()

        fig, ax = plt.subplots(figsize=(max(6, len(ls_curves) * 1.3),
                                       max(5, len(ls_curves) * 1.1)))
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                val = corr_matrix.values[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if abs(val) > 0.6 else 'black')
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_yticklabels(corr_matrix.index, fontsize=8)
        ax.set_title('Factor Long-Short Return Correlation')
        plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
        plt.tight_layout()
        fig.savefig(f"{output_dir}/factor_correlation_heatmap.png")
        plt.close(fig)
        print(f"[图表] {output_dir}/factor_correlation_heatmap.png")


# ============================================================
# 11. 工具函数
# ============================================================

def _max_drawdown(cumret: np.ndarray) -> float:
    """计算最大回撤（从峰值到谷底）。"""
    peak = np.maximum.accumulate(cumret)
    drawdown = (cumret - peak) / (1.0 + peak)
    return float(np.min(drawdown))


def _parse_factor_file(filepath: str) -> list:
    """
    从文件解析因子表达式。

    支持格式:
    - Rank N | fitness=X | nodes=Y | height=Z
      EXPR
    - name = EXPR
    - 纯表达式（每行一个）

    Returns
    -------
    list of (name, expr_str)
    """
    factors = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        # 格式1: "Rank N | ..."  → 下一行是表达式
        if line.startswith('Rank') and '|' in line:
            # 提取 rank 编号作为名称
            parts = line.split('|')
            rank_part = parts[0].strip()  # "Rank 1"
            name = rank_part.replace(' ', '_').lower()  # "Rank_1" → "rank_1"
            # 下一行是表达式
            if i + 1 < len(lines):
                expr = lines[i + 1].strip()
                if expr and not expr.startswith('#'):
                    factors.append((name, expr))
            i += 2
            continue

        # 格式2: "name = EXPR"
        if '=' in line and '(' in line:
            parts = line.split('=', 1)
            name = parts[0].strip()
            expr = parts[1].strip()
            if expr:
                factors.append((name, expr))
            i += 1
            continue

        # 格式3: 纯表达式
        if '(' in line and not line.startswith('#'):
            factors.append((f"factor_{len(factors)+1:03d}", line))

        i += 1

    return factors


# ============================================================
# 12. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='独立因子分析与可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 分析单个因子表达式
  python factor_analysis_standalone.py --expr "STD(NEQ(WMA(DECAYLINEAR(GAP, 230), 32), DBM(SIGN(RETURN), PRICE_POS)), 5)"

  # 从文件批量分析
  python factor_analysis_standalone.py --file result/best_factors.txt

  # 分析HoF所有因子并生成对比图
  python factor_analysis_standalone.py --hof result/gp_checkpoint.pkl --compare

  # 自定义输出目录和前向收益周期
  python factor_analysis_standalone.py --expr "MEAN(CLOSE, 20)" --output my_results --fwd 10
        """
    )

    # 输入源（三选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--expr', type=str,
                             help='DEAP格式的单个因子表达式字符串')
    input_group.add_argument('--file', type=str,
                             help='包含因子表达式的文件路径（支持best_factors.txt格式）')
    input_group.add_argument('--hof', type=str,
                             help='GP checkpoint文件路径（分析Hall of Fame中所有因子）')

    # 可选参数
    parser.add_argument('--name', type=str, default=None,
                        help='因子名称（用于图表标题，仅--expr模式）')
    parser.add_argument('--output', type=str, default='result/analysis',
                        help='输出目录（默认: result/analysis）')
    parser.add_argument('--fwd', type=int, default=5,
                        help='前向收益周期（默认: 5个交易日）')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='跳过因子预处理（去极值/填充/标准化）')
    parser.add_argument('--winsorize', type=float, default=5.0,
                        help='MAD去极值倍数（默认: 5.0）')
    parser.add_argument('--compare', action='store_true',
                        help='生成多因子横向对比图表（仅--file和--hof模式）')
    parser.add_argument('--max-factors', type=int, default=10,
                        help='最大分析因子数（默认: 10，用于--file和--hof模式）')

    args = parser.parse_args()

    # ---- 加载数据 ----
    print("=" * 60)
    print("  独立因子分析与可视化工具")
    print("=" * 60)
    print("\n[加载] 正在加载数据...")
    data_3d, stock_codes, trade_dates, feature_cols = load_data()

    # ---- 构建原语集 ----
    print("\n[构建] 正在构建原语集...")
    pset = build_pset_from_features(feature_cols)
    print(f"  原语集已构建 (特征数={len(feature_cols)})")

    # ---- 创建分析器 ----
    analyzer = FactorAnalyzer(
        data_3d, feature_cols, trade_dates, stock_codes, pset,
        forward_period=args.fwd
    )

    # ---- 收集待分析因子 ----
    factors_to_analyze = []

    if args.expr:
        name = args.name or "CustomFactor"
        factors_to_analyze.append((name, args.expr))

    elif args.file:
        parsed = _parse_factor_file(args.file)
        if not parsed:
            print(f"[错误] 未能从文件解析到有效表达式: {args.file}")
            sys.exit(1)
        print(f"\n[解析] 从文件解析到 {len(parsed)} 个因子表达式")
        factors_to_analyze = parsed[:args.max_factors]

    elif args.hof:
        import pickle
        if not os.path.exists(args.hof):
            print(f"[错误] Checkpoint文件不存在: {args.hof}")
            sys.exit(1)
        print(f"\n[加载] 正在加载checkpoint: {args.hof}")
        with open(args.hof, 'rb') as f:
            ckpt = pickle.load(f)
        hof = ckpt['halloffame']
        print(f"  HoF包含 {len(hof)} 个因子")
        for i, ind in enumerate(hof[:args.max_factors]):
            name = f"HoF_Rank{i+1}"
            expr = str(ind)
            factors_to_analyze.append((name, expr))

    print(f"\n[分析] 共 {len(factors_to_analyze)} 个因子待分析\n")

    # ---- 逐个分析 ----
    all_results = []
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    for name, expr_str in factors_to_analyze:
        try:
            results = analyzer.run(
                expr_str, name=name,
                preprocess=not args.no_preprocess,
                winsorize_std=args.winsorize,
            )
            analyzer.print_report(results)

            # 生成单因子图表 (每个因子独立文件夹)
            safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            factor_dir = os.path.join(output_dir, safe_name)
            os.makedirs(factor_dir, exist_ok=True)
            analyzer.plot_all(results, output_dir=factor_dir, prefix="")

            all_results.append(results)

        except Exception as e:
            print(f"[错误] 因子 '{name}' 分析失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ---- 多因子横向对比 ----
    if args.compare and len(all_results) >= 2:
        multi_factor_compare(analyzer, all_results, output_dir=output_dir)

    # ---- 保存汇总CSV ----
    if all_results:
        summary_rows = []
        for r in all_results:
            ic = r['ic_stats']
            summary_rows.append({
                'Name': r['name'],
                'Expression': r['expr_str'][:200],
                'IC_Mean': ic['ic_mean'],
                'IC_Std': ic['ic_std'],
                'ICIR': ic['icir'],
                'IC_Pos_Ratio': ic['pos_ratio'],
                't_stat': ic['t_stat'],
                'IC_N': ic['n_obs'],
                'Q_Spread': r['q_spread'],
                'LS_Mean': r['ls_mean'],
                'LS_IR': r['ls_ir'],
                'LS_Sharpe': r['ls_sharpe'],
                'LS_WinRate': r['ls_win'],
                'LS_MaxDD': r['ls_maxdd'],
                'Turnover': r['turnover'],
                'IC_AC_5d': r['ac_5d'],
                'IC_AC_20d': r['ac_20d'],
                'Nodes': r['nodes'],
                'Height': r['height'],
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values('ICIR', ascending=False, key=abs)
        csv_path = f"{output_dir}/factor_analysis_summary.csv"
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n[保存] 分析汇总: {csv_path}")
        print(f"  {len(summary_df)} 个因子 × {len(summary_df.columns)} 个指标")

    # ---- 完成 ----
    print(f"\n{'='*60}")
    print(f"  分析完成！结果已保存至: {output_dir}/")
    print(f"{'='*60}")

    return analyzer, all_results


if __name__ == '__main__':
    main()
