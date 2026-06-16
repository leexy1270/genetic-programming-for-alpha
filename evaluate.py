"""
evaluate.py — GP个体评估模块

对遗传编程产生的因子表达式进行评估：
1. 将表达式树编译为可调用函数
2. 在3D numpy数组 (股票×日期×特征) 上计算每只股票的因子值
3. 计算截面Rank IC序列
4. 以ICIR (IC均值/IC标准差) 作为适应度，并施加复杂度惩罚
"""

import numpy as np
from scipy.stats import rankdata
from deap import gp
from parameters import FORWARD_RETURN, MIN_STOCKS, MIN_DAYS, PARSIMONY_C

# 与 prepare_data 中 feature_cols 保持一致的默认列名
_DEFAULT_FEATURE_COLS = [
    'OPEN', 'HIGH', 'LOW', 'CLOSE', 'RETURN', 'VOLUME',
    'VWAP', 'VWAP_DEV', 'BODY', 'UPPER_SHADOW', 'LOWER_SHADOW',
    'PRICE_POS', 'AMP', 'GAP', 'VOL_CHG',
]


def _compute_forward_returns(data, ret_idx, forward_period):
    """
    从每日收益率计算未来N日 forward return。

    Parameters
    ----------
    data : np.ndarray, shape (n_stocks, n_dates, n_features)
    ret_idx : int
        RETURN 列在 axis-2 上的索引。
    forward_period : int
        未来 N 日。

    Returns
    -------
    forward_ret : np.ndarray, shape (n_stocks, n_dates)
        forward_ret[i, t] = 第 i 只股票在第 t 日收盘买入，
        持有 forward_period 日后的累计收益率（小数形式）。
        末尾 forward_period 天为 NaN。
    """
    n_stocks, n_dates, _ = data.shape
    # RETURN 列是百分比形式 (tushare pct_chg)，转为小数
    daily_ret = data[:, :, ret_idx] / 100.0

    forward_ret = np.full((n_stocks, n_dates), np.nan, dtype=np.float64)
    if n_dates <= forward_period:
        return forward_ret

    # 对数收益率累加 → exp 还原，避免浮点累积误差
    log_ret = np.log(1.0 + daily_ret)
    for t in range(n_dates - forward_period):
        forward_ret[:, t] = np.exp(
            np.sum(log_ret[:, t + 1 : t + 1 + forward_period], axis=1)
        ) - 1.0

    return forward_ret


def _rank_ic(factor_cross, forward_cross):
    """
    计算单期截面 Rank IC（Spearman 秩相关系数）。

    Parameters
    ----------
    factor_cross : np.ndarray, 1D
        某一日所有股票的因子值。
    forward_cross : np.ndarray, 1D
        对应的未来收益。

    Returns
    -------
    ic : float or np.nan
    """
    # 剔除 NaN / Inf
    mask = ~(
        np.isnan(factor_cross)
        | np.isnan(forward_cross)
        | np.isinf(factor_cross)
        | np.isinf(forward_cross)
    )
    if mask.sum() < max(MIN_STOCKS, 5):
        return np.nan

    fv = factor_cross[mask]
    fwd = forward_cross[mask]

    # 排名（平均排名处理平局）
    fv_rank = rankdata(fv)
    fwd_rank = rankdata(fwd)

    # 排名方差为 0 → 因子值或收益完全同质，IC 无定义
    if np.std(fv_rank) < 1e-12 or np.std(fwd_rank) < 1e-12:
        return np.nan

    # Pearson on ranks = Spearman
    ic = np.corrcoef(fv_rank, fwd_rank)[0, 1]
    return ic if not np.isnan(ic) else np.nan


def evaluate(individual, data,pset, feature_cols=None,):
    """
    评估一个 GP 个体的适应度。

    流程:
    1. 编译表达式树 → 可调用函数 func(*features)
    2. 对每只股票，传入时序特征数组，计算因子值序列
    3. 构建 forward return 面板
    4. 逐日计算截面 Rank IC
    5. 计算 ICIR = mean(IC) / std(IC)
    6. 施加树复杂度惩罚: fitness = ICIR - PARSIMONY_C × nodes

    Parameters
    ----------
    individual : gp.PrimitiveTree
        GP 表达式树。
    data : np.ndarray
        3D 数组，shape = (n_stocks, n_dates, n_features)。
        Axis-0: 股票, Axis-1: 日期, Axis-2: 特征。
    pset : gp.PrimitiveSet
        原语集合，用于编译表达式。
    feature_cols : list of str, optional
        Axis-2 对应的特征名列表。默认使用 prepare_data 中的 15 个特征。

    Returns
    -------
    (fitness_value,) : tuple of float
        适应度值，越大越好。无效因子返回 (-999.0,)。
    """

    # ---- 0. 确保 feature_cols 有效 ----
    if feature_cols is None:
        feature_cols = _DEFAULT_FEATURE_COLS

    # ---- 1. 编译表达式 ----
    try:
        func = gp.compile(individual, pset)
    except Exception:
        return (-999.0,)

    n_stocks, n_dates, n_features = data.shape

    # ---- 2. 计算因子值面板 ----
    factor_values = np.full((n_stocks, n_dates), np.nan, dtype=np.float64)

    args = [data[:,:, j] for j in range(n_features)]

    try:
        fv = func(*args)
        fv = np.atleast_1d(np.squeeze(fv))
        if fv.ndim == 2 and fv.shape[1] == n_dates:
            factor_values = fv
    except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError, TypeError):
        pass

    # 快速检查：是否所有因子值都无效
    if np.all(np.isnan(factor_values)):
        return (-999.0,)

    # ---- 3. 构建 forward return ----
    ret_idx = feature_cols.index('RETURN') 
    forward_ret = _compute_forward_returns(data, ret_idx, FORWARD_RETURN)

    # ---- 4. 逐日截面 Rank IC ----
    ic_list = []
    for t in range(n_dates):
        ic = _rank_ic(factor_values[:, t], forward_ret[:, t])
        if not np.isnan(ic):
            ic_list.append(ic)

    # ---- 5. ICIR 计算 ----
    if len(ic_list) < MIN_DAYS:
        return (-999.0,)

    ic_array = np.array(ic_list)
    ic_mean = np.mean(ic_array)
    ic_std = np.std(ic_array, ddof=1)   # 样本标准差

    if ic_std < 1e-12:
        # IC 序列几乎不变 → 因子缺乏区分度
        return (0.0,)

    icir = abs(ic_mean) / ic_std      # 取绝对值，稳定负相关也是好因子

    # ---- 6. 复杂度惩罚（防膨胀） ----
    fitness = icir - PARSIMONY_C * len(individual)

    return (float(fitness),)
