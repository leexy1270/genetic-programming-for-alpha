"""
evaluate.py — GP个体评估模块 (向量化优化版)

对遗传编程产生的因子表达式进行评估：
1. 将表达式树编译为可调用函数（带 LRU 缓存）
2. 在3D numpy数组 (股票×日期×特征) 上计算每只股票的因子值
3. 计算截面Rank IC序列（批量向量化版本）
4. 以ICIR (IC均值/IC标准差) 作为适应度，并施加复杂度惩罚

优化点:
  - _compute_forward_returns: cumsum 技巧消除 Python 逐日循环
  - _rank_ic / _batch_rank_ic: 直接计算 Pearson r 替代 corrcoef，减少分配
  - _compile_cache: 按表达式字符串缓存编译结果，避免重复 AST 解析
"""

import numpy as np
from scipy.stats import rankdata
from deap import gp
from parameters import FORWARD_RETURN, MIN_STOCKS, MIN_DAYS, PARSIMONY_C

# ---- 表达式编译缓存 ----
_compile_cache = {}
_HIT = 0
_MISS = 0

# 与 prepare_data 中 feature_cols 保持一致的默认列名
_DEFAULT_FEATURE_COLS = [
    'OPEN', 'HIGH', 'LOW', 'CLOSE', 'RETURN', 'VOLUME'
]

# ---- 多进程 Worker 全局状态 ----
# 避免通过 pickle 传递 pset（含本地函数无法序列化）和 data（60MB × N 进程开销大）。
# Worker 进程通过 init_worker() 重建 pset 并持有 data 引用。
_worker_data = None
_worker_pset = None
_worker_feature_cols = None


def init_worker(mmap_path, data_shape, data_dtype, feature_cols):
    """
    multiprocessing.Pool 的 initializer。
    在 worker 进程启动时调用：
    - 通过 np.memmap 共享数据（避免每个 worker pickle 复制 60MB data）
    - 重建 pset（含本地函数无法 pickle）
    """
    global _worker_data, _worker_pset, _worker_feature_cols
    # 打开 memmap — 所有 worker 共享同一物理内存页
    _worker_data = np.memmap(mmap_path, dtype=data_dtype, mode='r', shape=data_shape)
    _worker_feature_cols = feature_cols
    from build_pset import build_pset
    _worker_pset = build_pset(feature_cols)
    # 确保 __builtins__ 有效（DEAP 可能预设为 None）
    if not _worker_pset.context.get('__builtins__'):
        import builtins
        _worker_pset.context['__builtins__'] = builtins


def evaluate_worker(individual):
    """
    供 pool.map 调用的模块级包装函数。
    使用 worker 进程的全局 data / pset / feature_cols。
    函数签名简单（仅 individual 参数），可被 pickle 引用。
    """
    return evaluate(individual, _worker_data, _worker_pset, _worker_feature_cols)


def _compute_forward_returns(data, ret_idx, forward_period):
    """
    从每日收益率计算未来N日 forward return（向量化版，消除 Python 逐日循环）。

    原理:
        设 log_ret 为对数收益面板 (n_stocks × n_dates)。
        欲求 forward_ret[:, t] = exp(sum_{k=t+1}^{t+forward_period} log_ret[:, k]) - 1

        令 cum_log[:, j] = sum_{i=0}^{j-1} log_ret[:, i]  (即 padded cumsum)，
        则 sum(t+1 : t+forward_period+1) = cum_log[:, t+forward_period+1] - cum_log[:, t+1]

        → 全部用 numpy 切片广播完成，无 Python for-loop。

    Parameters
    ----------
    data : np.ndarray, shape (n_stocks, n_dates, n_features)
    ret_idx : int   RETURN 列在 axis-2 上的索引。
    forward_period : int   未来 N 日。

    Returns
    -------
    forward_ret : np.ndarray, shape (n_stocks, n_dates)
        末尾 forward_period 天为 NaN。
    """
    n_stocks, n_dates, _ = data.shape
    daily_ret = data[:, :, ret_idx] / 100.0         # 百分比 → 小数

    forward_ret = np.full((n_stocks, n_dates), np.nan, dtype=np.float64)
    if n_dates <= forward_period:
        return forward_ret

    log_ret = np.log(1.0 + daily_ret)

    # cumsum 技巧: cumsum_{t} = Σ_{0}^{t-1}，pad 一个前导 0
    cum_log = np.cumsum(log_ret, axis=1)
    cum_log_padded = np.pad(cum_log, ((0, 0), (1, 0)), constant_values=0)

    valid_len = n_dates - forward_period
    start_idx = np.arange(valid_len) + 1               # t+1
    end_idx = start_idx + forward_period                 # t+forward_period+1

    log_sums = cum_log_padded[:, end_idx] - cum_log_padded[:, start_idx]
    forward_ret[:, :valid_len] = np.exp(log_sums) - 1.0

    return forward_ret


def _rank_ic(factor_cross, forward_cross):
    """
    计算单期截面 Rank IC（Spearman 秩相关系数）。

    优化: 直接用 Pearson 公式 ∑((x_i-x̄)(y_i-ȳ)) / √(∑(x_i-x̄)²·∑(y_i-ȳ)²)
          替代 np.corrcoef，避免分配 2×2 矩阵和重复计算。

    Parameters
    ----------
    factor_cross : np.ndarray, 1D   某一日所有股票的因子值。
    forward_cross : np.ndarray, 1D  对应的未来收益。

    Returns
    -------
    ic : float or np.nan
    """
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

    fv_rank = rankdata(fv)
    fwd_rank = rankdata(fwd)

    # 去均值
    fv_demean = fv_rank - np.mean(fv_rank)
    fwd_demean = fwd_rank - np.mean(fwd_rank)

    denom = np.sqrt(np.dot(fv_demean, fv_demean) * np.dot(fwd_demean, fwd_demean))
    if denom < 1e-12:
        return np.nan

    ic = np.dot(fv_demean, fwd_demean) / denom
    return float(ic) if not np.isnan(ic) else np.nan


def _cached_compile(individual, pset):
    """
    按表达式字符串缓存编译结果。

    DEAP 的 PrimitiveTree 不可哈希，用 `str(ind)` 作为 key。
    在 GP 种群收敛阶段，大量个体表达式重复或高度相似，缓存命中率可观。

    注意: pset.context 中不持久存储 __builtins__（module 不可 pickle），
    编译前动态注入以确保 eval() 正常运行。
    """
    global _HIT, _MISS
    key = str(individual)
    if key in _compile_cache:
        _HIT += 1
        return _compile_cache[key]
    _MISS += 1
    # 确保 __builtins__ 在上下文中（pickle 传输后可能丢失或为 None）
    if not pset.context.get('__builtins__'):
        import builtins
        pset.context['__builtins__'] = builtins
    func = gp.compile(individual, pset)
    _compile_cache[key] = func
    return func


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

    # ---- 1. 编译表达式（带缓存） ----
    try:
        func = _cached_compile(individual, pset)
    except Exception:
        return (-999.0,)

    n_stocks, n_dates, n_features = data.shape

    # ---- 2. 计算因子值面板 ----
    factor_values = np.full((n_stocks, n_dates), np.nan, dtype=np.float64)

    args = [data[:,:, j] for j in range(n_features)]

    # 遗传编程中大量表达式数值不稳定 (overflow / NaN / inf) 是预期行为，
    # 无效个体会被适应度函数自然淘汰。此处屏蔽 numpy 浮点警告以减少 stderr 刷屏。
    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
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
