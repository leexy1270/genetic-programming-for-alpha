"""
build_pset.py — 遗传编程原语集
"""

from deap import gp
import numpy as np
import random
from functools import partial


def build_pset(INPUT_COLS) -> gp.PrimitiveSet:
    """
    构建类型化原语集。

    所有数据算子输入/输出类型均为 np.ndarray (float64),
    窗口参数类型为 int, 标量常数为 float。
    """
    pset = gp.PrimitiveSetTyped("MAIN", [np.ndarray] * len(INPUT_COLS), np.ndarray)

    # 将 ARG0, ARG1, ... 重命名为特征列名
    for i, col_name in enumerate(INPUT_COLS):
        pset.renameArguments(**{f"ARG{i}": col_name})

    # ================================================================
    # 1. 常数
    # ================================================================
    # 随机浮点常数 (uniform -1 ~ 1) — GP 进化时随机采样
    pset.addEphemeralConstant("rand", partial(random.uniform, -1, 1), float)

    # 随机窗口参数 — GP 进化时从池中随机选取
    _WINDOW_POOL = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        12, 14, 15, 16, 18, 20, 21, 24, 26, 30,
        32, 37, 40, 50, 60, 80, 100, 120, 150, 180,
        200, 230, 240, 250, 252,
    ]
    pset.addEphemeralConstant("window", partial(random.choice, _WINDOW_POOL), int)

    # 固定窗口终端 — 支持从字符串解析表达式 (warm_start.txt / --expr)
    # 必须与 _WINDOW_POOL 一致，确保解析后的表达式在 GP 中可正常交叉/变异
    for n in _WINDOW_POOL:
        pset.addTerminal(n, int)

    # 固定标量常数
    for val in [0.0, 1.0, -1.0, 0.5, 100.0]:
        pset.addTerminal(val, float)

    # ================================================================
    # 2. 基本算术算子 (9)
    # ================================================================

    def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """保护除法: |b| < 1e-8 → 0"""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(b) > 1e-8, a / b, 0.0)

    def safe_sqrt(a: np.ndarray) -> np.ndarray:
        """保护平方根: 负数 → 0"""
        return np.sqrt(np.maximum(a, 0))

    def safe_log(a: np.ndarray) -> np.ndarray:
        """保护对数: < 1e-8 → 0"""
        return np.log(np.maximum(a, 1e-8))

    pset.addPrimitive(np.add, [np.ndarray, np.ndarray], np.ndarray, name="ADD")
    pset.addPrimitive(np.subtract, [np.ndarray, np.ndarray], np.ndarray, name="SUB")
    pset.addPrimitive(np.multiply, [np.ndarray, np.ndarray], np.ndarray, name="MUL")
    pset.addPrimitive(safe_div, [np.ndarray, np.ndarray], np.ndarray, name="DIV")
    pset.addPrimitive(np.negative, [np.ndarray], np.ndarray, name="NEG")
    pset.addPrimitive(np.abs, [np.ndarray], np.ndarray, name="ABS")
    pset.addPrimitive(safe_sqrt, [np.ndarray], np.ndarray, name="SQRT")
    pset.addPrimitive(np.square, [np.ndarray], np.ndarray, name="SQUARE")
    pset.addPrimitive(safe_log, [np.ndarray], np.ndarray, name="LOG")

    # ================================================================
    # 3. 时序滚动算子 (14, 均需要 window:int 参数)
    #    这些是 alpha 因子的核心 — WQ101 中 80%+ 的因子依赖它们
    # ================================================================

    def ts_delta(arr: np.ndarray, window: int) -> np.ndarray:
        """arr[t] - arr[t-window]"""
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        out[:, window:] = arr[:, window:] - arr[:, :-window]
        return out

    def ts_delay(arr: np.ndarray, window: int) -> np.ndarray:
        """arr[t-window]"""
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        out[:, window:] = arr[:, :-window]
        return out

    def ts_sum(arr: np.ndarray, window: int) -> np.ndarray:
        """滚动求和"""
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n <= window or window <= 0:
            return out
        cum = np.nancumsum(arr, axis=1)
        out[:, window:] = cum[:, window:] - cum[:, :-window]
        return out

    def ts_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """滚动均值"""
        return ts_sum(arr, window) / window

    def ts_std(arr: np.ndarray, window: int) -> np.ndarray:
        """滚动标准差"""
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n <= window or window <= 0:
            return out
        cum = np.nancumsum(arr, axis=1)
        cum2 = np.nancumsum(arr * arr, axis=1)
        sum_x = cum[:, window:] - cum[:, :-window]
        sum_x2 = cum2[:, window:] - cum2[:, :-window]
        var = sum_x2 / window - (sum_x / window) ** 2
        out[:, window:] = np.sqrt(np.maximum(var, 0))
        return out

    def ts_min(arr: np.ndarray, window: int) -> np.ndarray:
        """滚动最小值"""
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=1)
        out[:, window - 1:] = win.min(axis=-1)
        return out

    def ts_max(arr: np.ndarray, window: int) -> np.ndarray:
        """滚动最大值"""
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=1)
        out[:, window - 1:] = win.max(axis=-1)
        return out

    def ts_rank(arr: np.ndarray, window: int) -> np.ndarray:
        """滚动排名分位数 (0~1)"""
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 1:
            return out
        # 移动最后一维到末尾方便 sliding_window_view
        arr_moved = np.moveaxis(arr, 1, -1)
        out_moved = np.moveaxis(out, 1, -1)
        win = np.lib.stride_tricks.sliding_window_view(arr_moved, window, axis=-1)
        last = win[..., -1:]
        rank_count = (win < last).sum(axis=-1)
        out_moved[..., window - 1:] = rank_count / (window - 1)
        return np.moveaxis(out_moved, -1, 1)

    def ts_roc(arr: np.ndarray, window: int) -> np.ndarray:
        """变化率: arr[t] / arr[t-window] - 1"""
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n <= window or window <= 0:
            return out
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, window:] = arr[:, window:] / np.maximum(np.abs(arr[:, :-window]), 1e-8) - 1.0
        return out

    # ZSCORE 已移除 — 等价于 DIV(SUB(x, MEAN(x,N)), STD(x,N)), 让GP自行组合
    # def ts_zscore(arr, window): ...

    def ts_decay_linear(arr: np.ndarray, window: int) -> np.ndarray:
        """线性衰减加权移动平均 (权重 1,2,...,window)"""
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        weights = np.arange(1, window + 1, dtype=np.float64)
        weights /= weights.sum()
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=1)
        out[:, window - 1:] = np.tensordot(win, weights, axes=([-1], [0]))
        return out

    def ts_wma(arr: np.ndarray, window: int) -> np.ndarray:
        """加权移动平均 (同 DECAYLINEAR, 别名保留兼容性)"""
        return ts_decay_linear(arr, window)

    def ts_prod(arr: np.ndarray, window: int) -> np.ndarray:
        """滚动乘积 (对数空间计算避免溢出)"""
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        log_arr = np.log(np.maximum(np.abs(arr), 1e-12))
        s = np.nancumsum(log_arr, axis=1)
        out[:, window:] = np.exp(s[:, window:] - s[:, :-window])
        return out

    def ts_cumsum(arr: np.ndarray) -> np.ndarray:
        """累计求和 (无窗口参数)"""
        return np.cumsum(np.nan_to_num(arr, nan=0.0), axis=1)

    # 注册时序滚动算子
    pset.addPrimitive(ts_delta, [np.ndarray, int], np.ndarray, name="DELTA")
    pset.addPrimitive(ts_delay, [np.ndarray, int], np.ndarray, name="DELAY")
    pset.addPrimitive(ts_sum, [np.ndarray, int], np.ndarray, name="SUM")
    pset.addPrimitive(ts_mean, [np.ndarray, int], np.ndarray, name="MEAN")
    pset.addPrimitive(ts_std, [np.ndarray, int], np.ndarray, name="STD")
    pset.addPrimitive(ts_min, [np.ndarray, int], np.ndarray, name="TSMIN")
    pset.addPrimitive(ts_max, [np.ndarray, int], np.ndarray, name="TSMAX")
    pset.addPrimitive(ts_rank, [np.ndarray, int], np.ndarray, name="TSRANK")
    pset.addPrimitive(ts_roc, [np.ndarray, int], np.ndarray, name="ROC")
    # pset.addPrimitive(ts_zscore, [np.ndarray, int], np.ndarray, name="ZSCORE")
    pset.addPrimitive(ts_decay_linear, [np.ndarray, int], np.ndarray, name="DECAYLINEAR")
    pset.addPrimitive(ts_wma, [np.ndarray, int], np.ndarray, name="WMA")
    pset.addPrimitive(ts_prod, [np.ndarray, int], np.ndarray, name="PROD")
    pset.addPrimitive(ts_cumsum, [np.ndarray], np.ndarray, name="SUMAC")

    # ================================================================
    # 4. 截面算子 (3)
    #    跨品种/股票同步计算, 是 alpha 另一核心维度
    # ================================================================

    def cs_rank(arr: np.ndarray) -> np.ndarray:
        """截面排名 (0~1 分位数, 逐日计算)"""
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        for t in range(arr.shape[1]):
            cross = arr[:, t]
            mask = ~(np.isnan(cross) | np.isinf(cross))
            n = mask.sum()
            if n < 2:
                continue
            order = np.argsort(cross[mask])
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1, dtype=np.float64)
            out[mask, t] = ranks / n
        return out

    def cs_zscore(arr: np.ndarray) -> np.ndarray:
        """截面标准化 (逐日 Z-Score)"""
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        for t in range(arr.shape[1]):
            cross = arr[:, t]
            mask = ~(np.isnan(cross) | np.isinf(cross))
            if mask.sum() < 2:
                continue
            vals = cross[mask]
            mu, sigma = vals.mean(), vals.std(ddof=1)
            if sigma < 1e-12:
                out[mask, t] = 0.0
            else:
                out[mask, t] = (vals - mu) / sigma
        return out

    def cs_scale(arr: np.ndarray) -> np.ndarray:
        """截面缩放到单位总和 (逐日 sum(abs(x)) = 1)"""
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        for t in range(arr.shape[1]):
            cross = arr[:, t]
            mask = ~(np.isnan(cross) | np.isinf(cross))
            if mask.sum() < 2:
                continue
            abs_sum = np.abs(cross[mask]).sum()
            if abs_sum > 1e-12:
                out[mask, t] = cross[mask] / abs_sum
            else:
                out[mask, t] = 0.0
        return out

    pset.addPrimitive(cs_rank, [np.ndarray], np.ndarray, name="CS_RANK")
    pset.addPrimitive(cs_zscore, [np.ndarray], np.ndarray, name="CS_ZSCORE")
    pset.addPrimitive(cs_scale, [np.ndarray], np.ndarray, name="CS_SCALE")

    # ================================================================
    # 5. 配对/回归算子 (4)
    #    两个序列的滚动统计关系, WQ101 中特征性使用
    # ================================================================

    def ts_corr(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
        """滚动 Pearson 相关系数"""
        n = a.shape[1]
        out = np.full(a.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        aw = np.lib.stride_tricks.sliding_window_view(a, window, axis=1)
        bw = np.lib.stride_tricks.sliding_window_view(b, window, axis=1)
        am = aw - aw.mean(axis=-1, keepdims=True)
        bm = bw - bw.mean(axis=-1, keepdims=True)
        num = (am * bm).sum(axis=-1)
        den = np.sqrt((am * am).sum(axis=-1) * (bm * bm).sum(axis=-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, window - 1:] = np.where(np.abs(den) > 1e-8, num / den, 0.0)
        return out

    def ts_cov(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
        """滚动协方差"""
        n = a.shape[1]
        out = np.full(a.shape, np.nan, dtype=np.float64)
        if n < window or window <= 1:
            return out
        aw = np.lib.stride_tricks.sliding_window_view(a, window, axis=1)
        bw = np.lib.stride_tricks.sliding_window_view(b, window, axis=1)
        am = aw - aw.mean(axis=-1, keepdims=True)
        bm = bw - bw.mean(axis=-1, keepdims=True)
        out[:, window - 1:] = (am * bm).sum(axis=-1) / (window - 1)
        return out

    def ts_regbeta(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
        """滚动回归系数 β = Cov(A,B) / Var(B)"""
        n = a.shape[1]
        out = np.full(a.shape, np.nan, dtype=np.float64)
        if n < window or window <= 1:
            return out
        aw = np.lib.stride_tricks.sliding_window_view(a, window, axis=1)
        bw = np.lib.stride_tricks.sliding_window_view(b, window, axis=1)
        am = aw - aw.mean(axis=-1, keepdims=True)
        bm = bw - bw.mean(axis=-1, keepdims=True)
        num = (am * bm).sum(axis=-1)
        den = (bm * bm).sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, window - 1:] = np.where(np.abs(den) > 1e-8, num / den, 0.0)
        return out

    def ts_regresi(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
        """滚动回归残差 ε = A - (α + β·B)"""
        n = a.shape[1]
        out = np.full(a.shape, np.nan, dtype=np.float64)
        if n < window or window <= 1:
            return out
        aw = np.lib.stride_tricks.sliding_window_view(a, window, axis=1)
        bw = np.lib.stride_tricks.sliding_window_view(b, window, axis=1)
        am = aw - aw.mean(axis=-1, keepdims=True)
        bm = bw - bw.mean(axis=-1, keepdims=True)
        num = (am * bm).sum(axis=-1)
        den = (bm * bm).sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(np.abs(den) > 1e-8, num / den, 0.0)
        alpha = aw.mean(axis=-1) - beta * bw.mean(axis=-1)
        resid = aw[..., -1] - (alpha + beta * bw[..., -1])
        out[:, window - 1:] = resid
        return out

    pset.addPrimitive(ts_corr, [np.ndarray, np.ndarray, int], np.ndarray, name="CORR")
    pset.addPrimitive(ts_cov, [np.ndarray, np.ndarray, int], np.ndarray, name="COVIANCE")
    #pset.addPrimitive(ts_regbeta, [np.ndarray, np.ndarray, int], np.ndarray, name="REGBETA")
    #pset.addPrimitive(ts_regresi, [np.ndarray, np.ndarray, int], np.ndarray, name="REGRESI")

    # ================================================================
    # 6. 高级时序/形状算子 (5)
    # ================================================================

    def ts_argmax(arr: np.ndarray, window: int) -> np.ndarray:
        """距滚动最大值的天数"""
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=1)
        out[:, window - 1:] = window - 1 - win.argmax(axis=-1)
        return out

    def ts_argmin(arr: np.ndarray, window: int) -> np.ndarray:
        """距滚动最小值的天数"""
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=1)
        out[:, window - 1:] = window - 1 - win.argmin(axis=-1)
        return out

    def ts_signed_power(arr: np.ndarray, exponent: int) -> np.ndarray:
        """sign(x) * |x|^a (WQ101 专用)"""
        with np.errstate(over="ignore", invalid="ignore"):
            return np.sign(arr) * (np.abs(arr) ** exponent)

    pset.addPrimitive(ts_argmax, [np.ndarray, int], np.ndarray, name="TSARGMAX")
    pset.addPrimitive(ts_argmin, [np.ndarray, int], np.ndarray, name="TSARGMIN")
    pset.addPrimitive(ts_signed_power, [np.ndarray, int], np.ndarray, name="SIGNED_POWER")
    pset.addPrimitive(np.sign, [np.ndarray], np.ndarray, name="SIGN")
    pset.addPrimitive(np.tanh, [np.ndarray], np.ndarray, name="TANH")

    # ================================================================
    # 7. 二元选择 (2)
    # ================================================================
    pset.addPrimitive(np.maximum, [np.ndarray, np.ndarray], np.ndarray, name="MAX2")
    pset.addPrimitive(np.minimum, [np.ndarray, np.ndarray], np.ndarray, name="MIN2")

    # ================================================================
    # 8. 技术指标算子 (4)
    #    True Range + 日内动量 — WQ101 中高频使用
    # ================================================================

    def ts_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """True Range = MAX(H-L, |H-C[-1]|, |L-C[-1]|)"""
        close_lag1 = ts_delay(close, 1)
        a = high - low
        b = np.abs(high - close_lag1)
        c = np.abs(low - close_lag1)
        return np.maximum(np.maximum(a, b), c)

    def ts_sma(arr: np.ndarray, n: int, m: int) -> np.ndarray:
        """指数移动平均 (EMA): α=m/n, SMA = α*x + (1-α)*SMA[-1]"""
        if n <= 0:
            n = 1
        alpha = max(0.0, min(1.0, m / n))
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        for i in range(arr.shape[0]):
            row = arr[i]
            first = np.where(~np.isnan(row))[0]
            if len(first) == 0:
                continue
            start = first[0]
            out[i, start] = row[start]
            for t in range(start + 1, len(row)):
                prev = out[i, t - 1]
                cur = row[t]
                if np.isnan(cur):
                    out[i, t] = prev
                elif np.isnan(prev):
                    out[i, t] = cur
                else:
                    out[i, t] = alpha * cur + (1.0 - alpha) * prev
        return out

    # DTM/DBM 已移除 — 使用 np.where 条件判断产生 0-or-value 稀疏信号
    # def ts_dtm(open_, high): return np.where(open_ <= open_lag1, 0.0, max(...))
    # def ts_dbm(open_, low):   return np.where(open_ >= open_lag1, 0.0, max(...))

    pset.addPrimitive(ts_tr, [np.ndarray, np.ndarray, np.ndarray], np.ndarray, name="TR")
    pset.addPrimitive(ts_sma, [np.ndarray, int, int], np.ndarray, name="SMA")
    # pset.addPrimitive(ts_dtm, [np.ndarray, np.ndarray], np.ndarray, name="DTM")
    # pset.addPrimitive(ts_dbm, [np.ndarray, np.ndarray], np.ndarray, name="DBM")

    # ================================================================
    # 完成
    # ================================================================
    return pset


# ================================================================
# 安全生成函数 (类型安全回退)
# ================================================================

def genFullSafe(pset, min_, max_, type_=None):
    """genFull 安全版: 某类型无 primitives 时自动从 terminals 选取"""
    def condition(height, depth):
        return depth == height
    return _generate_safe(pset, min_, max_, condition, type_)


def genGrowSafe(pset, min_, max_, type_=None):
    """genGrow 安全版"""
    def condition(height, depth):
        return depth == height or \
            (depth >= min_ and random.random() < pset.terminalRatio)
    return _generate_safe(pset, min_, max_, condition, type_)


def genHalfAndHalfSafe(pset, min_, max_, type_=None):
    """genHalfAndHalf 安全版"""
    method = random.choice((genGrowSafe, genFullSafe))
    return method(pset, min_, max_, type_)


import sys as _sys


def _generate_safe(pset, min_, max_, condition, type_=None):
    """generate() 安全变体: primitives[type_] 为空时回退到 terminals"""
    from deap.gp import MetaEphemeral

    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth) or not pset.primitives[type_]:
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = _sys.exc_info()
                raise IndexError(
                    "gp.generate: no terminal of type '%s' available." % (type_,)
                ).with_traceback(traceback)
            if isinstance(term, MetaEphemeral):
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = _sys.exc_info()
                raise IndexError(
                    "gp.generate: no primitive of type '%s' available." % (type_,)
                ).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr
