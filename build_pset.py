from deap import gp
import numpy as np
import random
from functools import partial


def build_pset(INPUT_COLS) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSetTyped("MAIN", [np.ndarray]*len(INPUT_COLS),np.ndarray)
    # 不显式设置 pset.context["__builtins__"]，因为 module 对象无法被 pickle
    # Python 的 eval() 在 globals 缺少 __builtins__ 时会自动继承调用者的内置空间
    # _cached_compile 中会确保编译前 __builtins__ 就位

    for i, col_name in enumerate(INPUT_COLS):
        pset.renameArguments(**{f"ARG{i}": col_name})

    #1. ---------------- 常数 ---------------
    pset.addEphemeralConstant("rand", partial(random.uniform, -1, 1),float)
    pset.addEphemeralConstant('window',partial(random.randint,1,252),int)

    # 固定窗口常数 → terminals[int]
    for n in [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,20,21,24,26,30,32,37,40,50,60,80,100,120,150,180,200,230,240,250,252]:
        pset.addTerminal(n, int)

    pset.addTerminal(0.5, float)
    pset.addTerminal(0.618, float)
    pset.addTerminal(-1.0, float)
    pset.addTerminal(0.0, float)
    pset.addTerminal(1.0, float)
    pset.addTerminal(100.0, float)

    #2. =================================== 基本算子 ====================================
    pset.addPrimitive(np.abs, [np.ndarray], np.ndarray, name="ABS")

    def safe_sqrt(a: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(a, 0))
    pset.addPrimitive(safe_sqrt, [np.ndarray], np.ndarray, name="SQRT")

    def safe_log(a: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(a, 1e-8))

    pset.addPrimitive(safe_log, [np.ndarray], np.ndarray, name="LOG")

    pset.addPrimitive(np.sign, [np.ndarray], np.ndarray, name="SIGN")

    pset.addPrimitive(np.square, [np.ndarray], np.ndarray, name="SQUARE")

    pset.addPrimitive(np.add, [np.ndarray, np.ndarray], np.ndarray, name="ADD")

    pset.addPrimitive(np.subtract, [np.ndarray, np.ndarray], np.ndarray, name="SUB")

    pset.addPrimitive(np.multiply, [np.ndarray, np.ndarray], np.ndarray, name="MUL")

    def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(b) > 1e-8, a / b, 0.0)
    pset.addPrimitive(safe_div, [np.ndarray, np.ndarray], np.ndarray, name="DIV")

    pset.addPrimitive(np.maximum, [np.ndarray, np.ndarray], np.ndarray, name="MAX2")
    pset.addPrimitive(np.minimum, [np.ndarray, np.ndarray], np.ndarray, name="MIN2")


    #3. ================================== 截面算子 =================================
    # 截面排名：每天将所有股票在该因子上的值转为 0~1 分位数
    def cs_rank(arr: np.ndarray) -> np.ndarray:
        """cross-sectional rank (0~1), ignores NaN/Inf"""
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        for t in range(arr.shape[1]):
            cross = arr[:, t]
            mask = ~(np.isnan(cross) | np.isinf(cross))
            n = mask.sum()
            if n < 2:
                continue
            # argsort twice = rankdata("average"), fully vectorized
            order = np.argsort(cross[mask])
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1, dtype=np.float64)
            out[mask, t] = ranks / n
        return out
    pset.addPrimitive(cs_rank, [np.ndarray], np.ndarray, name="CS_RANK")

    # 截面标准化：每天 z-score (axis=0)
    def cs_zscore(arr: np.ndarray) -> np.ndarray:
        """cross-sectional z-score, ignores NaN/Inf"""
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
    pset.addPrimitive(cs_zscore, [np.ndarray], np.ndarray, name="CS_ZSCORE")

    #4. =================================== 二元时序算子 ==============================
    def ts_delta(arr: np.ndarray, window: int) -> np.ndarray:
        """arr[t] - arr[t-window]  |  O(1)"""
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        idx = [slice(None)] * arr.ndim
        idx[1] = slice(window,None)
        idx_prev = [slice(None)] * arr.ndim
        idx_prev[1] = slice(None,-window)
        out[tuple(idx)] = arr[tuple(idx)] - arr[tuple(idx_prev)]
        return out
    pset.addPrimitive(ts_delta,[np.ndarray,int],np.ndarray,name="DELTA")

    def ts_delay(arr:np.ndarray,window:int) ->np.ndarray:
        """arr[t-window]  |  O(1)"""
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        idx = [slice(None)] * arr.ndim
        idx_prev = [slice(None)] * arr.ndim
        idx[1] = slice(window, None)
        idx_prev[1] = slice(None, -window)
        out[tuple(idx)] = arr[tuple(idx_prev)]
        return out
    pset.addPrimitive(ts_delay,[np.ndarray,int],np.ndarray,name="DELAY")

    def ts_sum(arr:np.ndarray,window:int) ->np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n <= window or window <= 0:
            return out
        cum = np.cumsum(arr, axis=1)
        idx_hi = [slice(None)] * arr.ndim
        idx_lo = [slice(None)] * arr.ndim
        idx_hi[1] = slice(window, None)
        idx_lo[1] = slice(None, -window)
        out[tuple(idx_hi)] = cum[tuple(idx_hi)] - cum[tuple(idx_lo)]
        return out
    pset.addPrimitive(ts_sum,[np.ndarray,int],np.ndarray,name="SUM")

    def ts_mean(arr:np.ndarray,window:int) ->np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n <= window or window <= 0:
            return out
        cum = np.cumsum(arr, axis=1)
        idx_hi = [slice(None)] * arr.ndim
        idx_lo = [slice(None)] * arr.ndim
        idx_hi[1] = slice(window, None)
        idx_lo[1] = slice(None, -window)
        out[tuple(idx_hi)] = (cum[tuple(idx_hi)] - cum[tuple(idx_lo)]) / window
        return out
    pset.addPrimitive(ts_mean,[np.ndarray,int],np.ndarray,name="MEAN")

    def ts_std(arr:np.ndarray,window:int) ->np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n <= window or window <= 0:
            return out
        cum = np.cumsum(arr, axis=1)
        cum2 = np.cumsum(arr * arr, axis=1)
        idx_hi = [slice(None)] * arr.ndim
        idx_lo = [slice(None)] * arr.ndim
        idx_hi[1] = slice(window, None)
        idx_lo[1] = slice(None, -window)
        sum_x = cum[tuple(idx_hi)] - cum[tuple(idx_lo)]
        sum_x2 = cum2[tuple(idx_hi)] - cum2[tuple(idx_lo)]
        var = sum_x2 / window - (sum_x / window) ** 2
        out[tuple(idx_hi)] = np.sqrt(np.maximum(var, 0))
        return out
    pset.addPrimitive(ts_std,[np.ndarray,int],np.ndarray,name = "STD")

    def ts_min(arr:np.ndarray,window:int) ->np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=1)
        out_slice = [slice(None)] * arr.ndim
        out_slice[1] = slice(window - 1, None)
        out[tuple(out_slice)] = win.min(axis=-1)
        return out
    pset.addPrimitive(ts_min,[np.ndarray,int],np.ndarray,name="TSMIN")

    def ts_max(arr:np.ndarray,window:int) ->np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=1)
        out_slice = [slice(None)] * arr.ndim
        out_slice[1] = slice(window - 1, None)
        out[tuple(out_slice)] = win.max(axis=-1)
        return out
    pset.addPrimitive(ts_max,[np.ndarray,int],np.ndarray,"TSMAX")

    def ts_rank(arr:np.ndarray,window:int) ->np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 1:
            return out
        # 移动目标轴到最后，方便处理
        moved = 1 != -1 and 1 != arr.ndim - 1
        if moved:
            arr = np.moveaxis(arr, 1, -1)
            out = np.moveaxis(out, 1, -1)   # out 也要同步移动!
        # arr shape: (..., n)
        # sliding_window_view 沿最后一维 → (..., n-w+1, w)
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=-1)
        # win shape: (..., n-w+1, w)
        # 排名分位数 = (比最后一个元素小的个数) / (w-1)
        last = win[..., -1:]          # (..., n-w+1, 1)
        rank_count = (win < last).sum(axis=-1)  # (..., n-w+1)
        rank_pct = rank_count / (window - 1)
        out_slice = [slice(None)] * arr.ndim
        out_slice[-1] = slice(window - 1, None)
        out[tuple(out_slice)] = rank_pct
        if moved:
            out = np.moveaxis(out, -1, 1)
        return out
    pset.addPrimitive(ts_rank,[np.ndarray,int],np.ndarray,name='TSRANK')

    #5. --------------- 高级二元时序算子 (np.ndarray, int → np.ndarray) ---------------
    # ts_roc: rate of change = arr[t] / arr[t-window] - 1
    def ts_roc(arr: np.ndarray, window: int) -> np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n <= window or window <= 0:
            return out
        idx = [slice(None)] * arr.ndim
        idx_prev = [slice(None)] * arr.ndim
        idx[1] = slice(window, None)
        idx_prev[1] = slice(None, -window)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[tuple(idx)] = (
                arr[tuple(idx)] / np.maximum(np.abs(arr[tuple(idx_prev)]), 1e-8) - 1.0
            )
        return out
    pset.addPrimitive(ts_roc, [np.ndarray, int], np.ndarray, name="ROC")

    # ts_zscore: rolling z-score
    def ts_zscore(arr: np.ndarray, window: int) -> np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        cum = np.cumsum(arr, axis=1)
        cum2 = np.cumsum(arr * arr, axis=1)
        idx_hi = [slice(None)] * arr.ndim
        idx_lo = [slice(None)] * arr.ndim
        idx_hi[1] = slice(window, None)
        idx_lo[1] = slice(None, -window)
        sum_x = cum[tuple(idx_hi)] - cum[tuple(idx_lo)]
        sum_x2 = cum2[tuple(idx_hi)] - cum2[tuple(idx_lo)]
        mean = sum_x / window
        var = sum_x2 / window - mean ** 2
        std = np.sqrt(np.maximum(var, 0))
        arr_cur = arr[tuple(idx_hi)]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[tuple(idx_hi)] = np.where(std > 1e-8, (arr_cur - mean) / std, 0.0)
        return out
    pset.addPrimitive(ts_zscore, [np.ndarray, int], np.ndarray, name="ZSCORE")

    # ts_decay_linear: linearly decaying weighted moving average
    # 权重 = [1, 2, 3, ..., window] / sum(1..window)
    def ts_decay_linear(arr: np.ndarray, window: int) -> np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        weights = np.arange(1, window + 1, dtype=np.float64)
        weights /= weights.sum()
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=1)
        out_slice = [slice(None)] * arr.ndim
        out_slice[1] = slice(window - 1, None)
        out[tuple(out_slice)] = np.tensordot(win, weights, axes=([-1], [0]))
        return out
    pset.addPrimitive(ts_decay_linear, [np.ndarray, int], np.ndarray, name="DECAYLINEAR")

    #6. --------------- 三元算子 ---------------
    # ts_corr: rolling Pearson correlation between two arrays
    def ts_corr(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
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
            corr = np.where(np.abs(den) > 1e-8, num / den, 0.0)
        out_slice = [slice(None)] * a.ndim
        out_slice[1] = slice(window - 1, None)
        out[tuple(out_slice)] = corr
        return out
    pset.addPrimitive(ts_corr, [np.ndarray, np.ndarray, int], np.ndarray, name="CORR")

    # ts_cov: rolling covariance between two arrays
    def ts_cov(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
        n = a.shape[1]
        out = np.full(a.shape, np.nan, dtype=np.float64)
        if n < window or window <= 1:
            return out
        aw = np.lib.stride_tricks.sliding_window_view(a, window, axis=1)
        bw = np.lib.stride_tricks.sliding_window_view(b, window, axis=1)
        am = aw - aw.mean(axis=-1, keepdims=True)
        bm = bw - bw.mean(axis=-1, keepdims=True)
        cov = (am * bm).sum(axis=-1) / (window - 1)
        out_slice = [slice(None)] * a.ndim
        out_slice[1] = slice(window - 1, None)
        out[tuple(out_slice)] = cov
        return out
    pset.addPrimitive(ts_cov, [np.ndarray, np.ndarray, int], np.ndarray, name="COVIANCE")

    # ts_sma: EMA with alpha = m/n.  SMA(x, n, m) => sma[t] = m/n*x[t] + (1-m/n)*sma[t-1]
    def ts_sma(arr: np.ndarray, n: int, m: int) -> np.ndarray:
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
    pset.addPrimitive(ts_sma, [np.ndarray, int, int], np.ndarray, name="SMA")

    # if_positive: a > 0 ? b : c  (元素级条件)
    def if_positive(cond: np.ndarray, true_val: np.ndarray, false_val: np.ndarray) -> np.ndarray:
        return np.where(cond > 0, true_val, false_val)
    pset.addPrimitive(if_positive, [np.ndarray, np.ndarray, np.ndarray], np.ndarray, name="IF_POS")

    # ts_power: arr ** window (window 作为指数)
    def ts_power(arr: np.ndarray, exponent: int) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(np.abs(arr) < 1e6, np.power(arr, exponent), 0.0)
    pset.addPrimitive(ts_power, [np.ndarray, int], np.ndarray, name="POWER")

    #7. --------------- 一元通用算子 ---------------
    pset.addPrimitive(np.tanh, [np.ndarray], np.ndarray, name="TANH")
    pset.addPrimitive(np.negative, [np.ndarray], np.ndarray, name="NEG")

    #8. --------------- 高级时序算子 (补充GTJA191) ---------------
    # TSARGMAX: days since rolling max (returns 0~window-1)
    def ts_argmax(arr: np.ndarray, window: int) -> np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=1)
        argmax = window - 1 - win.argmax(axis=-1)
        out_slice = [slice(None)] * arr.ndim
        out_slice[1] = slice(window - 1, None)
        out[tuple(out_slice)] = argmax
        return out
    pset.addPrimitive(ts_argmax, [np.ndarray, int], np.ndarray, name="TSARGMAX")

    # TSARGMIN: days since rolling min
    def ts_argmin(arr: np.ndarray, window: int) -> np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=1)
        argmin = window - 1 - win.argmin(axis=-1)
        out_slice = [slice(None)] * arr.ndim
        out_slice[1] = slice(window - 1, None)
        out[tuple(out_slice)] = argmin
        return out
    pset.addPrimitive(ts_argmin, [np.ndarray, int], np.ndarray, name="TSARGMIN")

    # PROD: rolling product
    def ts_prod(arr: np.ndarray, window: int) -> np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        log_arr = np.log(np.maximum(np.abs(arr), 1e-12))
        s = np.cumsum(log_arr, axis=1)
        idx_hi = [slice(None)] * arr.ndim; idx_lo = [slice(None)] * arr.ndim
        idx_hi[1] = slice(window, None); idx_lo[1] = slice(None, -window)
        out[tuple(idx_hi)] = np.exp(s[tuple(idx_hi)] - s[tuple(idx_lo)])
        return out
    pset.addPrimitive(ts_prod, [np.ndarray, int], np.ndarray, name="PROD")

    # SUMAC: cumulative sum (no window argument)
    def ts_cumsum(arr: np.ndarray) -> np.ndarray:
        return np.cumsum(np.nan_to_num(arr, nan=0.0), axis=1)
    pset.addPrimitive(ts_cumsum, [np.ndarray], np.ndarray, name="SUMAC")

    # WMA: weighted MA with ascending weights [1,2,...,n]/sum(1..n)
    def ts_wma(arr: np.ndarray, window: int) -> np.ndarray:
        n = arr.shape[1]
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if n < window or window <= 0:
            return out
        w = np.arange(1, window + 1, dtype=np.float64)
        w /= w.sum()
        win = np.lib.stride_tricks.sliding_window_view(arr, window, axis=1)
        out_slice = [slice(None)] * arr.ndim
        out_slice[1] = slice(window - 1, None)
        out[tuple(out_slice)] = np.tensordot(win, w, axes=([-1], [0]))
        return out
    pset.addPrimitive(ts_wma, [np.ndarray, int], np.ndarray, name="WMA")

    # SIGNED_POWER: sign(x) * |x|^a
    def ts_signed_power(arr: np.ndarray, exponent: int) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.sign(arr) * (np.abs(arr) ** exponent)
    pset.addPrimitive(ts_signed_power, [np.ndarray, int], np.ndarray, name="SIGNED_POWER")

    #9. --------------- 滚动回归算子 ---------------
    # REGBETA: 滚动回归系数 β = Cov(A,B) / Var(B)
    def ts_regbeta(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
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
        out_slice = [slice(None)] * a.ndim
        out_slice[1] = slice(window - 1, None)
        out[tuple(out_slice)] = beta
        return out
    pset.addPrimitive(ts_regbeta, [np.ndarray, np.ndarray, int], np.ndarray, name="REGBETA")

    # REGRESI: 滚动回归残差 ε = A[t] - (α + β * B[t])
    def ts_regresi(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
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
        # α = mean(A) - β * mean(B)
        alpha = aw.mean(axis=-1) - beta * bw.mean(axis=-1)
        # 残差 = 当前A - (α + β * 当前B)
        resid = aw[..., -1] - (alpha + beta * bw[..., -1])
        out_slice = [slice(None)] * a.ndim
        out_slice[1] = slice(window - 1, None)
        out[tuple(out_slice)] = resid
        return out
    pset.addPrimitive(ts_regresi, [np.ndarray, np.ndarray, int], np.ndarray, name="REGRESI")

    #10. --------------- 序列生成算子 ---------------
    # SEQUENCE: 生成 1~n 的等差序列，返回 (1, n) 行向量，可通过 broadcasting 参与运算
    def ts_sequence(n: int) -> np.ndarray:
        if n <= 0:
            n = 1
        return np.arange(1, n + 1, dtype=np.float64).reshape(1, -1)
    pset.addPrimitive(ts_sequence, [int], np.ndarray, name="SEQUENCE")

    #11. --------------- 比较算子（条件→0/1掩码, 方案C）---------------
    def ts_gt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a > b → 1.0 else 0.0"""
        with np.errstate(invalid="ignore"):
            return np.where(a > b, 1.0, 0.0)
    pset.addPrimitive(ts_gt, [np.ndarray, np.ndarray], np.ndarray, name="GT")

    def ts_lt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a < b → 1.0 else 0.0"""
        with np.errstate(invalid="ignore"):
            return np.where(a < b, 1.0, 0.0)
    pset.addPrimitive(ts_lt, [np.ndarray, np.ndarray], np.ndarray, name="LT")

    def ts_ge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a >= b → 1.0 else 0.0"""
        with np.errstate(invalid="ignore"):
            return np.where(a >= b, 1.0, 0.0)
    pset.addPrimitive(ts_ge, [np.ndarray, np.ndarray], np.ndarray, name="GE")

    def ts_le(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a <= b → 1.0 else 0.0"""
        with np.errstate(invalid="ignore"):
            return np.where(a <= b, 1.0, 0.0)
    pset.addPrimitive(ts_le, [np.ndarray, np.ndarray], np.ndarray, name="LE")

    def ts_eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """|a-b| < 1e-8 → 1.0 else 0.0"""
        with np.errstate(invalid="ignore"):
            return np.where(np.abs(a - b) < 1e-8, 1.0, 0.0)
    pset.addPrimitive(ts_eq, [np.ndarray, np.ndarray], np.ndarray, name="EQ")

    def ts_neq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """|a-b| >= 1e-8 → 1.0 else 0.0"""
        with np.errstate(invalid="ignore"):
            return np.where(np.abs(a - b) >= 1e-8, 1.0, 0.0)
    pset.addPrimitive(ts_neq, [np.ndarray, np.ndarray], np.ndarray, name="NEQ")

    #12. --------------- 条件函数（掩码应用, 方案C）---------------
    def ts_pos(a: np.ndarray) -> np.ndarray:
        """保留正值，其余置0"""
        return np.where(a > 0, a, 0.0)
    pset.addPrimitive(ts_pos, [np.ndarray], np.ndarray, name="POS")

    def ts_neg(a: np.ndarray) -> np.ndarray:
        """保留负值，其余置0"""
        return np.where(a < 0, a, 0.0)
    pset.addPrimitive(ts_neg, [np.ndarray], np.ndarray, name="NEGVAL")

    def ts_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a 上穿 b (a>b) → 1.0 else 0.0"""
        with np.errstate(invalid="ignore"):
            return np.where(a > b, 1.0, 0.0)
    pset.addPrimitive(ts_cross, [np.ndarray, np.ndarray], np.ndarray, name="CROSS")

    #13. --------------- 热启动：技术指标 DTM/DBM/TR/HD/LD ---------------
    # DTM = (OPEN <= DELAY(OPEN,1) ? 0 : MAX(HIGH-OPEN, OPEN-DELAY(OPEN,1)))
    def ts_dtm(open_: np.ndarray, high: np.ndarray) -> np.ndarray:
        open_lag1 = ts_delay(open_, 1)
        diff1 = high - open_
        diff2 = open_ - open_lag1
        return np.where(open_ <= open_lag1, 0.0, np.maximum(diff1, diff2))
    pset.addPrimitive(ts_dtm, [np.ndarray, np.ndarray], np.ndarray, name="DTM")

    # DBM = (OPEN >= DELAY(OPEN,1) ? 0 : MAX(OPEN-LOW, OPEN-DELAY(OPEN,1)))
    def ts_dbm(open_: np.ndarray, low: np.ndarray) -> np.ndarray:
        open_lag1 = ts_delay(open_, 1)
        diff1 = open_ - low
        diff2 = open_ - open_lag1
        return np.where(open_ >= open_lag1, 0.0, np.maximum(diff1, diff2))
    pset.addPrimitive(ts_dbm, [np.ndarray, np.ndarray], np.ndarray, name="DBM")

    # TR = MAX(MAX(HIGH-LOW, ABS(HIGH-DELAY(CLOSE,1))), ABS(LOW-DELAY(CLOSE,1)))
    def ts_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        close_lag1 = ts_delay(close, 1)
        a = high - low
        b = np.abs(high - close_lag1)
        c = np.abs(low - close_lag1)
        return np.maximum(np.maximum(a, b), c)
    pset.addPrimitive(ts_tr, [np.ndarray, np.ndarray, np.ndarray], np.ndarray, name="TR")

    # HD = HIGH - DELAY(HIGH, 1)
    def ts_hd(high: np.ndarray) -> np.ndarray:
        return ts_delta(high, 1)
    pset.addPrimitive(ts_hd, [np.ndarray], np.ndarray, name="HD")

    # LD = DELAY(LOW, 1) - LOW
    def ts_ld(low: np.ndarray) -> np.ndarray:
        return -ts_delta(low, 1)
    pset.addPrimitive(ts_ld, [np.ndarray], np.ndarray, name="LD")

    #14. --------------- SELF: 表达式自身 t-1 值（递归自引用）---------------
    def ts_self(arr: np.ndarray) -> np.ndarray:
        """返回表达式在 t-1 时刻的值，首次评估返回 NaN"""
        if _self_cache is not None:
            return _self_cache
        return np.full(arr.shape, np.nan, dtype=np.float64)
    pset.addPrimitive(ts_self, [np.ndarray], np.ndarray, name="SELF")

    return pset


# ---- SELF 缓存接口 ----
_self_cache = None

def set_self_cache(value: np.ndarray):
    """在每轮评估前设置 SELF 的滞后值（通常为上一轮因子值的 DELAY(1)）"""
    global _self_cache
    _self_cache = value

def clear_self_cache():
    global _self_cache
    _self_cache = None


# ---- 安全生成函数：当类型没有 primitives 时自动回退到 terminals ----

import sys as _sys

def genGrowSafe(pset, min_, max_, type_=None):
    """genGrow 的安全版本：当某类型在 primitives 字典中无条目时，
    自动从 terminals 中选取，避免 IndexError。"""
    import random as _random

    def condition(height, depth):
        return depth == height or \
            (depth >= min_ and _random.random() < pset.terminalRatio)

    return _generate_safe(pset, min_, max_, condition, type_)


def genFullSafe(pset, min_, max_, type_=None):
    """genFull 的安全版本：当某类型在 primitives 字典中无条目时，
    自动从 terminals 中选取，避免 IndexError。"""

    def condition(height, depth):
        return depth == height

    return _generate_safe(pset, min_, max_, condition, type_)


def genHalfAndHalfSafe(pset, min_, max_, type_=None):
    """genHalfAndHalf 的安全版本。"""
    import random as _random
    method = _random.choice((genGrowSafe, genFullSafe))
    return method(pset, min_, max_, type_)


def _generate_safe(pset, min_, max_, condition, type_=None):
    """generate() 的安全变体：当 primitives[type_] 为空时回退到 terminals。

    这是对 deap.gp.generate 的微小修改，仅添加了一个条件：
        condition(...) or not pset.primitives[type_]
    确保纯叶子类型（如 int）在非终端条件触发时也能正常生成。
    """
    import random as _random
    from deap.gp import MetaEphemeral

    if type_ is None:
        type_ = pset.ret
    expr = []
    height = _random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        # 关键修改：primitives[type_] 为空 → 强制走 terminal
        if condition(height, depth) or not pset.primitives[type_]:
            try:
                term = _random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = _sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add "
                    "a terminal of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            if type(term) is MetaEphemeral:
                term = term()
            expr.append(term)
        else:
            try:
                prim = _random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = _sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add "
                    "a primitive of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr
