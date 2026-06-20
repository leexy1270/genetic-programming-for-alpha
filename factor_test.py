"""
factor_test.py — 多因子测试框架 (Multi-Factor Testing Framework)

标准化量化多因子评估流程，适用于GP挖掘的Alpha因子，产出结果可用于研究报告/论文。

Pipeline:
    1. 因子计算    — 从GP表达式树 → 面板数据 (股票×日期)
    2. 数据预处理  — 去极值 → 缺失填充 → 标准化 → 中性化(可选)
    3. 单因子评估  — IC分析 + 分层回测 + 多空组合 + 因子分布
    4. 因子筛选    — ICIR阈值 + 相关性去重 + 换手率控制
    5. 多因子合成  — 等权 / ICIR加权 / 滚动回归
    6. 可视化输出  — 论文级图表

Usage:
    from factor_test import FactorTester
    ft = FactorTester(data_3d, feature_cols, dates, stock_codes, pset)
    ft.load_expressions(gtja191.FACTOR_EXPRS)       # 加载因子表达式
    ft.load_expressions(wq101.WQ101_EXPRS, prefix='wq')
    ft.run_pipeline()                                # 运行完整流程

Author: GP Alpha Research
Date: 2026-06-17
"""

import numpy as np
import pandas as pd
from deap import gp
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 配置参数
# ============================================================

@dataclass
class FactorTestConfig:
    """多因子测试配置"""
    # 预处理
    winsorize_std: float = 5.0        # 去极值：MAD倍数
    fill_method: str = 'cross_median' # 缺失填充: 'cross_median' | 'zero' | 'ffill'
    standardize: bool = True          # 截面标准化

    # IC分析
    forward_period: int = 1           # 未来N日收益
    ic_type: str = 'rank'             # 'rank' | 'pearson'

    # 分层回测
    n_groups: int = 5                 # 分组数
    group_weight: str = 'equal'       # 'equal' | 'value_weighted'

    # 因子筛选
    icir_min: float = 0.3             # ICIR最低阈值
    max_correlation: float = 0.7      # 因子间最大相关性
    min_coverage: float = 0.5         # 最小覆盖度

    # 多因子合成
    composite_method: str = 'icir_weighted'  # 'equal' | 'icir_weighted' | 'rolling_regression'
    composite_lookback: int = 60      # 滚动回归回看窗口


# ============================================================
# 核心类
# ============================================================

class FactorTester:
    """
    多因子测试器

    Parameters
    ----------
    data_3d : np.ndarray, shape (n_stocks, n_dates, n_features)
    feature_cols : list of str
    dates : pd.DatetimeIndex
    stock_codes : list of str
    pset : gp.PrimitiveSetTyped
    config : FactorTestConfig
    """

    def __init__(
        self,
        data_3d: np.ndarray,
        feature_cols: List[str],
        dates: pd.DatetimeIndex,
        stock_codes: List[str],
        pset: gp.PrimitiveSetTyped,
        config: Optional[FactorTestConfig] = None,
    ):
        self.data = data_3d
        self.feature_cols = list(feature_cols)
        self.dates = pd.DatetimeIndex(dates)
        self.stock_codes = list(stock_codes)
        self.pset = pset
        self.config = config or FactorTestConfig()

        self.n_stocks, self.n_dates, self.n_features = data_3d.shape

        # 预计算 forward returns
        self._prepare_forward_returns()

        # 存储
        self.expressions: Dict[str, str] = {}       # {name: expr_str}
        self.factors: Dict[str, np.ndarray] = {}    # {name: factor_values}
        self.factor_ics: Dict[str, np.ndarray] = {} # {name: IC_series}
        self.factor_stats: Dict[str, dict] = {}     # {name: stats_dict}
        self.selected_factors: List[str] = []        # 筛选后的因子名
        self.composite_factor: Optional[np.ndarray] = None
        self.composite_stats: dict = {}

    # ---- 1. 数据准备 ----

    def _prepare_forward_returns(self):
        """预计算未来收益"""
        if 'RETURN' in self.feature_cols:
            ret_idx = self.feature_cols.index('RETURN')
            ret_panel = self.data[:, :, ret_idx]  # (stocks, dates)
        else:
            # 用CLOSE计算
            close_idx = self.feature_cols.index('CLOSE')
            close = self.data[:, :, close_idx]
            ret_panel = np.full_like(close, np.nan)
            ret_panel[:, 1:] = close[:, 1:] / close[:, :-1] - 1

        # forward return: 未来N日
        fwd = self.config.forward_period
        self.forward_ret = np.full_like(ret_panel, np.nan)
        if fwd == 1:
            self.forward_ret[:, :-1] = ret_panel[:, 1:]
        else:
            self.forward_ret[:, :-fwd] = ret_panel[:, fwd:]

    def _get_feature_index(self, name: str) -> int:
        """安全获取特征列索引"""
        if name in self.feature_cols:
            return self.feature_cols.index(name)
        # 尝试映射
        mapping = {'OPEN': 'OPEN', 'HIGH': 'HIGH', 'LOW': 'LOW', 'CLOSE': 'CLOSE',
                   'RETURN': 'RETURN', 'VOLUME': 'VOLUME', 'VWAP': 'VWAP'}
        for map_to in mapping.values():
            if map_to in self.feature_cols:
                return self.feature_cols.index(map_to)
        return 0

    # ---- 2. 因子计算 ----

    def load_expressions(self, expr_dict: Dict[str, str], prefix: str = ''):
        """加载因子表达式字典
        Args:
            expr_dict: {name: expr_str} 或 {name: None}
            prefix: 为因子名前缀（如 'wq' → 'wq001'）
        """
        from gtja191 import parse_expr
        for name, expr_str in expr_dict.items():
            if expr_str is None:
                continue
            store_name = f"{prefix}{name}" if prefix else name
            # 去除可能已有的前缀避免重复
            if prefix and name.startswith(prefix):
                store_name = name
            self.expressions[store_name] = expr_str

    def compute_factor(self, name: str, expr_str: str) -> Optional[np.ndarray]:
        """计算单个因子值 (2D: stocks × dates)"""
        from gtja191 import parse_expr
        try:
            tree = parse_expr(expr_str, self.pset)
            func = gp.compile(tree, self.pset)
            args = [self.data[:, :, j] for j in range(self.n_features)]
            fv = func(*args)
            fv = np.atleast_1d(np.squeeze(fv))
            if fv.ndim == 2 and fv.shape == (self.n_stocks, self.n_dates):
                return fv.astype(np.float64)
            elif fv.ndim == 2 and fv.shape[1] == self.n_dates:
                return fv[:self.n_stocks, :].astype(np.float64)
        except Exception as e:
            pass
        return None

    def compute_all_factors(self, verbose: bool = True) -> Dict[str, np.ndarray]:
        """计算所有已加载因子"""
        from tqdm import tqdm
        items = list(self.expressions.items())
        iterator = tqdm(items, desc="Computing factors", unit='factor') if verbose else items

        for name, expr_str in iterator:
            fv = self.compute_factor(name, expr_str)
            if fv is not None:
                self.factors[name] = fv

        if verbose:
            print(f"Computed {len(self.factors)}/{len(self.expressions)} factors")
        return self.factors

    # ---- 3. 数据预处理 ----

    def preprocess_factor(self, factor: np.ndarray) -> np.ndarray:
        """
        标准预处理流水线：
        1. MAD去极值
        2. 缺失值填充
        3. 截面标准化
        """
        cfg = self.config
        cleaned = factor.copy()

        # 3.1 MAD去极值（逐日截面）
        for t in range(cleaned.shape[1]):
            cross = cleaned[:, t]
            mask = ~np.isnan(cross)
            if mask.sum() < 3:
                continue
            median = np.median(cross[mask])
            mad = np.median(np.abs(cross[mask] - median)) * 1.4826
            if mad < 1e-12:
                continue
            upper = median + cfg.winsorize_std * mad
            lower = median - cfg.winsorize_std * mad
            cleaned[mask, t] = np.clip(cross[mask], lower, upper)

        # 3.2 缺失值填充
        if cfg.fill_method == 'cross_median':
            for t in range(cleaned.shape[1]):
                col = cleaned[:, t]
                mask = np.isnan(col)
                if mask.any() and (~mask).sum() > 0:
                    col[mask] = np.nanmedian(col[~mask])
        elif cfg.fill_method == 'zero':
            cleaned = np.nan_to_num(cleaned, nan=0.0)
        # 'ffill' handled per-stock below

        # 3.3 截面标准化 (Z-Score per day)
        if cfg.standardize:
            for t in range(cleaned.shape[1]):
                cross = cleaned[:, t]
                mask = ~np.isnan(cross)
                if mask.sum() < 3:
                    continue
                mu, sigma = cross[mask].mean(), cross[mask].std(ddof=1)
                if sigma < 1e-12:
                    cleaned[mask, t] = 0.0
                else:
                    cleaned[mask, t] = (cross[mask] - mu) / sigma

        return cleaned

    def preprocess_all(self):
        """对所有因子做预处理"""
        for name in list(self.factors.keys()):
            self.factors[name] = self.preprocess_factor(self.factors[name])

    # ---- 4. IC 分析 ----

    def compute_ic_series(self, factor: np.ndarray) -> np.ndarray:
        """计算逐日截面IC序列"""
        ic_list = []
        for t in range(self.n_dates - self.config.forward_period):
            f_cross = factor[:, t]
            r_cross = self.forward_ret[:, t]
            mask = ~(np.isnan(f_cross) | np.isnan(r_cross))
            if mask.sum() < 10:
                ic_list.append(np.nan)
                continue

            if self.config.ic_type == 'rank':
                # Spearman Rank IC
                from scipy.stats import spearmanr
                ic, _ = spearmanr(f_cross[mask], r_cross[mask])
            else:
                # Pearson IC
                ic = np.corrcoef(f_cross[mask], r_cross[mask])[0, 1]
            ic_list.append(ic)

        return np.array(ic_list)

    def compute_ic_stats(self, ic_series: np.ndarray) -> dict:
        """计算IC统计量"""
        valid = ic_series[~np.isnan(ic_series)]
        if len(valid) < 10:
            return {'ic_mean': np.nan, 'ic_std': np.nan, 'icir': np.nan,
                    'ic_positive_ratio': np.nan, 'n_obs': len(valid)}

        return {
            'ic_mean': np.mean(valid),
            'ic_std': np.std(valid, ddof=1),
            'icir': np.mean(valid) / np.std(valid, ddof=1) if np.std(valid, ddof=1) > 1e-12 else 0.0,
            'ic_positive_ratio': np.mean(valid > 0),
            'n_obs': len(valid),
            't_stat': np.mean(valid) / (np.std(valid, ddof=1) / np.sqrt(len(valid))),
        }

    def compute_all_ics(self):
        """对所有因子计算IC"""
        from tqdm import tqdm
        for name, factor in tqdm(self.factors.items(), desc="Computing ICs"):
            ic_series = self.compute_ic_series(factor)
            self.factor_ics[name] = ic_series
            self.factor_stats[name] = self.compute_ic_stats(ic_series)

    # ---- 5. 分层回测 ----

    def stratified_backtest(self, factor: np.ndarray) -> dict:
        """
        分层回测
        Returns:
            dict with keys: group_returns, longshort_returns, group_stats
        """
        n_groups = self.config.n_groups
        group_returns = np.full((n_groups, self.n_dates), np.nan)
        longshort_returns = np.full(self.n_dates, np.nan)

        for t in range(self.n_dates - self.config.forward_period):
            f_cross = factor[:, t]
            fwd_ret = self.forward_ret[:, t]
            mask = ~(np.isnan(f_cross) | np.isnan(fwd_ret))
            if mask.sum() < n_groups * 3:
                continue

            # 分组
            valid_f = f_cross[mask]
            valid_r = fwd_ret[mask]
            quantiles = np.linspace(0, 1, n_groups + 1)
            bins = np.quantile(valid_f, quantiles)
            bins[0] = -np.inf
            bins[-1] = np.inf

            for g in range(n_groups):
                in_group = (valid_f >= bins[g]) & (valid_f < bins[g + 1]) if g < n_groups - 1 else \
                           (valid_f >= bins[g]) & (valid_f <= bins[g + 1])
                if in_group.sum() > 0:
                    group_returns[g, t] = valid_r[in_group].mean()

            # 多空收益 = Q5 - Q1
            if not np.isnan(group_returns[0, t]) and not np.isnan(group_returns[-1, t]):
                longshort_returns[t] = group_returns[-1, t] - group_returns[0, t]

        # 累计收益
        group_cumret = np.cumprod(1 + np.nan_to_num(group_returns, nan=0.0), axis=1) - 1
        ls_cumret = np.cumprod(1 + np.nan_to_num(longshort_returns, nan=0.0)) - 1

        # 多空统计
        ls_valid = longshort_returns[~np.isnan(longshort_returns)]
        annual_ret = np.mean(ls_valid) * 252 if len(ls_valid) > 0 else np.nan
        annual_vol = np.std(ls_valid, ddof=1) * np.sqrt(252) if len(ls_valid) > 0 else np.nan
        sharpe = annual_ret / annual_vol if annual_vol and annual_vol > 1e-12 else np.nan

        return {
            'group_returns': group_returns,
            'group_cumret': group_cumret,
            'longshort_returns': longshort_returns,
            'longshort_cumret': ls_cumret,
            'annual_return': annual_ret,
            'annual_vol': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': self._max_drawdown(ls_cumret),
        }

    def _max_drawdown(self, cumret: np.ndarray) -> float:
        """计算最大回撤"""
        peak = np.maximum.accumulate(cumret)
        drawdown = (cumret - peak) / (1 + peak)
        return np.min(drawdown)

    def run_stratified_all(self) -> Dict[str, dict]:
        """对所有因子做分层回测"""
        results = {}
        for name, factor in self.factors.items():
            results[name] = self.stratified_backtest(factor)
        return results

    # ---- 6. 因子筛选 ----

    def factor_correlation_matrix(self) -> pd.DataFrame:
        """计算因子间截面相关性矩阵"""
        names = list(self.factors.keys())
        n = len(names)
        corr_matrix = np.eye(n)

        # 对每个时间截面计算因子相关性，然后取均值
        corr_sum = np.zeros((n, n))
        corr_count = 0
        for t in range(self.n_dates):
            # 收集所有因子在t时刻的值
            fvals = []
            for name in names:
                fvals.append(self.factors[name][:, t])
            fvals = np.array(fvals)  # (n_factors, n_stocks)
            mask = ~np.any(np.isnan(fvals), axis=0)
            if mask.sum() < 10:
                continue
            corr_sum += np.corrcoef(fvals[:, mask])
            corr_count += 1
        if corr_count > 0:
            corr_matrix = corr_sum / corr_count
        return pd.DataFrame(corr_matrix, index=names, columns=names)

    def select_factors(self) -> List[str]:
        """
        因子筛选:
        1. ICIR >= icir_min
        2. 覆盖度 >= min_coverage
        3. 相关性去重 (保留ICIR高的)
        """
        cfg = self.config
        candidates = []

        for name, stats in self.factor_stats.items():
            if np.isnan(stats['icir']):
                continue
            # ICIR过滤
            if abs(stats['icir']) < cfg.icir_min:
                continue
            # 覆盖度过滤
            coverage = np.mean(~np.isnan(self.factors[name]))
            if coverage < cfg.min_coverage:
                continue
            candidates.append((name, stats['icir']))

        # 按ICIR绝对值降序
        candidates.sort(key=lambda x: abs(x[1]), reverse=True)

        # 相关性去重
        corr_df = self.factor_correlation_matrix()
        selected = []
        selected_icir = []

        for name, icir in candidates:
            # 检查与已选因子的相关性
            too_correlated = False
            for i, sel_name in enumerate(selected):
                if abs(corr_df.loc[name, sel_name]) > cfg.max_correlation:
                    too_correlated = True
                    break
            if not too_correlated:
                selected.append(name)
                selected_icir.append(icir)

        self.selected_factors = selected
        return selected

    # ---- 7. 多因子合成 ----

    def combine_factors(self) -> Tuple[np.ndarray, dict]:
        """多因子合成"""
        if not self.selected_factors:
            raise ValueError("No factors selected. Run select_factors() first.")

        cfg = self.config
        selected_factors = {n: self.factors[n] for n in self.selected_factors}

        if cfg.composite_method == 'equal':
            weights = {n: 1.0 / len(selected_factors) for n in selected_factors}
            composite = np.zeros_like(list(selected_factors.values())[0])
            for name, factor in selected_factors.items():
                composite += weights[name] * factor

        elif cfg.composite_method == 'icir_weighted':
            weights = {}
            total_w = 0.0
            for name in selected_factors:
                icir = max(0, self.factor_stats[name]['icir'])
                weights[name] = icir
                total_w += icir
            if total_w < 1e-12:
                weights = {n: 1.0 / len(selected_factors) for n in selected_factors}
            else:
                weights = {n: w / total_w for n, w in weights.items()}

            composite = np.zeros_like(list(selected_factors.values())[0])
            for name, factor in selected_factors.items():
                composite += weights[name] * factor

        elif cfg.composite_method == 'rolling_regression':
            composite = self._rolling_regression_combine(selected_factors)
            weights = {'method': 'rolling_regression', 'lookback': cfg.composite_lookback}

        else:
            raise ValueError(f"Unknown composite method: {cfg.composite_method}")

        # 对合成因子做预处理
        composite = self.preprocess_factor(composite)

        # 评估合成因子
        ic_series = self.compute_ic_series(composite)
        bt_results = self.stratified_backtest(composite)
        ic_stats = self.compute_ic_stats(ic_series)

        self.composite_factor = composite
        self.composite_stats = {
            'ic_series': ic_series,
            'ic_stats': ic_stats,
            'backtest': bt_results,
            'weights': weights,
        }

        return composite, self.composite_stats

    def _rolling_regression_combine(self, factors_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """滚动回归合成"""
        lookback = self.config.composite_lookback
        names = list(factors_dict.keys())
        result = np.full((self.n_stocks, self.n_dates), np.nan)

        for t in range(lookback, self.n_dates - self.config.forward_period):
            # 训练期: t - lookback 到 t-1
            train_start = max(0, t - lookback)
            train_end = t

            # 收集训练数据
            X_list, y_list = [], []
            for tt in range(train_start, train_end):
                f_cross = np.array([factors_dict[n][:, tt] for n in names])  # (n_factors, n_stocks)
                r_cross = self.forward_ret[:, tt]
                mask = ~np.any(np.isnan(f_cross), axis=0) & ~np.isnan(r_cross)
                if mask.sum() < 20:
                    continue
                X_list.append(f_cross[:, mask].T)  # (n_valid, n_factors)
                y_list.append(r_cross[mask])

            if len(X_list) < 10:
                continue

            X_train = np.vstack(X_list)
            y_train = np.hstack(y_list)

            try:
                beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            # 预测期: t
            f_now = np.array([factors_dict[n][:, t] for n in names])
            result[:, t] = f_now.T @ beta

        return result

    # ---- 8. 完整流程 ----

    def run_pipeline(self, verbose: bool = True) -> dict:
        """运行完整多因子测试流水线"""
        if verbose:
            print("=" * 60)
            print("Multi-Factor Testing Pipeline")
            print("=" * 60)

        # Step 1: 计算因子
        if verbose:
            print("\n[1/5] Computing factors...")
        self.compute_all_factors(verbose=verbose)

        # Step 2: 预处理
        if verbose:
            print(f"\n[2/5] Preprocessing {len(self.factors)} factors...")
        self.preprocess_all()

        # Step 3: IC分析 + 分层回测
        if verbose:
            print(f"\n[3/5] Computing ICs and backtests...")
        self.compute_all_ics()

        # Step 4: 因子筛选
        if verbose:
            print(f"\n[4/5] Selecting factors...")
        selected = self.select_factors()
        if verbose:
            print(f"  Selected {len(selected)}/{len(self.factors)} factors")

        # Step 5: 多因子合成
        if verbose:
            print(f"\n[5/5] Combining factors ({self.config.composite_method})...")
        if len(selected) >= 2:
            self.combine_factors()
        elif len(selected) == 1:
            if verbose:
                print("  Only 1 factor selected, using directly as composite")
            self.selected_factors = selected
            name = selected[0]
            self.composite_factor = self.factors[name]
        else:
            if verbose:
                print("  WARNING: No factors passed selection criteria")

        if verbose:
            self.print_summary()

        return self._build_report()

    def print_summary(self):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("FACTOR PERFORMANCE SUMMARY")
        print("=" * 60)

        # 单因子排名 (Top 10)
        sorted_factors = sorted(self.factor_stats.items(),
                                key=lambda x: abs(x[1].get('icir', 0)),
                                reverse=True)

        print(f"\n{'Factor':<20} {'ICIR':>8} {'IC_Mean':>8} {'IC_Std':>8} {'IC>0%':>7}")
        print("-" * 55)
        for name, stats in sorted_factors[:10]:
            print(f"{name:<20} {stats['icir']:>8.3f} {stats['ic_mean']:>8.4f} "
                  f"{stats['ic_std']:>8.4f} {stats['ic_positive_ratio']:>7.1%}")

        # 多因子合成结果
        if self.composite_stats:
            cs = self.composite_stats['ic_stats']
            print(f"\n{'Composite':<20} {cs['icir']:>8.3f} {cs['ic_mean']:>8.4f} "
                  f"{cs['ic_std']:>8.4f} {cs['ic_positive_ratio']:>7.1%}")

            bt = self.composite_stats['backtest']
            print(f"\nLong-Short Performance:")
            print(f"  Annual Return: {bt['annual_return']:.2%}")
            print(f"  Sharpe Ratio:  {bt['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:  {bt['max_drawdown']:.2%}")

        if self.selected_factors:
            print(f"\nSelected Factors ({len(self.selected_factors)}): "
                  f"{', '.join(self.selected_factors[:8])}...")

    def _build_report(self) -> dict:
        """构建完整报告"""
        return {
            'factors': self.factors,
            'factor_ics': self.factor_ics,
            'factor_stats': self.factor_stats,
            'selected_factors': self.selected_factors,
            'composite_factor': self.composite_factor,
            'composite_stats': self.composite_stats,
            'config': self.config,
        }


# ============================================================
# 9. 可视化函数 (论文级别)
# ============================================================

class FactorVisualizer:
    """因子可视化工具 — 输出论文级图表"""

    def __init__(self, tester: FactorTester):
        self.tester = tester
        self._setup_style()

    def _setup_style(self):
        """论文级 matplotlib 样式"""
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'figure.facecolor': 'white',
        })

    def plot_ic_heatmap(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        因子IC热力图
        行=因子 (按ICIR排序), 列=月份, 颜色=月度IC均值
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # 取 ICIR 最高的前 top_n 个因子
        sorted_factors = sorted(self.tester.factor_stats.items(),
                                key=lambda x: abs(x[1].get('icir', 0)), reverse=True)
        top_factors = sorted_factors[:top_n]

        # 按月聚合IC
        dates = self.tester.dates[:len(next(iter(self.tester.factor_ics.values())))]
        monthly_ic = {}
        for name, _ in top_factors:
            ic_series = pd.Series(self.tester.factor_ics[name], index=dates[:len(self.tester.factor_ics[name])])
            monthly = ic_series.resample('ME').mean()
            monthly_ic[name] = monthly

        # 构造矩阵
        all_months = sorted(set().union(*[set(m.index) for m in monthly_ic.values()]))
        ic_matrix = np.full((len(top_factors), len(all_months)), np.nan)
        for i, (name, _) in enumerate(top_factors):
            for j, month in enumerate(all_months):
                if month in monthly_ic[name].index:
                    ic_matrix[i, j] = monthly_ic[name][month]

        fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.35)))
        im = ax.imshow(ic_matrix, aspect='auto', cmap='RdYlBu_r',
                       vmin=-0.15, vmax=0.15, interpolation='nearest')

        ax.set_yticks(range(len(top_factors)))
        ax.set_yticklabels([n for n, _ in top_factors], fontsize=8)
        ax.set_xticks(range(0, len(all_months), max(1, len(all_months) // 12)))
        ax.set_xticklabels([all_months[i].strftime('%Y-%m')
                           for i in range(0, len(all_months), max(1, len(all_months) // 12))],
                           rotation=45, ha='right', fontsize=8)
        ax.set_title('Factor IC Heatmap (Monthly Mean Rank IC)', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Mean IC', shrink=0.8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_stratified_returns(self, factor_name: Optional[str] = None, save_path: Optional[str] = None):
        """
        分层累计收益曲线
        factor_name=None → 使用合成因子
        """
        import matplotlib.pyplot as plt

        if factor_name is None:
            if self.tester.composite_factor is not None:
                bt = self.tester.stratified_backtest(self.tester.composite_factor)
                title = 'Composite Factor — Stratified Cumulative Returns'
            else:
                raise ValueError("No composite factor available")
        else:
            bt = self.tester.stratified_backtest(self.tester.factors[factor_name])
            title = f'{factor_name} — Stratified Cumulative Returns'

        fig, ax = plt.subplots(figsize=(12, 6))
        n_groups = bt['group_cumret'].shape[0]
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_groups))

        for g in range(n_groups):
            label = f'Q{g+1} ({"Bottom" if g==0 else "Top" if g==n_groups-1 else f"Q{g+1}"})'
            ax.plot(self.tester.dates, bt['group_cumret'][g], color=colors[g],
                    linewidth=1.5 if g in [0, n_groups-1] else 0.8, label=label)

        # 多空曲线 (右轴)
        ax2 = ax.twinx()
        ax2.plot(self.tester.dates, bt['longshort_cumret'], 'k--', linewidth=1.2, label='Long-Short (Q5-Q1)')
        ax2.set_ylabel('Long-Short Cumulative Return')
        ax2.legend(loc='upper right', fontsize=8)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend(loc='upper left', fontsize=7)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_ic_decay(self, factor_name: str, max_lag: int = 20, save_path: Optional[str] = None):
        """
        IC衰减曲线
        计算因子对未来1~max_lag期的IC均值 ± std
        """
        import matplotlib.pyplot as plt

        factor = self.tester.factors[factor_name]
        ic_means, ic_stds = [], []

        for lag in range(1, max_lag + 1):
            ic_list = []
            for t in range(self.tester.n_dates - lag):
                f_cross = factor[:, t]
                r_cross = self.tester.data[:, t + lag, self.tester._get_feature_index('CLOSE')] / \
                          self.tester.data[:, t, self.tester._get_feature_index('CLOSE')] - 1 if \
                          t + lag < self.tester.n_dates else np.full(self.tester.n_stocks, np.nan)
                mask = ~(np.isnan(f_cross) | np.isnan(r_cross))
                if mask.sum() < 10:
                    continue
                from scipy.stats import spearmanr
                ic, _ = spearmanr(f_cross[mask], r_cross[mask])
                ic_list.append(ic)
            valid = [x for x in ic_list if not np.isnan(x)]
            if valid:
                ic_means.append(np.mean(valid))
                ic_stds.append(np.std(valid, ddof=1))
            else:
                ic_means.append(np.nan)
                ic_stds.append(np.nan)

        fig, ax = plt.subplots(figsize=(10, 5))
        lags = range(1, max_lag + 1)
        ax.plot(lags, ic_means, 'b-o', markersize=5, label='Mean IC')
        ax.fill_between(lags, np.array(ic_means) - np.array(ic_stds),
                        np.array(ic_means) + np.array(ic_stds),
                        alpha=0.2, color='b', label='±1 Std')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_title(f'{factor_name} — IC Decay Curve', fontweight='bold')
        ax.set_xlabel('Forward Period (days)')
        ax.set_ylabel('Rank IC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_correlation_matrix(self, save_path: Optional[str] = None):
        """因子相关性矩阵 (仅显示已选因子)"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        corr_df = self.tester.factor_correlation_matrix()
        if self.tester.selected_factors:
            corr_df = corr_df.loc[self.tester.selected_factors, self.tester.selected_factors]

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
        sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                    vmin=-1, vmax=1, center=0, square=True,
                    linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Factor Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_longshort_drawdown(self, factor_name: Optional[str] = None, save_path: Optional[str] = None):
        """多空净值与回撤"""
        import matplotlib.pyplot as plt

        if factor_name is None:
            if self.tester.composite_factor is not None:
                bt = self.tester.stratified_backtest(self.tester.composite_factor)
                title = 'Composite Factor — Long-Short Performance'
            else:
                raise ValueError("No composite factor available")
        else:
            bt = self.tester.stratified_backtest(self.tester.factors[factor_name])
            title = f'{factor_name} — Long-Short Performance'

        cumret = bt['longshort_cumret']
        valid_mask = ~np.isnan(cumret)
        cumret = cumret[valid_mask]
        valid_dates = self.tester.dates[valid_mask]

        # 回撤
        peak = np.maximum.accumulate(cumret)
        drawdown = (cumret - peak) / (1 + peak)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                        gridspec_kw={'height_ratios': [2, 1]})

        ax1.plot(valid_dates, cumret, 'b-', linewidth=1.5, label='Long-Short Cumulative Return')
        ax1.fill_between(valid_dates, 0, cumret, alpha=0.1, color='b')
        ax1.axhline(y=0, color='gray', linewidth=0.5)
        ax1.set_title(title, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(valid_dates, 0, drawdown, color='red', alpha=0.3, label='Drawdown')
        ax2.plot(valid_dates, drawdown, 'r-', linewidth=0.8)
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_composite_comparison(self, save_path: Optional[str] = None):
        """
        合成因子 vs 单因子IR对比
        散点图: X=ICIR绝对值, Y=多空夏普
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        for name in self.tester.factor_stats:
            stats = self.tester.factor_stats[name]
            bt = self.tester.stratified_backtest(self.tester.factors[name])
            ax.scatter(abs(stats['icir']), bt['sharpe_ratio'],
                      s=50, alpha=0.5, c='steelblue', edgecolors='none')
            if name in self.tester.selected_factors:
                ax.scatter(abs(stats['icir']), bt['sharpe_ratio'],
                          s=120, alpha=0.9, c='darkorange', edgecolors='black', linewidth=0.5,
                          marker='D', label='_nolegend_')

        # 合成因子
        if self.tester.composite_stats:
            cs_ic = self.tester.composite_stats['ic_stats']
            cs_bt = self.tester.composite_stats['backtest']
            ax.scatter(abs(cs_ic['icir']), cs_bt['sharpe_ratio'],
                      s=200, alpha=1.0, c='red', edgecolors='black', linewidth=1.5,
                      marker='*', label='Composite', zorder=10)

        ax.set_xlabel('|ICIR|')
        ax.set_ylabel('Long-Short Sharpe Ratio')
        ax.set_title('Factor Performance: Single Factors vs Composite', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


# ============================================================
# 10. 便捷入口
# ============================================================

def run_full_analysis(
    data_3d: np.ndarray,
    feature_cols: List[str],
    dates: pd.DatetimeIndex,
    stock_codes: List[str],
    pset: gp.PrimitiveSetTyped,
    gtja_exprs: dict,
    wq101_exprs: dict,
    config: Optional[FactorTestConfig] = None,
    output_dir: str = 'results',
) -> Tuple[FactorTester, FactorVisualizer]:
    """
    一键运行完整分析流程并输出图表

    Parameters
    ----------
    data_3d, feature_cols, dates, stock_codes, pset : 数据和原语集
    gtja_exprs : GTJA 191 因子表达式字典
    wq101_exprs : WQ101 因子表达式字典
    config : 测试配置
    output_dir : 输出目录

    Returns
    -------
    tester, visualizer
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 初始化
    tester = FactorTester(data_3d, feature_cols, dates, stock_codes, pset, config)

    # 加载因子
    tester.load_expressions(gtja_exprs, prefix='gtja_')
    tester.load_expressions(wq101_exprs, prefix='wq_')

    # 运行流程
    tester.run_pipeline()

    # 可视化
    viz = FactorVisualizer(tester)

    if tester.factors:
        viz.plot_ic_heatmap(top_n=20, save_path=f'{output_dir}/ic_heatmap.png')

    if tester.composite_factor is not None:
        viz.plot_stratified_returns(save_path=f'{output_dir}/stratified_returns.png')
        viz.plot_longshort_drawdown(save_path=f'{output_dir}/longshort_drawdown.png')

    if tester.selected_factors:
        viz.plot_correlation_matrix(save_path=f'{output_dir}/correlation_matrix.png')
        viz.plot_composite_comparison(save_path=f'{output_dir}/composite_comparison.png')

    # 导出CSV
    report_df = pd.DataFrame([
        {'factor': name, **stats}
        for name, stats in tester.factor_stats.items()
    ])
    report_df = report_df.sort_values('icir', ascending=False, key=abs)
    report_df.to_csv(f'{output_dir}/factor_report.csv', index=False)
    print(f"\nResults saved to {output_dir}/")

    return tester, viz


# ============================================================
# 11. 测试入口
# ============================================================

if __name__ == '__main__':
    print("Factor Testing Framework")
    print("Usage:")
    print("  from factor_test import FactorTester, FactorVisualizer, run_full_analysis")
    print("  tester = FactorTester(data_3d, feature_cols, dates, stock_codes, pset)")
    print("  tester.load_expressions(gtja191.FACTOR_EXPRS)")
    print("  tester.load_expressions(wq101.WQ101_EXPRS)")
    print("  tester.run_pipeline()")
