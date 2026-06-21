import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from deap import gp
import matplotlib.pyplot as plt
from parameters import *

def get_data_tdx(STOCK_LIST):
    "通过本地通达信数据获取股票数据"
    from mootdx.reader import Reader
    reader = Reader.factory(market='std', tdxdir='C:/new_tdx64')
    failed_list=[]  #失败获取数据
    stock_dict={}   #成功获取股票数据
    pbar = tqdm(STOCK_LIST, desc="读取股票日线", unit='只')
    for code in pbar:
        try:
            data = reader.daily(symbol=code)
            del data['amount']
            data['code'] = code
            data['date'] = pd.to_datetime(data.index)

            stock_dict[code] = data
        except Exception:
            failed_list.append(code)
        pbar.set_postfix({'失败': len(failed_list)})
    if failed_list:
        print(f'失败获取的股票有：{failed_list}')

    return stock_dict

def get_data_tushare(STOCK_LIST,API,start_date,end_date):
    "通过tushare获取股票日度数据，带频率控制和失败重试"
    import tushare as ts
    import time
    import os
    import pickle

    CACHE_FILE = "data/tushare_cache.pkl"

    # 断点续传：加载已有缓存
    stock_dict = {}
    failed_list = []
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                stock_dict = cache.get('data', {})
                failed_list = cache.get('failed', [])
                print(f"[缓存] 已加载 {len(stock_dict)} 只成功 + {len(failed_list)} 只失败")
        except Exception:
            pass

    # 过滤已获取的股票
    remaining = [c for c in STOCK_LIST if c not in stock_dict and c not in failed_list]

    pro = ts.pro_api(API)

    def save_cache():
        os.makedirs("data", exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'data': stock_dict, 'failed': failed_list}, f)

    pbar = tqdm(remaining, desc="通过 Tushare 获取股票日线", unit='只')
    SLEEP_PER_CALL = 1.5   # 50次/分钟限制 → 1.2s间隔，用1.5s留余量
    RETRY_SLEEP = 65        # 触发限频后等65秒重置计数器

    for code in pbar:
        success = False
        for _ in range(3):  # 最多3次尝试
            try:
                data = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
                if data is None or len(data) == 0:
                    failed_list.append(code)
                    pbar.set_postfix({'成功': len(stock_dict), '失败': len(failed_list)})
                    success = True
                    break
                stock_dict[code] = data
                pbar.set_postfix({'成功': len(stock_dict), '失败': len(failed_list)})
                success = True
                break
            except Exception as e:
                err_msg = str(e)
                if '频率' in err_msg or '频次' in err_msg or 'frequency' in err_msg.lower():
                    # 触发限频，等待计数器重置后重试
                    pbar.set_postfix({'状态': '限频等待60s...'})
                    time.sleep(RETRY_SLEEP)
                else:
                    # 其他错误（如网络抖动），短暂等待后重试
                    time.sleep(3)
        if not success:
            failed_list.append(code)
            pbar.set_postfix({'成功': len(stock_dict), '失败': len(failed_list)})

        time.sleep(SLEEP_PER_CALL)
        # 每 10 只存一次盘
        if len(stock_dict) % 10 == 0:
            save_cache()

    save_cache()

    if failed_list:
        print(f'[WARNING] Data fetch failed ({len(failed_list)}/{len(STOCK_LIST)}): {failed_list}')
    if not stock_dict:
        raise RuntimeError("所有股票数据获取均失败，请检查网络或 API Token。")

    return stock_dict

def prepare_data(STOCK_LIST,API,save=False):
    #调取数据
    data_dict = get_data_tushare(STOCK_LIST,API=API,start_date='20200101',end_date='20260531')

    # ---- 1. 数据清洗：重命名、转日期索引、去重、升序 ----
    for code in data_dict:
        df = data_dict[code]

        df = df[['trade_date','open','high','low','close','pct_chg','vol']]
        df = df.rename(columns = {
            'trade_date':'date',
            'open':'OPEN',
            'high':'HIGH',
            'low':'LOW',
            'close':'CLOSE',
            'pct_chg':'RETURN',
            'vol':'VOLUME'
        })

        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.drop_duplicates(subset=['date'])
        df = df.set_index('date')
        df = df.sort_index(ascending=True)

        # ---- 2. 预计算因子变量（升序后，shift 看向过去） ----
        # 日内形态
        df['VWAP']         = (df['HIGH'] + df['LOW'] + df['CLOSE']) / 3
        df['VWAP_DEV']     = df['CLOSE'] / df['VWAP'] - 1              # 收盘偏离均价
        df['BODY']         = (df['CLOSE'] - df['OPEN']) / df['OPEN']    # 实体涨跌幅
        df['UPPER_SHADOW'] = (df['HIGH'] - df[['OPEN','CLOSE']].max(axis=1)) / df['OPEN']  # 上影线
        df['LOWER_SHADOW'] = (df[['OPEN','CLOSE']].min(axis=1) - df['LOW']) / df['OPEN']   # 下影线
        df['PRICE_POS']    = (df['CLOSE'] - df['LOW']) / (df['HIGH'] - df['LOW'])  # 收盘在日内区间位置 0~1
        df['AMP']          = (df['HIGH'] - df['LOW']) / df['CLOSE'].shift(1)       # 日振幅（相对前收）
        df['GAP']          = df['OPEN'] / df['CLOSE'].shift(1) - 1                 # 隔夜跳空

        # 量价
        df['VOL_CHG']      = df['VOLUME'] / df['VOLUME'].shift(1)     # 量比（相对昨日）

        data_dict[code] = df

    # ---- 3. 收集所有日期，取并集并排序 ----
    all_dates = pd.DatetimeIndex([])
    for df in data_dict.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values(ascending=True)

    # ---- 4. 将每只股票 reindex 到统一日期 ----
    aligned_dict = {}
    for code, df in data_dict.items():
        aligned = df.reindex(all_dates)          # 缺失的日期自动变 NaN
        aligned_dict[code] = aligned

    # ---- 5. 检查 shape 是否一致 ----
    shapes = [df.shape for df in aligned_dict.values()]
    unique_shapes = set(shapes)

    if len(unique_shapes) == 1:
        pass
    else:
        raise ValueError("股票数据shape 不一致")

    # ---- 6. 转换为 3D numpy 数组 (股票 × 日期 × 特征) ----
    stock_codes = list(aligned_dict.keys())          # Axis-0: 股票代码
    dates = all_dates                                # Axis-1: 日期

    sample_df = aligned_dict[stock_codes[0]]
    feature_cols = sample_df.columns.tolist()        # 全数值列，无需 select_dtypes

    # 将每只股票的 DataFrame[feature_cols] 堆叠为 3D 数组
    data_3d = np.array([aligned_dict[code][feature_cols].values for code in stock_codes])

    if save:
    # ---- 7. 储存 3D 数组及元数据 ----
        save_path = "data\\stock_data_3d.npz"
        np.savez_compressed(
            save_path,
            data_3d=data_3d,
            stock_codes=np.array(stock_codes, dtype=str),
            dates=dates.values.astype('datetime64[D]'),  # 转为 numpy 日期格式
            feature_cols=np.array(feature_cols, dtype=str),
        )
        print(f"\n已保存至: {save_path}")
    

    return data_3d, stock_codes, dates, feature_cols

def check_pset(pset):
    print("=" * 60)
    print("Primitives:")
    print("=" * 60)
    for ret_type, prim_list in pset.primitives.items():
        for prim in prim_list:
            args_str = ", ".join(str(a) for a in prim.args)
            print(f"  {prim.name}({args_str}) -> {ret_type.__name__}")

def convert_code(code):
    """
    将6位数字股票代码转换为'代码.市场后缀'的格式
    """
    code_str = str(code).zfill(6)  # 确保代码是6位字符串，补齐前导零
    # 根据首位数字判断市场后缀
    if code_str.startswith('6'):
        return f"{code_str}.SH"  # 沪市（上海）
    elif code_str.startswith(('0', '3')):
        return f"{code_str}.SZ"  # 深市（深圳，含创业板）
    elif code_str.startswith(('8', '4')):
        return f"{code_str}.BJ"  # 北交所
    else:
        # 可在此添加其他规则，或返回原代码
        return code_str


def analyze_hof(hof: list, pset: gp.PrimitiveSet, stock_data: dict):
    """打印 Hall of Fame 中最佳因子的表达式和详细统计"""
    print(f"\n{'='*60}")
    print(f"  Hall of Fame — 最佳发现的因子 (Top {len(hof)})")
    print(f"{'='*60}\n")

    for rank, ind in enumerate(hof):
        fit_val = ind.fitness.values[0]
        expr_str = str(ind)

        print(f"--- Rank {rank+1} | fitness = {fit_val:.4f} | "
              f"nodes = {len(ind)} | height = {ind.height} ---")
        # 截断过长表达式
        if len(expr_str) > 600:
            print(f"  {expr_str[:600]}...")
        else:
            print(f"  {expr_str}")
        print()

def plot_evolution(stats_log: dict):
    """画出进化曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle("GP Evolution — Stock Factor Discovery", fontsize=14, fontweight="bold")

    gens = stats_log["gen"]

    # 图1: 适应度变化
    axes[0].plot(gens, stats_log["max_fitness"], "b-", lw=2, label="Max Fitness")
    axes[0].plot(gens, stats_log["avg_fitness"], "C1", ls="--", lw=2, label="Avg Fitness")
    axes[0].fill_between(gens, stats_log["min_fitness"], stats_log["max_fitness"],
                         alpha=0.12, color="blue")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Fitness")
    axes[0].set_title("Fitness Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 图2: 树大小 (bloat control)
    axes[1].plot(gens, stats_log["avg_size"], "g-", lw=2)
    axes[1].fill_between(gens, 0, stats_log["avg_size"], alpha=0.15, color="green")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Avg Tree Size (nodes)")
    axes[1].set_title("Bloat Control")
    axes[1].grid(True, alpha=0.3)

    # 图3: Fitness-Complexity 前沿
    sc = axes[2].scatter(stats_log["avg_size"], stats_log["max_fitness"],
                         c=gens, cmap="viridis", s=45, alpha=0.85)
    axes[2].set_xlabel("Avg Tree Size")
    axes[2].set_ylabel("Max Fitness")
    axes[2].set_title("Fitness–Complexity Trade-off")
    cbar = plt.colorbar(sc, ax=axes[2])
    cbar.set_label("Generation")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gp_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[图] gp_evolution.png — 进化曲线已保存")


def visualize_best_factor(hof_item, pset: gp.PrimitiveSet, stock_data: dict):
    """最佳因子截面分组回测: 每天按因子值分 5 组，看各组未来收益"""
    ind = hof_item
    func = gp.compile(ind, pset)

    # 对所有股票计算因子值
    records = []
    for code, feat in stock_data.items():
        try:
            fv = func(
                feat["open"], feat["high"], feat["low"], feat["close"],
                feat["volume"], feat["amount"], feat["vwap"], feat["ret_1d"],
                feat["ret_5d"], feat["ret_10d"], feat["amplitude"], feat["vol_ratio"],
            )
        except Exception:
            continue
        for t in range(MAX_WINDOW, len(fv)):
            if not np.isnan(fv[t]) and not np.isinf(fv[t]):
                records.append({
                    "date": feat["date"][t],
                    "code": code,
                    "factor": float(fv[t]),
                    "fwd_ret": float(feat["forward_ret"][t]),
                })

    df = pd.DataFrame(records)
    if df.empty:
        print("无有效因子值可用于分组回测")
        return None

    # 每天截面分 5 组
    df["group"] = df.groupby("date")["factor"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    df = df.dropna(subset=["group"])

    group_stats = df.groupby("group")["fwd_ret"].agg(["mean", "std", "count"])
    group_stats["t_stat"] = (group_stats["mean"] /
                             (group_stats["std"] / np.sqrt(group_stats["count"])))

    print(f"\n{'='*60}")
    print(f"  最佳因子截面分组回测 (Quintile Analysis)")
    print(f"  分组: Q0=最低因子值, Q4=最高因子值")
    print(f"{'='*60}")
    print(group_stats.to_string())

    if 4 in group_stats.index and 0 in group_stats.index:
        spread = group_stats.loc[4, "mean"] - group_stats.loc[0, "mean"]
        print(f"\n  Q4-Q0 多空收益差: {spread:.6f}")

    # 分组柱状图
    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = sorted(group_stats.index)
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(groups)))
    bars = ax.bar(groups, group_stats.loc[groups, "mean"], color=colors, edgecolor="black")
    # 标注数值
    for bar, val in zip(bars, group_stats.loc[groups, "mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.5f}", ha="center", va="bottom" if val > 0 else "top",
                fontsize=8, fontweight="bold")

    ax.set_xticks(groups)
    ax.set_xticklabels([f"Q{int(g)}" for g in groups])
    ax.set_xlabel("Factor Quintile")
    ax.set_ylabel(f"Mean Forward {FORWARD_RETURN}D Return")
    ax.set_title("Factor Quintile Return Distribution")
    ax.axhline(y=0, color="black", lw=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("factor_quintile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[图] factor_quintile.png — 分组回测已保存")

    return group_stats


# ============================================================
# 期货数据获取与预处理
# ============================================================

def prepare_futures_data(symbols=None, start_date='20200101', end_date='20260531', save=True):
    """
    获取主流期货合约主导行情数据，清洗对齐后转为 3D numpy 数组。

    与股票数据的区别:
      - 无 VWAP/VWAP_DEV (期货没有均价)
      - 新增 OPENINTEREST (持仓量) 和 OI_CHG (持仓变化率)
      - 使用 akshare futures_main_sina 接口（免费无需 token）

    Parameters
    ----------
    symbols : list of str  期货品种代码列表，如 ['CU','IF','RB']，默认使用 FUTURES_LIST
    start_date : str
    end_date : str
    save : bool  是否保存为 npz 文件

    Returns
    -------
    data_3d : np.ndarray  (n_contracts, n_dates, n_features)
    contract_codes : list of str
    trade_dates : pd.DatetimeIndex
    feature_cols : list of str
    """
    import akshare as ak

    if symbols is None:
        from parameters import FUTURES_LIST
        symbols = FUTURES_LIST

    print(f"[期货] 获取 {len(symbols)} 个品种主导合约数据...")
    contract_dict = {}
    failed = []

    for symbol in tqdm(symbols, desc="期货行情", unit='品种'):
        try:
            # futures_main_sina 获取主导合约（主力/次主力连续）
            data = ak.futures_main_sina(
                symbol=symbol + '0',
                start_date=start_date,
                end_date=end_date
            )
            if data is None or len(data) == 0:
                failed.append(symbol)
                continue

            data = data.rename(columns={
                '日期': 'date',
                '开盘价': 'OPEN',
                '最高价': 'HIGH',
                '最低价': 'LOW',
                '收盘价': 'CLOSE',
                '成交量': 'VOLUME',
                '持仓量': 'OPENINTEREST',
                '动态结算价': 'DYNAMIC_PRICE',
            })

            data['date'] = pd.to_datetime(data['date'])
            # 移除时区信息（不同品种可能有时区差异）
            if data['date'].dt.tz is not None:
                data['date'] = data['date'].dt.tz_localize(None)

            data = data.drop_duplicates(subset=['date'])
            data = data.set_index('date')
            data = data.sort_index(ascending=True)

            # ---- 计算衍生特征 ----
            data['RETURN'] = data['CLOSE'].pct_change() * 100.0  # 日收益率 (%)

            # 日内形态
            vwap_proxy = (data['HIGH'] + data['LOW'] + data['CLOSE']) / 3.0
            data['BODY'] = (data['CLOSE'] - data['OPEN']) / data['OPEN'].replace(0, np.nan)
            data['UPPER_SHADOW'] = (data['HIGH'] - data[['OPEN', 'CLOSE']].max(axis=1)) / data['OPEN'].replace(0, np.nan)
            data['LOWER_SHADOW'] = (data[['OPEN', 'CLOSE']].min(axis=1) - data['LOW']) / data['OPEN'].replace(0, np.nan)
            data['PRICE_POS'] = (data['CLOSE'] - data['LOW']) / (data['HIGH'] - data['LOW']).replace(0, np.nan)
            data['AMP'] = (data['HIGH'] - data['LOW']) / data['CLOSE'].shift(1).replace(0, np.nan)
            data['GAP'] = data['OPEN'] / data['CLOSE'].shift(1).replace(0, np.nan) - 1.0

            # 量仓
            data['VOL_CHG'] = data['VOLUME'] / data['VOLUME'].shift(1).replace(0, np.nan)
            data['OI_CHG'] = data['OPENINTEREST'] / data['OPENINTEREST'].shift(1).replace(0, np.nan)

            # 保留 FUTURES_FEATURE_COLS 中的列
            from parameters import FUTURES_FEATURE_COLS
            keep_cols = [c for c in FUTURES_FEATURE_COLS if c in data.columns
                         or c in ['RETURN', 'BODY', 'UPPER_SHADOW', 'LOWER_SHADOW',
                                  'PRICE_POS', 'AMP', 'GAP', 'VOL_CHG', 'OI_CHG']]
            data = data[[c for c in FUTURES_FEATURE_COLS if c in data.columns]]

            contract_dict[symbol] = data
        except Exception as e:
            failed.append(symbol)
            print(f"  [{symbol}] 获取失败: {e}")

    if failed:
        print(f"[期货] 获取失败 ({len(failed)}/{len(symbols)}): {failed}")
    if not contract_dict:
        raise RuntimeError("所有期货数据获取均失败")

    # ---- 对齐日期 ----
    all_dates = pd.DatetimeIndex([])
    for df in contract_dict.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values(ascending=True)

    aligned = {}
    for code, df in contract_dict.items():
        aligned[code] = df.reindex(all_dates)

    contract_codes = list(aligned.keys())
    dates = all_dates

    # 确保所有列一致
    sample_df = aligned[contract_codes[0]]
    feature_cols = [c for c in sample_df.columns if c not in ['DYNAMIC_PRICE']]
    feature_cols = [c for c in feature_cols if c in sample_df.columns]

    # ---- 转 3D 数组 ----
    data_3d = np.array([aligned[code][feature_cols].values for code in contract_codes],
                       dtype=np.float64)

    n_nan = np.isnan(data_3d).sum()
    print(f"[期货] 3D 数组: {data_3d.shape} | NaN 占比: {n_nan/data_3d.size:.1%}")
    print(f"[期货] 日期范围: {dates[0].date()} ~ {dates[-1].date()}")

    if save:
        os.makedirs("data", exist_ok=True)
        save_path = "data/futures_data_3d.npz"
        np.savez_compressed(
            save_path,
            data_3d=data_3d,
            contract_codes=np.array(contract_codes, dtype=str),
            dates=dates.values.astype('datetime64[D]'),
            feature_cols=np.array(feature_cols, dtype=str),
        )
        print(f"[期货] 已保存: {save_path}")

    return data_3d, contract_codes, dates, feature_cols
