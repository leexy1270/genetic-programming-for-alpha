from parameters import *
from tools import *
from build_pset import build_pset, genHalfAndHalfSafe,genFullSafe,genGrowSafe
from deap import creator,base,tools,gp
from evaluate import _HIT, _MISS, _compile_cache
from evaluate import init_worker, evaluate_worker

import operator
import os
import time
import random
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免无GUI环境报错
import matplotlib.pyplot as plt

def main():
    # ================================================================
    # 硬件优化初始化
    # ================================================================
    # 多进程模式下，每进程 BLAS 线程数设为 1，避免 MKL/OpenBLAS 内部
    # 多线程与进程级并行竞争 CPU 资源（over-subscription）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    N_CORES = os.cpu_count() or 4
    print(f'[硬件] CPU 核心数: {N_CORES} | 评估并行化已启用')

    # ---- NumPy BLAS 后端检测 ----
    try:
        import io
        buf = io.StringIO()
        np.show_config(file=buf)
        config_str = buf.getvalue()
        if 'mkl' in config_str.lower():
            blas_backend = 'MKL ✓'
        elif 'blis' in config_str.lower():
            blas_backend = 'BLIS ✓'
        elif 'openblas' in config_str.lower():
            blas_backend = 'OpenBLAS ✓'
        else:
            blas_backend = '未知'
    except Exception:
        blas_backend = '未知'
    print(f'[硬件] NumPy 后端: {blas_backend} | OMP/MKL 线程数 = 1 (进程级并行替代)')

    # ---------- 1. 获取所需股票数据 ----------
    print('\n  开始获取数据...')
    try:
        loaded = np.load("data\\stock_data_3d.npz", allow_pickle=True)
        data = loaded["data_3d"]
        stock_codes = loaded["stock_codes"].tolist()
        trade_dates = pd.DatetimeIndex(loaded["dates"])
        feature_cols = loaded["feature_cols"].tolist()
    except:
        try:
            with open('API_key','r') as f:
                api = f.readline()
        except:
            api = input('输入tushare_api=')
        data,stock_codes,trade_dates,feature_cols = prepare_data(STOCK_LIST=ZZ500_LIST,API=api)

    # ---- 2. 构建原语集，设定因子可能含有的变量与运算符 ----
    print('\n  开始构建原语集...')
    pset = build_pset(feature_cols)

    # ---- 多进程 Pool 初始化 ----
    # data 用 np.memmap 共享，避免每个 worker pickle 复制 60MB。
    os.makedirs("data", exist_ok=True)
    _mmap_path = os.path.abspath("data/stock_data_mmap.dat")
    _need_rebuild = True
    if os.path.exists(_mmap_path) and os.path.getsize(_mmap_path) == data.nbytes:
        _need_rebuild = False
    if _need_rebuild:
        _mmap = np.memmap(_mmap_path, dtype=data.dtype, mode='w+', shape=data.shape)
        _mmap[:] = data[:]
        _mmap.flush()
        del _mmap

    # Worker 数量：保守默认 4（避免 CPU 过热/内存耗尽导致崩溃）
    # 可通过环境变量 GP_WORKERS 自行调整，例如: set GP_WORKERS=8
    _env_workers = os.environ.get("GP_WORKERS", "")
    if _env_workers and _env_workers.isdigit():
        safe_workers = max(1, min(int(_env_workers), N_CORES))
    else:
        safe_workers = min(8, N_CORES)
    print(f'[硬件] 多进程 Pool: {safe_workers}/{N_CORES} workers '
          f'(memmap 共享数据, 设置 GP_WORKERS 环境变量可调整)')

    pool = multiprocessing.Pool(
        processes=safe_workers,
        initializer=init_worker,
        initargs=(_mmap_path, data.shape, data.dtype, feature_cols),
    )

    # ------- 3. 创建进化过程中所需的类型 -------
    print('\n  流程初始化...')
    # FitnessMax + PrimitiveTree ——> Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # pset ——>     expr     ——> individual ——> popluation
    #          + Individual
    toolbox = base.Toolbox()
    toolbox.register("expr", genHalfAndHalfSafe, pset=pset, min_=1, max_=MAX_TREE_HEIGHT)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #评估（使用 worker 包装函数，数据通过 Pool initializer 注入，避免 pickle pset）
    toolbox.register("evaluate", evaluate_worker)

    # 注册遗传算子
        # 选择: 锦标赛
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
        # 交叉：单点交叉
    toolbox.register("mate", gp.cxOnePoint)
        # 变异: 统一变异
    toolbox.register("expr_mut", genFullSafe, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        # 限制树深度防止过拟合
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT + 3))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT + 3))

    # 注册多进程并行 map
    toolbox.register("map", pool.map)

    # ---- Step 4: 初始化种群（支持热启动） ----
    print('\n  创建初始种群...')
    hof = tools.HallOfFame(5)
    seed_individuals = []

    if WARM_START:
        from gtja191 import parse_expr
        print(f"\n[Warm Start] Loading factor expressions...")
        warm_exprs = {}

        # 方式1: 从文本文件读取（每行格式: alphaNNN = expr 或 expr）
        if os.path.exists(WARM_START_FILE):
            print(f"  Reading from file: {WARM_START_FILE}")
            with open(WARM_START_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '--' in line:
                        continue
                    if '=' in line:
                        parts = line.split('=', 1)
                        name = parts[0].strip()
                        expr_str = parts[1].strip()
                    else:
                        expr_str = line
                        name = f'warm_{len(warm_exprs):03d}'
                    if expr_str and expr_str != '--':
                        warm_exprs[name] = expr_str

        # 方式2: 从 gtja191 模块读取（fallback）
        if not warm_exprs:
            from gtja191 import FACTOR_EXPRS
            print(f"  Loading from gtja191 module...")
            warm_exprs = {k: v for k, v in FACTOR_EXPRS.items() if v is not None}

        print(f"  Available expressions: {len(warm_exprs)}")

        for name, expr_str in warm_exprs.items():
            if len(seed_individuals) >= POPULATION_SIZE:
                break
            try:
                tree = parse_expr(expr_str, pset)
                ind = creator.Individual(tree)
                seed_individuals.append(ind)
            except Exception:
                pass
        print(f"  Parsed {len(seed_individuals)} seed individuals for warm-start")

    # 使用随机个体填充种群空位
    pop = toolbox.population(n=POPULATION_SIZE)
    if seed_individuals:
        n_seeds = min(len(seed_individuals), POPULATION_SIZE)
        pop[:n_seeds] = seed_individuals[:n_seeds]
        # 剩余的随机个体
        if n_seeds < POPULATION_SIZE:
            random_fill = toolbox.population(n=POPULATION_SIZE - n_seeds)
            pop[n_seeds:] = random_fill

    # 评估初始种群（多进程并行）
    print(f"\n  评估初始种群（{safe_workers} 进程并行）...")
    unfit_init = [ind for ind in pop if not ind.fitness.valid]
    if unfit_init:
        fitnesses = toolbox.map(toolbox.evaluate, unfit_init)
        for ind, (fit_val,) in zip(unfit_init, fitnesses):
            ind.fitness.values = (fit_val,)
    hof.update(pop)
    valid_fits0 = [ind.fitness.values[0] for ind in pop if ind.fitness.values[0] != -999.0]
    print(f"  初始种群: maxF={max(valid_fits0):.4f}" if valid_fits0 else "  初始种群: all invalid")
    if seed_individuals:
        seed_fits = [ind.fitness.values[0] for ind in pop[:len(seed_individuals)] if ind.fitness.values[0] != -999.0]
        if seed_fits:
            print(f"  热启动种子: maxF={max(seed_fits):.4f}  avgF={np.mean(seed_fits):.4f}")


    # 设置统计信息
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    stats_log = {"gen": [], "avg_fitness": [], "max_fitness": [],
                    "min_fitness": [], "avg_size": []}

    print(f"\n{'='*60}")
    print(f"  开始遗传编程进化")
    print(f"  种群: {POPULATION_SIZE} | 代数: {N_GENERATIONS} | "
            f"Pc={CROSSOVER_PROB} | Pm={MUTATION_PROB}")
    print(f"{'='*60}\n")

    for gen in range(N_GENERATIONS):
        t0 = time.time()

        # ---- 选择 ----
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # ---- 交叉 ----
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # ---- 变异 ----
        for mutant in offspring:
            if random.random() < MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # ---- 评估新个体（多进程并行） ----
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        if invalid:
            fitnesses = toolbox.map(toolbox.evaluate, invalid)
            for ind, (fit_val,) in zip(invalid, fitnesses):
                ind.fitness.values = (fit_val,)

        # ---- 精英保留 ----
        pop[:] = tools.selBest(pop + offspring, POPULATION_SIZE)
        hof.update(pop)

        # ---- 日志 ----
        valid_fits = [ind.fitness.values[0] for ind in pop if ind.fitness.valid]
        elapsed = time.time() - t0

        avg_f = np.mean(valid_fits) if valid_fits else 0
        max_f = np.max(valid_fits) if valid_fits else 0
        min_f = np.min(valid_fits) if valid_fits else 0
        avg_sz = np.mean([len(ind) for ind in pop])

        stats_log["gen"].append(gen)
        stats_log["avg_fitness"].append(avg_f)
        stats_log["max_fitness"].append(max_f)
        stats_log["min_fitness"].append(min_f)
        stats_log["avg_size"].append(avg_sz)

        hof_str = ", ".join(
            [f"{hof[i].fitness.values[0]:.3f}" for i in range(min(3, len(hof)))]
        )
        
        print(f"Gen {gen:3d} | maxF={max_f:.4f} | avgF={avg_f:.4f} | "
                f"sz={avg_sz:.0f} | HoF=[{hof_str}] | {elapsed:.1f}s")

    # ---- Step 5: 结果 ----
    print("\n[5/6] 结果分析...")

    # 打印最佳因子表达式
    print(f"\n{'='*60}")
    print(f"  Best Factor Expression")
    print(f"{'='*60}")
    for i in range(min(5, len(hof))):
        print(f"  Rank {i+1}: fitness={hof[i].fitness.values[0]:.4f}  {str(hof[i])}")

    # ---- Step 6: 保存结果 ----
    os.makedirs("result", exist_ok=True)

    # 6a. 保存 Hall of Fame 表达式为可读文本
    with open("result\\best_factors.txt", "w", encoding="utf-8") as f:
        f.write(f"# GP Factor Discovery Results\n")
        f.write(f"# Generations: {N_GENERATIONS} | Population: {POPULATION_SIZE}\n")
        f.write(f"# FORWARD_RETURN: {FORWARD_RETURN}D | PARSIMONY_C: {PARSIMONY_C}\n\n")
        for i, ind in enumerate(hof):
            f.write(f"Rank {i+1} | fitness={ind.fitness.values[0]:.6f} | "
                    f"nodes={len(ind)} | height={ind.height}\n")
            f.write(f"{str(ind)}\n\n")
    print("[Save] result/best_factors.txt")

    # 6b. 保存进化日志为 CSV
    pd.DataFrame(stats_log).to_csv("result\\evolution_log.csv", index=False)
    print("[Save] result/evolution_log.csv")

    # 6c. 计算并保存 HoF 因子值 (stock × date DataFrame)
    print("[Save] Computing factor values for Hall of Fame...")
    hof_values = {}
    for i, ind in enumerate(hof):
        try:
            func = gp.compile(ind, pset)
            args_3d = [data[:, :, j] for j in range(data.shape[2])]
            fv = func(*args_3d)
            fv = np.atleast_1d(np.squeeze(fv))
            if fv.ndim == 2 and fv.shape[1] == len(trade_dates):
                df = pd.DataFrame(fv.T, index=trade_dates, columns=stock_codes)
                hof_values[f"rank{i+1}"] = df
        except Exception as e:
            print(f"  Rank {i+1}: compute failed - {e}")

    if hof_values:
        # 保存为 npz (数值) + csv (表达式元信息)
        np.savez_compressed("result\\hof_factor_values.npz",
                            **{k: v.values for k, v in hof_values.items()},
                            stock_codes=np.array(stock_codes),
                            dates=trade_dates.values)
        print(f"[Save] result/hof_factor_values.npz ({len(hof_values)} factors × {len(stock_codes)} stocks × {len(trade_dates)} dates)")

        # 保存每个因子的截面均值时间序列
        hof_means = pd.DataFrame({k: v.mean(axis=1) for k, v in hof_values.items()}, index=trade_dates)
        hof_means.to_csv("result\\hof_factor_means.csv")
        print("[Save] result/hof_factor_means.csv")

    # 6d. 保存完整 checkpoint (pickle)，可恢复进化
    import pickle
    checkpoint = {
        "population": pop,
        "halloffame": hof,
        "stats_log": stats_log,
        "feature_cols": feature_cols,
        "stock_codes": stock_codes,
        "trade_dates": trade_dates,
        "generation": N_GENERATIONS,
        "random_state": random.getstate(),
    }
    with open("result\\gp_checkpoint.pkl", "wb") as f:
        pickle.dump(checkpoint, f)
    print("[Save] result/gp_checkpoint.pkl")

    # ================================================================
    # ---- Step 7: 因子分析与可视化 (Factor Analysis & Visualization) ----
    # ================================================================
    # 本步骤包含以下子模块:
    #   7a. 进化过程可视化 —— 收敛曲线、Pareto前沿、HoF分组收益、IC衰减
    #   7b. 逐因子深度分析 —— 每个HoF因子的IC统计、分组收益、多空组合、
    #       自相关衰减、换手率，并生成单因子分析图
    #   7c. HoF横向对比     —— 累计多空收益对比、ICIR柱状图、因子相关性热力图
    #   7d. 综合分析报告    —— 汇总CSV + 控制台报告
    # ================================================================

    print(f"\n{'='*60}")
    print(f"  Step 7: 因子分析与可视化")
    print(f"{'='*60}")

    # ---- 导入分析模块 ----
    from visualization import plot_evolution_report
    from factor_analysis import FactorAnalyzer
    from deap import gp as gp_module

    # ================================================================
    # 7a. 进化过程可视化 (Evolution Visualization)
    # ================================================================
    # 生成4张出版物级别的进化过程图:
    #   fig1_convergence.png — 适应度收敛 + 树膨胀控制
    #   fig2_pareto.png      — 适应度-复杂度权衡 (Pareto前沿代理)
    #   fig3_quintiles.png   — Top-3 HoF因子的分组收益分布
    #   fig4_ic_decay.png    — Top-3 HoF因子在不同预测周期下的IC衰减
    # ================================================================
    print("\n  [7a] 进化过程可视化...")
    try:
        plot_evolution_report(stats_log, hof, pset, data, feature_cols, trade_dates, stock_codes)
    except Exception as e:
        print(f"  [7a] 进化可视化失败: {e}")

    # ================================================================
    # 7b. 逐因子深度分析 (Per-Factor Deep Analysis)
    # ================================================================
    # 对Hall of Fame中每个因子，使用FactorAnalyzer进行全方位评估:
    #   - IC序列统计: 均值、标准差、ICIR、正向比例、t统计量
    #   - 分组收益: 按因子值分5组(Q0~Q4)，检验单调性
    #   - 多空组合: Top 20% vs Bottom 20%的多空收益、信息比率、胜率
    #   - 因子自相关衰减: 衡量因子值的时序稳定性（持久性）
    #   - 换手率: Top组的日度持仓变动比例
    # 并为每个因子生成3张分析图:
    #   analysis_ic_series_rank{N}.png  — IC时间序列 + 累计IC
    #   analysis_quintiles_rank{N}.png  — 五分组平均收益柱状图
    #   analysis_decay_rank{N}.png      — 因子自相关衰减图
    # ================================================================
    print("\n  [7b] 逐因子深度分析...")

    # ---- 计算前向收益 (5日持有期) ----
    # 从3D数据中提取日收益率 → 对数累加 → exp还原，得到未来5日累计收益
    ret_idx = feature_cols.index('RETURN') if 'RETURN' in feature_cols else 4
    daily_ret = data[:, :, ret_idx] / 100.0               # 百分比 → 小数
    log_ret = np.log(1.0 + daily_ret)                      # 对数收益（累加无偏）
    n_dates = data.shape[1]
    fwd_n = FORWARD_RETURN                                  # 前向持有天数（默认5）
    fwd_ret = np.full((data.shape[0], n_dates), np.nan, dtype=np.float64)
    for t in range(n_dates - fwd_n):
        # t日买入 → 持有 fwd_n 日 → 累计收益
        fwd_ret[:, t] = np.exp(np.sum(log_ret[:, t+1:t+1+fwd_n], axis=1)) - 1.0

    # ---- 对每个HoF因子逐一分析 ----
    hof_results = []   # 存储每个因子的分析摘要（用于后续横向对比）

    for rank_i, ind in enumerate(hof):
        rank_label = f"Rank{rank_i+1}"
        print(f"\n    --- HoF {rank_label} ---")

        try:
            # -- 编译表达式树并计算全样本因子值 --
            func = gp_module.compile(ind, pset)
            # 将3D数组按特征维度拆分为参数列表，传入表达式函数
            args_3d = [data[:, :, j] for j in range(data.shape[2])]
            fv_raw = func(*args_3d)
            fv_raw = np.atleast_1d(np.squeeze(fv_raw))

            # 确保因子值是2D数组 (n_stocks × n_dates)
            if fv_raw.ndim != 2 or fv_raw.shape[1] != n_dates:
                print(f"      [跳过] 因子值形状异常: {fv_raw.shape}")
                continue

            # -- 创建因子分析器 --
            analyzer = FactorAnalyzer(
                factor_values=fv_raw,
                forward_returns=fwd_ret,
                trade_dates=trade_dates,
                stock_codes=stock_codes,
                factor_name=f"HoF_{rank_label}"
            )

            # -- 7b-1. IC 统计 --
            # Rank IC: 每日截面上因子值与未来收益的Spearman秩相关系数
            # ICIR = mean(IC) / std(IC)，衡量IC的稳定性
            ic_summary = analyzer.ic_summary()
            print(f"      IC Mean={ic_summary['IC_Mean']:.6f}  "
                  f"Std={ic_summary['IC_Std']:.6f}  "
                  f"ICIR={ic_summary['ICIR']:.4f}  "
                  f"PosRatio={ic_summary['Pos_Ratio']:.2%}  "
                  f"t={ic_summary['t_stat']:.2f}  "
                  f"N={ic_summary['N']}")

            # -- 7b-2. 分组收益 --
            # 每日按因子值将股票分为5组(Q0最低→Q4最高)
            # 计算各组平均未来收益，检验Q4-Q0多空spread
            qdf = analyzer.quintile_analysis()
            if not qdf.empty:
                q_means = qdf.groupby('quintile')['mean_ret'].mean()
                spread = q_means.get(4, np.nan) - q_means.get(0, np.nan)
                q_str = " | ".join([f"Q{q}: {q_means.get(q, np.nan):.6f}" for q in range(5)])
                print(f"      分组收益: {q_str}")
                print(f"      Q4-Q0 Spread: {spread:.6f}")

            # -- 7b-3. 多空组合 --
            # 做多因子值最高的20%股票，做空最低的20%
            # 计算每日多空收益序列 → 均值、波动率、信息比率、胜率
            ls_df = analyzer.long_short_returns(top_quantile=0.2, bottom_quantile=0.2)
            if not ls_df.empty:
                ls_mean = ls_df['ls'].mean()
                ls_std = ls_df['ls'].std()
                ls_ir = ls_mean / ls_std if ls_std > 1e-12 else 0
                ls_win = np.mean(ls_df['ls'] > 0)
                print(f"      多空组合: LS_Ret={ls_mean:.6f}  "
                      f"LS_Std={ls_std:.6f}  "
                      f"LS_IR={ls_ir:.4f}  "
                      f"WinRate={ls_win:.2%}")
            else:
                ls_mean = np.nan

            # -- 7b-4. 因子自相关衰减 --
            # 计算因子截面均值的自相关系数(1~30日滞后)
            # 高自相关 → 因子值稳定 → 换手率低 → 交易成本低
            ac = analyzer.decay_analysis(max_lag=30)
            ac_5d = ac[5] if len(ac) > 5 else np.nan   # 5日自相关（关键指标）
            ac_20d = ac[20] if len(ac) > 20 else np.nan # 20日自相关
            print(f"      自相关: AC(5d)={ac_5d:.4f}  AC(20d)={ac_20d:.4f}")

            # -- 7b-5. 换手率 --
            # Top 20%组合的日度平均换手率（1 - 持仓重叠率）
            turnover = analyzer.factor_turnover(top_quantile=0.2)
            print(f"      换手率: {turnover:.2%}")

            # -- 7b-6. 生成单因子分析图 --
            # 每张图以 rank 编号保存，避免覆盖
            analyzer.plot_ic_series(
                savepath=f"result/analysis_ic_series_rank{rank_i+1}.png"
            )
            analyzer.plot_quintile_returns(
                savepath=f"result/analysis_quintiles_rank{rank_i+1}.png"
            )
            analyzer.plot_decay(
                savepath=f"result/analysis_decay_rank{rank_i+1}.png"
            )

            # -- 7b-7. 收集结果摘要（供7c横向对比）--
            hof_results.append({
                'Rank': rank_i + 1,
                'Expression': str(ind),
                'Fitness': ind.fitness.values[0],
                'IC_Mean': ic_summary['IC_Mean'],
                'IC_Std': ic_summary['IC_Std'],
                'ICIR': ic_summary['ICIR'],
                'Pos_Ratio': ic_summary['Pos_Ratio'],
                't_stat': ic_summary['t_stat'],
                'IC_N': ic_summary['N'],
                'Q4_Q0_Spread': spread if not qdf.empty else np.nan,
                'LS_Ret': ls_mean,
                'LS_IR': ls_ir if not ls_df.empty else np.nan,
                'LS_WinRate': ls_win if not ls_df.empty else np.nan,
                'AC_5d': ac_5d,
                'AC_20d': ac_20d,
                'Turnover': turnover,
                'Nodes': len(ind),
                'Height': ind.height,
            })

        except Exception as e:
            print(f"      [错误] 分析失败: {e}")
            # 即使分析失败，也记录基本信息
            hof_results.append({
                'Rank': rank_i + 1,
                'Expression': str(ind),
                'Fitness': ind.fitness.values[0],
                'IC_Mean': np.nan, 'IC_Std': np.nan, 'ICIR': np.nan,
                'Pos_Ratio': np.nan, 't_stat': np.nan, 'IC_N': 0,
                'Q4_Q0_Spread': np.nan, 'LS_Ret': np.nan, 'LS_IR': np.nan,
                'LS_WinRate': np.nan, 'AC_5d': np.nan, 'AC_20d': np.nan,
                'Turnover': np.nan, 'Nodes': len(ind), 'Height': ind.height,
            })

    # ================================================================
    # 7c. HoF因子横向对比 (Cross-Factor Comparison)
    # ================================================================
    # 将所有HoF因子放在一起对比，揭示因子间的相对优劣和互补性:
    #   - 累计多空收益曲线: 比较各因子的多空组合净值走势
    #   - ICIR柱状图: 一目了然地比较各因子的风险调整后预测能力
    #   - 因子相关性热力图: 检查因子间是否过度同质化
    # ================================================================
    print(f"\n  [7c] HoF因子横向对比...")

    if len(hof_results) >= 2:
        # -- 7c-1. 累计多空收益对比 --
        # 计算每个HoF因子的多空组合每日收益，累乘得到净值曲线
        # 各因子在同一坐标系下对比，直观展示收益走势和回撤特征
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图: 累计多空净值
        ax = axes[0]
        ls_curves = {}  # 存储各因子的累计多空收益，用于后续相关性计算
        for rank_i, ind in enumerate(hof):
            try:
                func = gp_module.compile(ind, pset)
                args_3d = [data[:, :, j] for j in range(data.shape[2])]
                fv_raw = func(*args_3d)
                fv_raw = np.atleast_1d(np.squeeze(fv_raw))
                if fv_raw.ndim != 2:
                    continue
                analyzer = FactorAnalyzer(fv_raw, fwd_ret, trade_dates, stock_codes,
                                          f"HoF_Rank{rank_i+1}")
                ls_df = analyzer.long_short_returns(top_quantile=0.2, bottom_quantile=0.2)
                if ls_df.empty:
                    continue
                # 累乘: (1+r1)*(1+r2)*... - 1 = 累计收益
                cum_ret = (1.0 + ls_df['ls'].values).cumprod() - 1.0
                ls_curves[f"Rank{rank_i+1}"] = ls_df['ls'].values
                ax.plot(ls_df['date'] if trade_dates is not None else range(len(cum_ret)),
                        cum_ret, lw=1.5, alpha=0.85,
                        label=f"Rank{rank_i+1} (IR={ls_df['ls'].mean()/ls_df['ls'].std() if ls_df['ls'].std()>1e-12 else 0:.2f})")
            except Exception:
                continue
        ax.axhline(y=0, color='black', lw=0.5, ls='--')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Cumulative Long-Short Return')
        ax.set_title(f'HoF Factors — Cumulative Long-Short ({fwd_n}D Forward)')
        ax.legend(fontsize=7, frameon=True)
        ax.grid(alpha=0.3)

        # 右图: ICIR 柱状图对比
        ax = axes[1]
        ranks = [r['Rank'] for r in hof_results]
        icirs = [r['ICIR'] for r in hof_results]
        colors_bar = plt.cm.viridis(np.linspace(0.15, 0.9, len(ranks)))
        bars = ax.bar(ranks, icirs, color=colors_bar, edgecolor='black', lw=0.5)
        # 在柱顶标注ICIR数值
        for bar, val in zip(bars, icirs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                        fontsize=9, fontweight='bold')
        ax.axhline(y=0, color='black', lw=0.5)
        ax.set_xlabel('HoF Rank')
        ax.set_ylabel('ICIR')
        ax.set_title('Factor ICIR Comparison')
        ax.set_xticks(ranks)
        ax.grid(axis='y', alpha=0.3)

        fig.suptitle('Hall of Fame — Cross-Factor Comparison', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig("result/fig5_cross_factor_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [7c] result/fig5_cross_factor_comparison.png")

        # -- 7c-2. 因子相关性热力图 --
        # 计算各因子多空日收益序列之间的Pearson相关系数
        # 高相关性 → 因子同质化 → 提示需要增加多样性
        if len(ls_curves) >= 2:
            # 用 concat 处理不同长度（不同因子有效天数可能不同）
            ls_series = {k: pd.Series(v) for k, v in ls_curves.items()}
            ls_df_all = pd.concat(ls_series, axis=1)
            corr_matrix = ls_df_all.corr()

            fig, ax = plt.subplots(figsize=(max(6, len(ls_curves)*1.2),
                                           max(5, len(ls_curves)*1.0)))
            im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            # 在每个格子中标注相关系数
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                            ha='center', va='center', fontsize=10,
                            fontweight='bold',
                            color='white' if abs(corr_matrix.values[i, j]) > 0.6 else 'black')
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticks(range(len(corr_matrix.index)))
            ax.set_yticklabels(corr_matrix.index)
            ax.set_title('HoF Factor Long-Short Return Correlation')
            plt.colorbar(im, ax=ax, label='Pearson r')
            plt.tight_layout()
            fig.savefig("result/fig6_factor_correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  [7c] result/fig6_factor_correlation_heatmap.png")

    # ================================================================
    # 7d. 综合分析报告 (Comprehensive Report)
    # ================================================================
    # 将所有分析结果汇总并持久化:
    #   - hof_analysis_summary.csv: 所有因子的关键指标表格
    #   - 控制台打印: 按ICIR排序的完整报告
    # ================================================================
    print(f"\n  [7d] 综合分析报告...")

    if hof_results:
        # -- 保存CSV --
        summary_df = pd.DataFrame(hof_results)
        # 按Fitness降序排列（适应度越高越好）
        summary_df = summary_df.sort_values('Fitness', ascending=False)
        summary_df.to_csv("result\\hof_analysis_summary.csv", index=False,
                          encoding='utf-8-sig')
        print(f"  [7d] result/hof_analysis_summary.csv "
              f"({len(summary_df)} factors × {len(summary_df.columns)} metrics)")

        # -- 控制台报告 --
        print(f"\n  {'='*70}")
        print(f"  [Report] Hall of Fame -- 综合分析报告")
        print(f"  {'='*70}")
        print(f"  进化代数: {N_GENERATIONS}  |  种群规模: {POPULATION_SIZE}  |  "
              f"前向收益: {FORWARD_RETURN}D")
        print(f"  复杂度惩罚系数: {PARSIMONY_C}  |  HoF容量: {len(hof)}")
        print(f"\n  {'─'*70}")
        print(f"  {'Rank':<6} {'Fitness':>8} {'ICIR':>8} {'LS_IR':>8} "
              f"{'IC_Mean':>10} {'AC_5d':>7} {'Turnover':>9} {'Nodes':>6}")
        print(f"  {'─'*70}")
        # 安全格式化辅助函数（处理NaN值）
        def fmt(v, w, d=4):
            """数值格式化，NaN显示为 'N/A'，正常值保留d位小数"""
            return f"{v:>{w}.{d}f}" if not (isinstance(v, float) and np.isnan(v)) else f"{'N/A':>{w}}"
        def fmt_pct(v, w):
            """百分比格式化，NaN显示为 'N/A'"""
            return f"{v:>{w}.2%}" if not (isinstance(v, float) and np.isnan(v)) else f"{'N/A':>{w}}"

        for r in summary_df.to_dict('records'):
            print(f"  {r['Rank']:<6} {fmt(r['Fitness'], 8)} {fmt(r['ICIR'], 8)} "
                  f"{fmt(r['LS_IR'], 8)} {fmt(r['IC_Mean'], 10, 6)} "
                  f"{fmt(r['AC_5d'], 7)} {fmt_pct(r['Turnover'], 9)} "
                  f"{r['Nodes']:>6}")

        # -- 打印最佳因子完整表达式 --
        print(f"\n  {'─'*70}")
        print(f"  [Top 3] 最佳因子 (Top 3) 完整表达式:")
        print(f"  {'─'*70}")
        for r in summary_df.head(3).to_dict('records'):
            expr = r['Expression']
            if len(expr) > 200:
                expr = expr[:200] + "..."
            icir_str = f"ICIR={r['ICIR']:.4f}" if not (isinstance(r['ICIR'], float) and np.isnan(r['ICIR'])) else "ICIR=N/A"
            print(f"\n  [Rank {r['Rank']}] Fitness={r['Fitness']:.4f}  {icir_str}")
            print(f"  {expr}")
        print(f"\n  {'='*70}\n")

    print(f"[Step 7] 因子分析与可视化 — 全部完成\n")

    # ---- 清理多进程资源 ----
    pool.close()
    pool.join()

    # 编译缓存统计
    total_compile = _HIT + _MISS
    if total_compile > 0:
        print(f"[缓存] 表达式编译: {_HIT} hits / {_MISS} misses "
              f"(命中率 {_HIT/total_compile*100:.1f}%, 缓存 {len(_compile_cache)} 个)")

    return pop, hof, stats_log

if __name__ == '__main__':
    main()

