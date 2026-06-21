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

    # ---------- 1. 获取所需数据 ----------
    print('\n  开始获取数据...')
    if DATA_SOURCE == "futures":
        stock_codes_key = "contract_codes"
        _data_npz = "data/futures_data_3d.npz"
        _mmap_name = "data/futures_data_mmap.dat"
        try:
            loaded = np.load(_data_npz, allow_pickle=True)
            data = loaded["data_3d"]
            stock_codes = loaded[stock_codes_key].tolist()
            trade_dates = pd.DatetimeIndex(loaded["dates"])
            feature_cols = loaded["feature_cols"].tolist()
            print(f'  期货数据已加载: {len(stock_codes)} 品种 × {len(trade_dates)} 日')
        except Exception:
            from tools import prepare_futures_data
            data, stock_codes, trade_dates, feature_cols = prepare_futures_data(save=True)
    else:
        stock_codes_key = "stock_codes"
        _data_npz = "data/stock_data_3d.npz"
        _mmap_name = "data/stock_data_mmap.dat"
        try:
            loaded = np.load(_data_npz, allow_pickle=True)
            data = loaded["data_3d"]
            stock_codes = loaded[stock_codes_key].tolist()
            trade_dates = pd.DatetimeIndex(loaded["dates"])
            feature_cols = loaded["feature_cols"].tolist()
        except:
            try:
                with open('API_key','r') as f:
                    api = f.readline()
            except:
                api = input('输入tushare_api=')
            data, stock_codes, trade_dates, feature_cols = prepare_data(STOCK_LIST=ZZ500_LIST, API=api)

    # ---- 2. 构建原语集，设定因子可能含有的变量与运算符 ----
    print('\n  开始构建原语集...')
    pset = build_pset(feature_cols)

    # ---- 多进程 Pool 初始化 ----
    # data 用 np.memmap 共享，避免每个 worker pickle 复制 60MB。
    os.makedirs("data", exist_ok=True)
    _mmap_path = os.path.abspath(_mmap_name)
    _need_rebuild = True
    if os.path.exists(_mmap_path) and os.path.getsize(_mmap_path) == data.nbytes:
        _need_rebuild = False
    if _need_rebuild:
        _mmap = np.memmap(_mmap_path, dtype=data.dtype, mode='w+', shape=data.shape)
        _mmap[:] = data[:]
        _mmap.flush()
        del _mmap

    # Worker 数量：
    # 可通过环境变量 GP_WORKERS 自行调整，例如: set GP_WORKERS=8
    _env_workers = os.environ.get("GP_WORKERS", "")
    if _env_workers and _env_workers.isdigit():
        safe_workers = max(1, min(int(_env_workers), N_CORES))
    else:
        safe_workers = min(USED_CORES, N_CORES)
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
        # 变异: 统一变异 (深度5)
    toolbox.register("expr_mut", genFullSafe, min_=1, max_=5)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        # 强变异: 更大子树替换 (趋同惩罚, 深度 HEAVY_MUTATION_DEPTH)
    toolbox.register("expr_heavy", genFullSafe, min_=2, max_=HEAVY_MUTATION_DEPTH)
    toolbox.register("heavy_mutate", gp.mutUniform, expr=toolbox.expr_heavy, pset=pset)
        # 限制树深度防止过拟合
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT + 3))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT + 3))
    toolbox.decorate("heavy_mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT + 3))

    # 注册多进程并行 map
    toolbox.register("map", pool.map)

    # ================================================================
    # 多轮独立 Run — 小种群快跑，合并 HoF 对抗趋同
    # ================================================================
    all_hof_individuals = []  # 跨 run 收集所有 HoF
    all_stats_logs = []       # 每轮进化日志

    for run_i in range(N_RUNS):
        run_seed = random.randint(0, 2**31 - 1)
        random.seed(run_seed)
        np.random.seed(run_seed % 2**32)  # numpy 种子同步, 避免随机性交叉污染
        print(f"\n{'='*60}")
        print(f"  Run {run_i+1}/{N_RUNS}  |  seed={run_seed}  |  POP={POPULATION_SIZE}  GEN={N_GENERATIONS}")
        print(f"{'='*60}")

        # ---- Step 4: 初始化种群（支持热启动） ----
        print('\n  创建初始种群...')
        hof = tools.HallOfFame(5)
        seed_individuals = []

        # 热启动: 从 warm_start.txt 读取用户自定义因子表达式
        if WARM_START and os.path.exists(WARM_START_FILE):
            print(f"\n[Warm Start] 从 {WARM_START_FILE} 加载因子...")
            count = 0
            with open(WARM_START_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释
                    if not line or line.startswith('#') or line.startswith('--'):
                        continue
                    if len(seed_individuals) >= POPULATION_SIZE:
                        break
                    # 支持 "名称 = 表达式" 格式, 也支持纯表达式
                    if '=' in line and not line.startswith('='):
                        expr_str = line.split('=', 1)[1].strip()
                    else:
                        expr_str = line
                    if not expr_str:
                        continue
                    try:
                        tree = gp.PrimitiveTree.from_string(expr_str, pset)
                        ind = creator.Individual(tree)
                        seed_individuals.append(ind)
                        count += 1
                    except Exception:
                        pass
            print(f"  Parsed {count} seed individuals for warm-start")

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

        # stats_log 在每轮 run 内部重新初始化
        stats_log = {"gen": [], "avg_fitness": [], "max_fitness": [],
                     "min_fitness": [], "avg_size": []}

        print(f"\n{'='*60}")
        print(f"  开始遗传编程进化")
        print(f"  种群: {POPULATION_SIZE} | 代数: {N_GENERATIONS} | "
                f"Pc={CROSSOVER_PROB} | Pm={MUTATION_PROB}")
        print(f"{'='*60}\n")

        for gen in range(N_GENERATIONS):
            t0 = time.time()

            # ---- 多样性检测 ----
            unique_exprs = len(set(str(ind) for ind in pop))
            unique_ratio = unique_exprs / len(pop)
            low_diversity = unique_ratio < DIVERSITY_THRESHOLD

            # ---- 选择 ----
            offspring = toolbox.select(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # ---- 交叉 ----
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_PROB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # ---- 强变异 (多样性低时加倍破坏, 跳出局部最优) ----
            heavy_prob = HEAVY_MUTATION_PROB * (2.0 if low_diversity else 1.0)
            for mutant in offspring:
                if random.random() < heavy_prob:
                    toolbox.heavy_mutate(mutant)
                    del mutant.fitness.values

            # ---- 普通变异 ----
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

            # ---- 随机移民 (替换最差个体, 多样性低时加倍) ----
            n_immigrants = max(1, int(POPULATION_SIZE * IMMIGRANT_RATE))
            if low_diversity:
                n_immigrants = max(2, int(POPULATION_SIZE * IMMIGRANT_RATE * 2.5))
            fresh = toolbox.population(n=n_immigrants)
            for ind in fresh:
                del ind.fitness.values
            pop[-n_immigrants:] = fresh
            # 评估新移民
            unfit_fresh = [ind for ind in fresh if not ind.fitness.valid]
            if unfit_fresh:
                fresh_fits = toolbox.map(toolbox.evaluate, unfit_fresh)
                for ind, (fit_val,) in zip(unfit_fresh, fresh_fits):
                    ind.fitness.values = (fit_val,)
            # 重新排序
            pop.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
            hof.update(pop)

            # ---- 日志 ----
            # 过滤 -999 无效标记 (DEAP 默认标记 fitness.valid=True, 需额外排除)
            valid_fits = [ind.fitness.values[0] for ind in pop
                          if ind.fitness.valid and ind.fitness.values[0] > -900]
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
            div_flag = "!" if low_diversity else " "
            print(f"Gen {gen:3d} | maxF={max_f:.4f} | avgF={avg_f:.4f} | "
                  f"sz={avg_sz:.0f} | uniq={unique_ratio:.0%}{div_flag} | "
                    f"HoF=[{hof_str}] | {elapsed:.1f}s")

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

        # 保存 per-run checkpoint
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
        with open(f"result\\gp_checkpoint_run{run_i+1}.pkl", "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"[Save] result/gp_checkpoint_run{run_i+1}.pkl")

        # 收集该 run 的结果 (过滤 -999 无效个体)
        for ind in hof:
            if ind.fitness.values[0] > -900:
                all_hof_individuals.append(ind)
        all_stats_logs.append(dict(stats_log))
        print(f"  Run {run_i+1} 完成: HoF top={hof[0].fitness.values[0]:.4f}")

    # ================================================================
    # 合并多 Run 结果
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  多 Run 合并 ({N_RUNS} runs × {POPULATION_SIZE} pop × {N_GENERATIONS} gen)")
    print(f"{'='*60}")

    # 去重后取 Top-N HoF
    seen = set()
    merged_hof = []
    all_hof_individuals.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
    for ind in all_hof_individuals:
        key = str(ind)
        if key not in seen:
            seen.add(key)
            merged_hof.append(ind)
    merged_hof = merged_hof[:10]
    print(f"  合并 HoF: {len(all_hof_individuals)} 个体 → {len(seen)} 唯一 → Top{len(merged_hof)}")

    # 保存合并 best_factors.txt
    with open("result\\best_factors.txt", "w", encoding="utf-8") as f:
        f.write(f"# GP Factor Discovery — Multi-Run ({N_RUNS} runs)\n")
        f.write(f"# Generations: {N_GENERATIONS} | Population: {POPULATION_SIZE} | Runs: {N_RUNS}\n")
        f.write(f"# FORWARD_RETURN: {FORWARD_RETURN}D | PARSIMONY_C: {PARSIMONY_C}\n\n")
        for i, ind in enumerate(merged_hof):
            f.write(f"Rank {i+1} | fitness={ind.fitness.values[0]:.6f} | "
                    f"nodes={len(ind)} | height={ind.height}\n")
            f.write(f"{str(ind)}\n\n")
    print("[Save] result/best_factors.txt (merged)")

    # 保存每轮进化日志合并
    combined_log = pd.concat(
        [pd.DataFrame(log) for log in all_stats_logs],
        keys=[f"run{i+1}" for i in range(len(all_stats_logs))],
        names=["run", "idx"]
    ).reset_index(level=0).reset_index(drop=True)
    combined_log.to_csv("result\\evolution_log.csv", index=False)
    print("[Save] result/evolution_log.csv")

    # 保存合并 checkpoint
    checkpoint_merged = {
        "merged_halloffame": merged_hof,
        "all_hof_individuals": all_hof_individuals,
        "all_stats_logs": all_stats_logs,
        "feature_cols": feature_cols,
        "stock_codes": stock_codes,
        "trade_dates": trade_dates,
        "n_runs": N_RUNS,
    }
    with open("result\\gp_checkpoint.pkl", "wb") as f:
        pickle.dump(checkpoint_merged, f)
    print("[Save] result/gp_checkpoint.pkl (merged)")

    # ================================================================
    # Step 7: 进化过程可视化 (基于最后一轮 + 合并 HoF)
    # ================================================================
    # ================================================================
    # ================================================================
    # Step 7: 进化过程可视化 (所有 Run 叠加对比)
    # ================================================================
    print("\n[Step 7] 进化过程可视化...")
    evo_dir = "result/evolution"
    os.makedirs(evo_dir, exist_ok=True)

    # 配色: 每轮一种颜色
    run_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    n_runs = len(all_stats_logs)

    # ---- 辅助函数 ----
    def _robust_ylim(all_series, clip_pct=5):
        """多轮数据合并计算 y 轴范围，裁剪极端值"""
        all_vals = np.concatenate([np.array(s) for s in all_series])
        valid = all_vals[all_vals > -100]
        if len(valid) < 3:
            return None, None
        lo = np.percentile(valid, clip_pct)
        hi = np.percentile(valid, 100 - clip_pct)
        margin = (hi - lo) * 0.1
        return lo - margin, hi + margin

    def _clean_min(min_arr):
        """将 min 中的 -999 替换为有效最小值"""
        mask = min_arr > -100
        return np.where(mask, min_arr, min_arr[mask].min()) if mask.any() else min_arr

    # ---- 图1: 适应度收敛曲线 (所有 Run 叠加) ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    all_avg = []
    for i, log in enumerate(all_stats_logs):
        g = log["gen"]
        c = run_colors[i % len(run_colors)]
        max_arr = np.array(log["max_fitness"])
        avg_arr = np.array(log["avg_fitness"])
        min_arr = _clean_min(np.array(log["min_fitness"]))
        all_avg.append(avg_arr)
        ax.plot(g, max_arr, color=c, lw=1.8, alpha=0.85, label=f"Run{i+1} max")
        ax.plot(g, avg_arr, color=c, ls="--", lw=1.2, alpha=0.6)
        ax.fill_between(g, min_arr, max_arr, alpha=0.05, color=c)
    ylo, yhi = _robust_ylim(all_avg)
    if ylo is not None and yhi is not None and yhi - ylo > 1e-12:
        ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (IC²)")
    ax.set_title(f"Fitness Convergence — All {n_runs} Runs")
    ax.legend(fontsize=7, ncol=min(n_runs, 3), frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{evo_dir}/fig1_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 图2: 树膨胀监控 (所有 Run 叠加) ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    all_sizes = []
    for i, log in enumerate(all_stats_logs):
        g = log["gen"]
        c = run_colors[i % len(run_colors)]
        size_arr = np.array(log["avg_size"])
        all_sizes.append(size_arr)
        ax.plot(g, size_arr, color=c, lw=1.5, alpha=0.8, label=f"Run{i+1}")
    p90 = np.percentile(np.concatenate(all_sizes), 90)
    ax.set_ylim(0, p90 * 1.15)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg Tree Size (nodes)")
    ax.set_title(f"Bloat Control — All {n_runs} Runs")
    ax.legend(fontsize=7, ncol=min(n_runs, 3), frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{evo_dir}/fig2_bloat.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 图3: 每 Run 独立收敛详情 (多子图) ----
    fig, axes = plt.subplots(n_runs, 1, figsize=(9, 3 * max(n_runs, 1)))
    if n_runs == 1:
        axes = [axes]
    for i, (log, ax) in enumerate(zip(all_stats_logs, axes)):
        g = log["gen"]
        c = run_colors[i % len(run_colors)]
        max_arr = np.array(log["max_fitness"])
        avg_arr = np.array(log["avg_fitness"])
        min_arr = _clean_min(np.array(log["min_fitness"]))
        ax.plot(g, max_arr, color=c, lw=1.8, label="Max")
        ax.plot(g, avg_arr, color=c, ls="--", lw=1.5, label="Avg")
        ax.plot(g, min_arr, color=c, ls=":", lw=1, alpha=0.5, label="Min")
        ax.fill_between(g, min_arr, max_arr, alpha=0.08, color=c)
        ylo, yhi = _robust_ylim([avg_arr])
        if ylo and yhi and yhi - ylo > 1e-12:
            ax.set_ylim(ylo, yhi)
        ax.set_title(f"Run {i+1}  |  maxF={max_arr[-1]:.4f}  avgF={avg_arr[-1]:.4f}  sz={log['avg_size'][-1]:.0f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{evo_dir}/fig3_per_run.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 图4: 合并 HoF — 多因子复杂度 vs 适应度 ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if all_hof_individuals:
        all_fits = [ind.fitness.values[0] for ind in all_hof_individuals]
        all_sizes = [len(ind) for ind in all_hof_individuals]
        ax.scatter(all_sizes, all_fits, c="C0", s=30, alpha=0.35,
                   edgecolors="none", label=f"All HoF ({len(all_hof_individuals)})")
    if merged_hof:
        hof_fits = [ind.fitness.values[0] for ind in merged_hof]
        hof_sizes = [len(ind) for ind in merged_hof]
        ax.scatter(hof_sizes, hof_fits, c="C3", s=100, marker="*",
                   edgecolors="black", linewidth=0.6, zorder=5,
                   label=f"Merged Top {len(merged_hof)}")
    ax.set_xlabel("Tree Size (nodes)")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Fitness–Complexity — Merged HoF ({N_RUNS} runs)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{evo_dir}/fig4_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [7a] {evo_dir}/fig1_convergence.png  — 全部 {n_runs} 轮收敛叠加")
    print(f"  [7b] {evo_dir}/fig2_bloat.png         — 全部 {n_runs} 轮膨胀叠加")
    print(f"  [7c] {evo_dir}/fig3_per_run.png        — 每轮独立收敛详情")
    print(f"  [7d] {evo_dir}/fig4_pareto.png         — 合并 HoF Pareto 前沿")
    print(f"  [7e] 合并 HoF top={merged_hof[0].fitness.values[0]:.4f}  "
          f"({len(merged_hof)} 个唯一因子)")

    # ---- 清理多进程资源 ----
    pool.close()
    pool.join()

    # 编译缓存统计
    total_compile = _HIT + _MISS
    if total_compile > 0:
        print(f"[缓存] 表达式编译: {_HIT} hits / {_MISS} misses "
              f"(命中率 {_HIT/total_compile*100:.1f}%, 缓存 {len(_compile_cache)} 个)")

    return merged_hof, all_stats_logs


if __name__ == '__main__':
    main()