from parameters import *
from tools import *
from build_pset import build_pset, genHalfAndHalfSafe,genFullSafe,genGrowSafe
from deap import creator,base,tools,gp
from evaluate import evaluate

import operator
import os
import time
import random
from functools import partial

import pandas as pd
import numpy as np

def main():

    try:
        loaded = np.load("data\\stock_data_3d.npz", allow_pickle=True)
        data = loaded["data_3d"]
        stock_codes = loaded["stock_codes"].tolist()
        trade_dates = pd.DatetimeIndex(loaded["dates"])
        feature_cols = loaded["feature_cols"].tolist()
    except:
        data,stock_codes,trade_dates,feature_cols = prepare_data(STOCK_LIST=ZZ500_LIST)

    # ---- 2. 构建原语集，设定因子可能含有的变量与运算符 ----
    pset = build_pset(feature_cols)

    # ---- 3. 创建进化过程中所需的类型 ----
    # FitnessMax + PrimitiveTree ——> Individual
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except RuntimeError:
        pass  # already created (e.g. during interactive re-run)
    try:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    except RuntimeError:
        pass

    # pset ——>     expr     ——> individual ——> popluation
    #          + Individual
    toolbox = base.Toolbox()
    toolbox.register("expr", genHalfAndHalfSafe, pset=pset, min_=1, max_=MAX_TREE_HEIGHT)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #评估
    toolbox.register("evaluate", partial(evaluate, data=data, pset=pset, feature_cols=feature_cols))

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

    # ---- Step 4: 初始化种群（支持热启动） ----
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

    # 构建种群：优先使用热启动种子，不足的用随机个体补充
    pop = toolbox.population(n=POPULATION_SIZE)
    if seed_individuals:
        n_seeds = min(len(seed_individuals), POPULATION_SIZE)
        pop[:n_seeds] = seed_individuals[:n_seeds]
        # 剩余的随机个体
        if n_seeds < POPULATION_SIZE:
            random_fill = toolbox.population(n=POPULATION_SIZE - n_seeds)
            pop[n_seeds:] = random_fill

    # 评估初始种群
    print("  评估初始种群...")
    for ind in pop:
        if not ind.fitness.valid:
            (fit_val,) = toolbox.evaluate(ind)
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

        # ---- 评估新个体 ----
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            (fit_val,) = toolbox.evaluate(ind)  
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

    # ---- Step 7: 可视化和因子分析 ----
    try:
        from visualization import plot_evolution_report
        plot_evolution_report(stats_log, hof, pset, data, feature_cols, trade_dates, stock_codes)
    except Exception as e:
        print(f"[Visualization] Skipped: {e}")

    try:
        from factor_analysis import analyze_hof_factors
        analyze_hof_factors(hof, pset, data, feature_cols, trade_dates, stock_codes)
    except Exception as e:
        print(f"[Analysis] Skipped: {e}")

    return pop, hof, stats_log

if __name__ == '__main__':
    main()

