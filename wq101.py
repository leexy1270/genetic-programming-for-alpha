"""
wq101.py — WorldQuant 101 Formulaic Alpha Expressions (DEAP-compatible)

Stores factors as expression strings parsable into gp.PrimitiveTree via parse_expr().
Based on "101 Formulaic Alphas" by Zura Kakushadze (WorldQuant, 2015).

ARG mapping:
    ARG0=OPEN, ARG1=HIGH, ARG2=LOW, ARG3=CLOSE, ARG4=RETURN,
    ARG5=VOLUME, ARG6=VWAP

WQ → DEAP translations:
    rank(x)        → CS_RANK(x)          ts_rank(x,d)    → TSRANK(x,d)
    delta(x,d)     → DELTA(x,d)          delay(x,d)      → DELAY(x,d)
    correlation(x,y,d)→ CORR(x,y,d)      covariance(x,y,d)→ COVIANCE(x,y,d)
    sum(x,d)       → SUM(x,d)            product(x,d)     → PROD(x,d)
    stddev(x,d)    → STD(x,d)            ts_min(x,d)      → TSMIN(x,d)
    ts_max(x,d)    → TSMAX(x,d)          ts_argmax(x,d)   → TSARGMAX(x,d)
    ts_argmin(x,d) → TSARGMIN(x,d)       decay_linear(x,d)→ DECAYLINEAR(x,d)
    signedpower(x,a)→ SIGNED_POWER(x,a)  scale(x,a=1)     → DIV(x,SUM(ABS(x),a))
    indneutralize  → NOT AVAILABLE       regbeta(x,y,d)   → REGBETA(x,y,d)
    −1*            → NEG(...)            ?a:b             → IF_POS(...)
    ^2             → SQUARE(x)           √                → SQRT(x)
    log            → LOG(x)              abs              → ABS(x)
    sign           → SIGN(x)

    returns         = ARG4 (daily return)
    adv{d}          = MEAN(ARG5, d) (average daily volume)
    vwap            = ARG6
    close, open, high, low, volume = ARG3, ARG0, ARG1, ARG2, ARG5

Usage:
    from wq101 import WQ101_EXPRS, load_factors
    from build_pset import build_pset
    pset = build_pset(feature_cols)
    trees = load_factors(pset)  # returns dict {name: PrimitiveTree or None}
"""

import re
import numpy as np
from deap import gp
from gtja191 import _build_name_map, _find_value_terminal, parse_expr, load_factors


# ============================================================
# WorldQuant 101 Alpha Expressions (DEAP syntax)
# ============================================================

WQ101_EXPRS = {
    # ===== Alpha 001-010 =====
    'wq001': "SUB(CS_RANK(TSARGMAX(SIGNED_POWER(IF_POS(NEG(ARG4),STD(ARG4,20),ARG3),2),5)),0.5)",
    'wq002': "NEG(CORR(CS_RANK(DELTA(LOG(ARG5),2)),CS_RANK(DIV(SUB(ARG3,ARG0),ARG0)),6))",
    'wq003': "NEG(CORR(CS_RANK(ARG0),CS_RANK(ARG5),10))",
    'wq004': "NEG(TSRANK(CS_RANK(ARG2),9))",
    'wq005': "MUL(CS_RANK(SUB(ARG0,DIV(SUM(ARG6,10),10))),NEG(ABS(CS_RANK(SUB(ARG3,ARG6)))))",
    # wq006: signedpower(correlation(open, volume, 10), 2) with sign
    'wq006': "NEG(CORR(ARG0,ARG5,10))",
    'wq007': None,  # adv20 < volume ? ... : ... (ternary with comparison, now expressible with GT/LT)
    'wq008': "NEG(CS_RANK(ADD(DELTA(ADD(MUL(ARG0,0.618),MUL(ARG6,0.618)),4),MUL(-1.0,TSRANK(CS_RANK(ARG5),10)))))",
    'wq009': None,  # scale + ts_min/ts_max/ts_argmin/ts_argmax complex
    'wq010': "CS_RANK(IF_POS(SUB(TSMAX(ARG4,4),TSMIN(ARG4,4)),SUB(TSMAX(ARG4,4),TSMIN(ARG4,4)),ADD(ARG4,1.0)))",

    # ===== Alpha 011-020 =====
    'wq011': "MUL(SUB(CS_RANK(SUB(TSMAX(ARG6,3),TSMIN(ARG6,3))),0.5),SUB(CS_RANK(SUB(TSMAX(ARG4,3),TSMIN(ARG4,3))),0.5))",
    'wq012': "SUB(CS_RANK(DIV(SUM(ARG5,10),10)),CS_RANK(DIV(ADD(ARG3,ARG0),2)))",
    'wq013': None,  # scale + signedpower
    'wq014': "NEG(CS_RANK(DELTA(ARG4,3)))",
    'wq015': "NEG(ADD(CS_RANK(CORR(CS_RANK(ARG1),CS_RANK(MEAN(ARG5,6)),6)),CS_RANK(CORR(CS_RANK(ARG6),CS_RANK(MEAN(ARG5,6)),6))))",
    'wq016': "NEG(CS_RANK(COVIANCE(CS_RANK(ARG1),CS_RANK(ARG5),5)))",
    'wq017': None,  # ts_rank(adv20, 20) comparison
    'wq018': "NEG(DIV(CS_RANK(STD(ABS(SUB(ARG3,ARG0)),5)),CS_RANK(ADD(SUB(ARG3,ARG0),ARG3))))",
    'wq019': "NEG(SIGN(DIV(SUB(ARG3,DELAY(ARG3,7)),DELAY(ARG3,7))))",
    'wq020': None,  # signedpower + scale

    # ===== Alpha 021-030 =====
    'wq021': None,  # regression + sum
    'wq022': "NEG(DIV(DELTA(CORR(ARG1,ARG5,5),5),STD(ARG3,20)))",
    'wq023': "MUL(DIV(SUM(IF_POS(SUB(ARG3,DELAY(ARG3,1)),SUB(ARG3,MIN2(ARG2,DELAY(ARG3,1))),SUB(ARG3,MAX2(ARG1,DELAY(ARG3,1)))),20),MEAN(ARG5,20)),100.0)",
    'wq024': "MUL(DIV(SUB(SUM(IF_POS(SUB(DELAY(ARG3,1),ARG3),STD(ARG3,20),0.0),20),SUM(IF_POS(SUB(ARG3,DELAY(ARG3,1)),STD(ARG3,20),0.0),20)),SUM(IF_POS(SUB(ARG3,DELAY(ARG3,1)),STD(ARG3,20),0.0),20)),100.0)",
    'wq025': "NEG(CS_RANK(MUL(DELTA(ARG3,7),SUB(1.0,CS_RANK(DECAYLINEAR(DIV(ARG5,MEAN(ARG5,20)),9))))))",
    'wq026': "ADD(SUB(DIV(SUM(ARG3,7),7),ARG3),CORR(ARG6,DELAY(ARG3,5),230))",
    'wq027': "ADD(NEG(CS_RANK(MUL(SUM(CORR(CS_RANK(ARG5),CS_RANK(ARG6),6),2),SUB(1.0,CS_RANK(1.0))))),1.0)",
    'wq028': None,  # scale + log + decay_linear
    'wq029': None,  # rank of product of (min and ts_rank)
    'wq030': None,  # sign of delta + scale

    # ===== Alpha 031-040 =====
    'wq031': "NEG(SUM(CS_RANK(CORR(CS_RANK(ARG3),CS_RANK(MEAN(ARG5,6)),6)),3))",
    'wq032': "NEG(DIV(CS_RANK(SUB(SUM(ARG3,7),ARG3)),CS_RANK(ADD(SUM(ARG3,7),ARG3))))",
    'wq033': "NEG(DIV(CS_RANK(SUB(DIV(SUM(ARG0,5),5),DIV(SUM(ARG3,5),5))),CS_RANK(ADD(DIV(SUM(ARG0,5),5),DIV(SUM(ARG3,5),5)))))",
    'wq034': None,  # rank(returns/rank(...)) style
    'wq035': None,  # scale(rank(ts_min)) style
    'wq036': "ADD(NEG(CS_RANK(SUM(CORR(CS_RANK(ARG5),CS_RANK(ARG6),6),2))),1.0)",
    'wq037': None,  # rank of correlation cascade
    'wq038': None,  # rank of decay_linear of high
    'wq039': None,  # scale + ts_rank
    'wq040': "MUL(DIV(SUM(IF_POS(SUB(ARG3,DELAY(ARG3,1)),ARG5,0.0),26),SUM(IF_POS(SUB(DELAY(ARG3,1),ARG3),ARG5,0.0),26)),100.0)",

    # ===== Alpha 041-050 =====
    'wq041': "NEG(CS_RANK(TSMAX(DELTA(ARG6,3),5)))",
    'wq042': "MUL(NEG(CS_RANK(STD(ARG1,10))),CORR(ARG1,ARG5,10))",
    'wq043': "SUM(IF_POS(SUB(ARG3,DELAY(ARG3,1)),ARG5,IF_POS(SUB(DELAY(ARG3,1),ARG3),NEG(ARG5),0.0)),6)",
    'wq044': "ADD(TSRANK(DECAYLINEAR(CORR(ARG2,MEAN(ARG5,10),7),6),4),TSRANK(DECAYLINEAR(DELTA(ARG6,3),10),15))",
    'wq045': "MUL(CS_RANK(DELTA(ADD(MUL(ARG3,0.618),MUL(ARG0,0.618)),1)),CS_RANK(CORR(ARG6,MEAN(ARG5,150),15)))",
    'wq046': "DIV(ADD(ADD(MEAN(ARG3,3),MEAN(ARG3,6)),ADD(MEAN(ARG3,12),MEAN(ARG3,24))),MUL(4,ARG3))",
    'wq047': "SMA(DIV(SUB(TSMAX(ARG1,6),ARG3),SUB(TSMAX(ARG1,6),TSMIN(ARG2,6))),9,1)",
    'wq048': None,  # correlation + signedpower
    'wq049': None,  # DTM/DBM (now expressible with GT/LT)
    'wq050': None,  # DTM/DBM (now expressible)

    # ===== Alpha 051-060 =====
    'wq051': None,  # DTM/DBM (now expressible)
    'wq052': "NEG(ADD(CS_RANK(DECAYLINEAR(DELTA(ARG6,5),3)),CS_RANK(DECAYLINEAR(NEG(DELTA(ADD(MUL(ARG0,0.618),MUL(ARG2,0.618)),2)),4))))",
    'wq053': "MUL(DIV(SUM(GT(ARG3,DELAY(ARG3,1)),12),12),100.0)",  # COUNT → SUM+GT
    'wq054': "NEG(MUL(CS_RANK(ADD(STD(ABS(SUB(ARG3,ARG0)),10),ADD(SUB(ARG3,ARG0),CORR(ARG3,ARG0,10)))),1.0))",
    'wq055': None,  # complex ternary
    'wq056': None,  # rank of (sum + correlation)
    'wq057': "SMA(DIV(SUB(ARG3,TSMIN(ARG2,9)),SUB(TSMAX(ARG1,9),TSMIN(ARG2,9))),3,1)",
    'wq058': "MUL(DIV(SUM(GT(ARG3,DELAY(ARG3,1)),20),20),100.0)",  # COUNT → SUM+GT
    'wq059': None,  # complex ternary
    'wq060': "NEG(MUL(MUL(DIV(SUM(SUB(ARG1,ARG2),20),SUM(SUB(ARG0,ARG3),20)),100.0),1.0))",

    # ===== Alpha 061-080 =====
    'wq061': "NEG(MAX2(CS_RANK(DECAYLINEAR(DELTA(ARG6,1),12)),CS_RANK(DECAYLINEAR(CORR(CS_RANK(ARG2),CS_RANK(MEAN(ARG5,80)),8),17))))",
    'wq062': "NEG(CORR(ARG1,CS_RANK(ARG5),5))",
    'wq063': None,  # signedpower + ts_rank
    'wq064': "NEG(CS_RANK(MAX2(DECAYLINEAR(CORR(CS_RANK(ARG6),CS_RANK(ARG5),3),4),DECAYLINEAR(TSMAX(CORR(CS_RANK(ARG2),CS_RANK(MEAN(ARG5,60)),4),13),14))))",
    'wq065': "NEG(SUB(CS_RANK(DELTA(ARG3,7)),CS_RANK(ADD(SUB(DIV(SUM(ARG5,20),20),ARG5),MUL(2,ARG5)))))",
    'wq066': None,  # rank + ts_argmax/min
    'wq067': None,  # SMA of complex
    'wq068': None,  # rank of ts_rank
    'wq069': None,  # DTM/DBM (now expressible)
    'wq070': "STD(MUL(ARG5,ARG6),6)",
    'wq071': "MUL(DIV(SUB(ARG3,MEAN(ARG3,24)),MEAN(ARG3,24)),100.0)",
    'wq072': "SMA(DIV(SUB(TSMAX(ARG1,6),ARG3),SUB(TSMAX(ARG1,6),TSMIN(ARG2,6))),15,1)",
    'wq073': None,  # regression + rank
    'wq074': None,  # complex nested RANK+CORR
    'wq075': None,  # benchmark index
    'wq076': None,  # complex nested
    'wq077': "MIN2(CS_RANK(DECAYLINEAR(SUB(ADD(DIV(ADD(ARG1,ARG2),2),ARG1),ADD(ARG6,ARG1)),20)),CS_RANK(DECAYLINEAR(CORR(DIV(ADD(ARG1,ARG2),2),MEAN(ARG5,40),3),6)))",
    'wq078': None,  # complex nested
    'wq079': None,  # SMA of complex
    'wq080': "MUL(DIV(SUB(ARG5,DELAY(ARG5,5)),DELAY(ARG5,5)),100.0)",

    # ===== Alpha 081-100 =====
    'wq081': "SMA(ARG5,21,2)",
    'wq082': "SMA(DIV(SUB(TSMAX(ARG1,6),ARG3),SUB(TSMAX(ARG1,6),TSMIN(ARG2,6))),20,1)",
    'wq083': "NEG(CS_RANK(COVIANCE(CS_RANK(ARG1),CS_RANK(ARG5),5)))",
    'wq084': "SUM(IF_POS(SUB(ARG3,DELAY(ARG3,1)),ARG5,IF_POS(SUB(DELAY(ARG3,1),ARG3),NEG(ARG5),0.0)),20)",
    'wq085': None,  # scale
    'wq086': None,  # ternary with comparison
    'wq087': None,  # complex nested
    'wq088': "MUL(DIV(SUB(ARG3,DELAY(ARG3,20)),DELAY(ARG3,20)),100.0)",
    'wq089': None,  # SMA multi-layer
    'wq090': "NEG(CS_RANK(CORR(CS_RANK(ARG6),CS_RANK(ARG5),5)))",
    'wq091': None,  # complex nested
    'wq092': None,  # rank of correlation of rank
    'wq093': None,  # complex ternary
    'wq094': "SUM(IF_POS(SUB(ARG3,DELAY(ARG3,1)),ARG5,IF_POS(SUB(DELAY(ARG3,1),ARG3),NEG(ARG5),0.0)),30)",
    'wq095': "STD(MUL(ARG5,ARG6),20)",
    'wq096': "SMA(SMA(DIV(SUB(ARG3,TSMIN(ARG2,9)),SUB(TSMAX(ARG1,9),TSMIN(ARG2,9))),3,1),3,1)",
    'wq097': "NEG(CS_RANK(STD(ARG5,10)))",
    'wq098': None,  # comparison + ternary
    'wq099': "NEG(CS_RANK(COVIANCE(CS_RANK(ARG3),CS_RANK(ARG5),5)))",
    'wq100': "STD(ARG5,20)",

    # ===== Alpha 101 =====
    'wq101': "NEG(CS_RANK(DIV(SUB(ARG3,ARG0),ADD(DIV(SUM(ARG0,5),5),DIV(SUM(ARG3,5),5)))))",
}


# ============================================================
# Statistics
# ============================================================

def stats():
    """Print summary: how many WQ101 factors are expressible."""
    total = len(WQ101_EXPRS)
    ok = sum(1 for v in WQ101_EXPRS.values() if v is not None)
    print(f"WQ101: {ok}/{total} expressible as DEAP GP trees ({ok*100//total}%)")
    return ok, total


def load_wq_factors(pset, skip_none=True):
    """Parse all WQ101 factor expressions into PrimitiveTree dict."""
    result = {}
    for name, expr_str in WQ101_EXPRS.items():
        if expr_str is None:
            if not skip_none:
                result[name] = None
            continue
        try:
            result[name] = parse_expr(expr_str, pset)
        except Exception:
            if not skip_none:
                result[name] = None
    return result


if __name__ == '__main__':
    stats()
