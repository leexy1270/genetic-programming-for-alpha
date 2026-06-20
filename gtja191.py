"""
gtja191.py — GTJA 191 Alpha Factor Expressions (DEAP-compatible)

Stores factors as expression strings parsable into gp.PrimitiveTree via parse_expr().

ARG mapping:
    ARG0=OPEN, ARG1=HIGH, ARG2=LOW, ARG3=CLOSE, ARG4=RETURN,
    ARG5=VOLUME, ARG6=VWAP

Available primitives:
    Unary: ABS, SQRT, LOG, SIGN, SQUARE, TANH, NEG, CS_RANK, CS_ZSCORE, POS, NEGVAL, SUMAC
    Binary: ADD, SUB, MUL, DIV, MAX2, MIN2, GT, LT, GE, LE, EQ, NEQ, CROSS
    TS (arr,int): DELTA, DELAY, SUM, MEAN, STD, TSMIN, TSMAX, TSRANK, ROC, ZSCORE, DECAYLINEAR, POWER, TSARGMAX, TSARGMIN, PROD, WMA, SIGNED_POWER
    TS (arr,int,int): SMA
    Ternary (arr,arr,int): CORR, COVIANCE, REGBETA, REGRESI
    Ternary (arr,arr,arr): IF_POS
    Int→Arr: SEQUENCE
    Tech: DTM(open,high), DBM(open,low), TR(high,low,close), HD(high), LD(low)
    Self: SELF(arr)

GTJA → DEAP translations:
    TSRANK(x)→CS_RANK(x)    TSTSRANK(x,d)→TSRANK(x,d)
    TSMAX→MAX  TSMIN→MIN  COVIANCE→COV  DECAYLINEAR→DECAY
    X>Y?a:b→IF_POS(SUB(X,Y),a,b)    X<Y?a:b→IF_POS(SUB(Y,X),a,b)
    X^2→SQUARE(X)  X^Y→POWER(X,Y)  -X→NEG(X)  MAX/MIN elem→MAX2/MIN2
    AMOUNT≈MUL(ARG5,ARG6)  1→1.0(terminal)  -1→-1.0(terminal)

Factors with None = cannot express (SELF, benchmark, REGBETA, SEQUENCE, etc.)

Usage:
    from gtja191 import FACTOR_EXPRS, load_factors
    from build_pset import build_pset
    pset = build_pset(feature_cols)
    trees = load_factors(pset)  # returns dict {name: PrimitiveTree or None}
"""

import re
import numpy as np
from deap import gp


def _build_name_map(pset):
    """Build {name: primitive_or_terminal} mapping from pset."""
    nm = {}
    for tl in pset.terminals.values():
        for t in tl:
            nm[t.name] = t
    for pl in pset.primitives.values():
        for p in pl:
            nm[p.name] = p
    return nm


def _find_value_terminal(val, pset):
    """Find a terminal in pset that has a matching .value attribute."""
    target_type = type(val)
    for tl in pset.terminals.values():
        for t in tl:
            if hasattr(t, 'value') and type(t.value) == target_type:
                if abs(t.value - val) < 1e-12:
                    return t
    return None


def parse_expr(expr_str, pset):
    """Parse DEAP expression string → gp.PrimitiveTree (works with PrimitiveSetTyped)."""
    name_map = _build_name_map(pset)

    def _split_args(s):
        args = []
        depth = 0
        cur = []
        for ch in s:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if ch == ',' and depth == 0:
                args.append(''.join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        if cur:
            args.append(''.join(cur).strip())
        return args

    def _parse(s):
        s = s.strip()
        if not s:
            raise ValueError("empty expression")
        # function call: NAME(...)
        if '(' in s and s.endswith(')'):
            idx = s.index('(')
            name = s[:idx].strip()
            args_str = s[idx + 1:-1]
            if name in name_map:
                parsed = [_parse(a) for a in _split_args(args_str)]
                result = [name_map[name]]
                for pa in parsed:
                    result.extend(pa)
                return result
        # Terminal by name
        if s in name_map:
            return [name_map[s]]
        # Numeric literal
        try:
            if '.' in s or 'e' in s.lower():
                val = float(s)
            else:
                val = int(s)
            term = _find_value_terminal(val, pset)
            if term is not None:
                return [term]
        except (ValueError, KeyError):
            pass
        raise ValueError(f"Cannot parse terminal: {s}")

    return gp.PrimitiveTree(_parse(expr_str))


def load_factors(pset, skip_none=True):
    """Parse all factor expressions into PrimitiveTree dict.
    Returns {name: PrimitiveTree} — failed factors become None unless skip_none=True."""
    result = {}
    for name, expr_str in FACTOR_EXPRS.items():
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


# ============================================================
# GTJA 191 Factor Expressions (DEAP syntax)
# ============================================================

FACTOR_EXPRS = {
    # ===== Alpha 001-020 =====
    'alpha001': "MUL(-1.0, CORR(CS_RANK(DELTA(LOG(ARG5), 1)), CS_RANK(DIV(SUB(ARG3, ARG0), ARG0)), 6))",
    'alpha002': "NEG(DELTA(DIV(SUB(SUB(ARG3, ARG2), SUB(ARG1, ARG3)), SUB(ARG1, ARG2)), 1))",
    # Alpha3: ternary with close==delay[close], can't express equality; approximate
    'alpha003': "SUM(IF_POS(SUB(ARG3, DELAY(ARG3, 1)), SUB(ARG3, MIN2(ARG2, DELAY(ARG3, 1))), SUB(ARG3, MAX2(ARG1, DELAY(ARG3, 1)))), 6)",
    # Alpha4: complex ternary; approximate with nested IF_POS
    'alpha004': None,  # too complex to express
    'alpha005': "NEG(TSMAX(CORR(TSRANK(ARG5, 5), TSRANK(ARG1, 5), 5), 3))",
    'alpha006': "MUL(CS_RANK(SIGN(DELTA(ADD(MUL(ARG0, 0.618), MUL(ARG1, 0.618)), 4))), -1.0)",
    # Alpha6: 0.85 and 0.15 not float terminals; use 0.618*1.37 approx. Actually use the formula directly
    'alpha006': "MUL(CS_RANK(SIGN(DELTA(ADD(MUL(ARG0, 0.618), MUL(ARG1, 0.618)), 4))), -1.0)",  # approximate: 0.85≈0.618
    'alpha007': "MUL(ADD(CS_RANK(TSMAX(SUB(ARG6, ARG3), 3)), CS_RANK(TSMIN(SUB(ARG6, ARG3), 3))), CS_RANK(DELTA(ARG5, 3)))",
    'alpha008': "CS_RANK(NEG(DELTA(ADD(MUL(DIV(ADD(ARG1, ARG2), 2), 0.618), MUL(ARG6, 0.618)), 4)))",
    'alpha009': "SMA(MUL(SUB(DIV(ADD(ARG1, ARG2), 2), DIV(ADD(DELAY(ARG1, 1), DELAY(ARG2, 1)), 2)), DIV(SUB(ARG1, ARG2), ARG5)), 7, 2)",
    'alpha010': "CS_RANK(TSMAX(IF_POS(NEG(ARG4), SQUARE(STD(ARG4, 20)), SQUARE(ARG3)), 5))",
    'alpha011': "SUM(MUL(DIV(SUB(SUB(ARG3, ARG2), SUB(ARG1, ARG3)), SUB(ARG1, ARG2)), ARG5), 6)",
    'alpha012': "MUL(CS_RANK(SUB(ARG0, DIV(SUM(ARG6, 10), 10))), NEG(CS_RANK(ABS(SUB(ARG3, ARG6)))))",
    'alpha013': "SUB(SQRT(MUL(ARG1, ARG2)), ARG6)",
    'alpha014': "SUB(ARG3, DELAY(ARG3, 5))",
    'alpha015': "SUB(DIV(ARG0, DELAY(ARG3, 1)), 1.0)",
    'alpha016': "NEG(TSMAX(CS_RANK(CORR(CS_RANK(ARG5), CS_RANK(ARG6), 5)), 5))",
    'alpha017': "POWER(CS_RANK(SUB(ARG6, TSMAX(ARG6, 15))), DELTA(ARG3, 5))",
    'alpha018': "DIV(ARG3, DELAY(ARG3, 5))",
    'alpha019': None,  # SELF recursive (SELF primitive now available, exact formula TBD)
    'alpha020': "MUL(DIV(SUB(ARG3, DELAY(ARG3, 6)), DELAY(ARG3, 6)), 100.0)",

    # ===== Alpha 021-040 =====
    'alpha021': "REGBETA(MEAN(ARG3, 6), SEQUENCE(6))",
    'alpha022': "SMA(SUB(DIV(SUB(ARG3, MEAN(ARG3, 6)), MEAN(ARG3, 6)), DELAY(DIV(SUB(ARG3, MEAN(ARG3, 6)), MEAN(ARG3, 6)), 3)), 12, 1)",
    'alpha023': "MUL(DIV(SMA(IF_POS(SUB(ARG3, DELAY(ARG3, 1)), STD(ARG3, 20), 0.0), 20, 1), ADD(SMA(IF_POS(SUB(ARG3, DELAY(ARG3, 1)), STD(ARG3, 20), 0.0), 20, 1), SMA(IF_POS(SUB(DELAY(ARG3, 1), ARG3), STD(ARG3, 20), 0.0), 20, 1))), 100.0)",
    'alpha024': "SMA(SUB(ARG3, DELAY(ARG3, 5)), 5, 1)",
    'alpha025': "MUL(NEG(CS_RANK(MUL(DELTA(ARG3, 7), SUB(1.0, CS_RANK(DECAYLINEAR(DIV(ARG5, MEAN(ARG5, 20)), 9)))))), ADD(1.0, CS_RANK(SUM(ARG4, 250))))",
    'alpha026': "ADD(SUB(DIV(SUM(ARG3, 7), 7), ARG3), CORR(ARG6, DELAY(ARG3, 5), 230))",
    'alpha027': "WMA(ADD(MUL(DIV(SUB(ARG3, DELAY(ARG3, 3)), DELAY(ARG3, 3)), 100.0), MUL(DIV(SUB(ARG3, DELAY(ARG3, 6)), DELAY(ARG3, 6)), 100.0)), 12)",
    'alpha028': "SUB(MUL(3, SMA(DIV(SUB(ARG3, TSMIN(ARG2, 9)), SUB(TSMAX(ARG1, 9), TSMIN(ARG2, 9))), 3, 1)), MUL(2, SMA(SMA(DIV(SUB(ARG3, TSMIN(ARG2, 9)), SUB(TSMAX(ARG1, 9), TSMAX(ARG2, 9))), 3, 1), 3, 1)))",
    'alpha029': "MUL(DIV(SUB(ARG3, DELAY(ARG3, 6)), DELAY(ARG3, 6)), ARG5)",
    'alpha030': None,  # REGRESI + WMA available, but needs benchmark index (MKT)
    'alpha031': "MUL(DIV(SUB(ARG3, MEAN(ARG3, 12)), MEAN(ARG3, 12)), 100.0)",
    'alpha032': "NEG(SUM(CS_RANK(CORR(CS_RANK(ARG1), CS_RANK(ARG5), 3)), 3))",
    'alpha033': None,  # complex nested
    'alpha034': "DIV(MEAN(ARG3, 12), ARG3)",
    'alpha035': "NEG(MIN2(CS_RANK(DECAYLINEAR(DELTA(ARG0, 1), 15)), CS_RANK(DECAYLINEAR(CORR(ARG5, ADD(MUL(ARG0, 0.618), SQUARE(ARG0)), 17), 7))))",
    'alpha036': "CS_RANK(SUM(CORR(CS_RANK(ARG5), CS_RANK(ARG6)), 6), 2)",
    'alpha037': "NEG(CS_RANK(SUB(MUL(SUM(ARG0, 5), SUM(ARG4, 5)), DELAY(MUL(SUM(ARG0, 5), SUM(ARG4, 5)), 10))))",
    'alpha038': "IF_POS(SUB(ARG1, DIV(SUM(ARG1, 20), 20)), NEG(DELTA(ARG1, 2)), 0.0)",
    'alpha039': "NEG(SUB(CS_RANK(DECAYLINEAR(DELTA(ARG3, 2), 8)), CS_RANK(DECAYLINEAR(CORR(ADD(MUL(ARG6, 0.618), MUL(ARG0, 0.618)), SUM(MEAN(ARG5, 180), 37), 14), 12))))",
    'alpha040': "MUL(DIV(SUM(IF_POS(SUB(ARG3, DELAY(ARG3, 1)), ARG5, 0.0), 26), SUM(IF_POS(SUB(DELAY(ARG3, 1), ARG3), ARG5, 0.0), 26)), 100.0)",

    # ===== Alpha 041-060 =====
    'alpha041': "NEG(CS_RANK(TSMAX(DELTA(ARG6, 3), 5)))",
    'alpha042': "MUL(NEG(CS_RANK(STD(ARG1, 10))), CORR(ARG1, ARG5, 10))",
    'alpha043': "SUM(IF_POS(SUB(ARG3, DELAY(ARG3, 1)), ARG5, IF_POS(SUB(DELAY(ARG3, 1), ARG3), NEG(ARG5), 0.0)), 6)",
    'alpha044': "ADD(TSRANK(DECAYLINEAR(CORR(ARG2, MEAN(ARG5, 10), 7), 6), 4), TSRANK(DECAYLINEAR(DELTA(ARG6, 3), 10), 15))",
    'alpha045': "MUL(CS_RANK(DELTA(ADD(MUL(ARG3, 0.618), MUL(ARG0, 0.618)), 1)), CS_RANK(CORR(ARG6, MEAN(ARG5, 150), 15)))",
    'alpha046': "DIV(ADD(ADD(MEAN(ARG3, 3), MEAN(ARG3, 6)), ADD(MEAN(ARG3, 12), MEAN(ARG3, 24))), MUL(4, ARG3))",
    'alpha047': "SMA(DIV(SUB(TSMAX(ARG1, 6), ARG3), SUB(TSMAX(ARG1, 6), TSMIN(ARG2, 6))), 9, 1)",
    'alpha048': None,  # complex SIGN cascades (theoretically expressible but extremely deep tree)
    # alpha049: down-gap strength = SUM(down_gap,12) / (SUM(down_gap,12)+SUM(up_gap,12))
    #   where gap = MAX(ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1)))
    #   down_gap triggered when (HIGH+LOW) < DELAY(HIGH+LOW,1)
    'alpha049': "DIV(SUM(MUL(LT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12),ADD(SUM(MUL(LT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12),SUM(MUL(GT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12)))",
    # alpha050: net direction = up-gap ratio - down-gap ratio (alpha051 - alpha049)
    'alpha050': "SUB(DIV(SUM(MUL(GT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12),ADD(SUM(MUL(LT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12),SUM(MUL(GT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12))),DIV(SUM(MUL(LT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12),ADD(SUM(MUL(LT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12),SUM(MUL(GT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12))))",
    # alpha051: up-gap strength = SUM(up_gap,12) / (SUM(down_gap,12)+SUM(up_gap,12))
    'alpha051': "DIV(SUM(MUL(GT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12),ADD(SUM(MUL(LT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12),SUM(MUL(GT(ADD(ARG1,ARG2),DELAY(ADD(ARG1,ARG2),1)),MAX2(ABS(SUB(ARG1,DELAY(ARG1,1))),ABS(SUB(ARG2,DELAY(ARG2,1))))),12)))",
    'alpha052': None,  # complex ternary + special (theoretically expressible with IF_POS + GT/LT)
    'alpha053': "MUL(DIV(SUM(GT(ARG3, DELAY(ARG3, 1)), 12), 12), 100.0)",  # COUNT → SUM+GT
    'alpha054': "NEG(CS_RANK(ADD(ADD(STD(ABS(SUB(ARG3, ARG0)), 10), SUB(ARG3, ARG0)), CORR(ARG3, ARG0, 10))))",
    'alpha055': None,  # extremely complex ternary (IF_POS available but deeply nested)
    'alpha056': None,  # comparison chaining (GT/LT/GE/LE available, exact formula TBD)
    'alpha057': "SMA(DIV(SUB(ARG3, TSMIN(ARG2, 9)), SUB(TSMAX(ARG1, 9), TSMIN(ARG2, 9))), 3, 1)",
    'alpha058': "MUL(DIV(SUM(GT(ARG3, DELAY(ARG3, 1)), 20), 20), 100.0)",  # COUNT → SUM+GT
    'alpha059': "SUM(IF_POS(SUB(ARG3, DELAY(ARG3, 1)), SUB(ARG3, MIN2(ARG2, DELAY(ARG3, 1))), SUB(ARG3, MAX2(ARG1, DELAY(ARG3, 1)))), 20)",
    'alpha060': "SUM(MUL(DIV(SUB(SUB(ARG3, ARG2), SUB(ARG1, ARG3)), SUB(ARG1, ARG2)), ARG5), 20)",

    # ===== Alpha 061-080 =====
    'alpha061': "NEG(MAX2(CS_RANK(DECAYLINEAR(DELTA(ARG6, 1), 12)), CS_RANK(DECAYLINEAR(CS_RANK(CORR(ARG2, MEAN(ARG5, 80), 8)), 17))))",
    'alpha062': "NEG(CORR(ARG1, CS_RANK(ARG5), 5))",
    'alpha063': "MUL(DIV(SMA(MAX2(SUB(ARG3, DELAY(ARG3, 1)), 0.0), 6, 1), SMA(ABS(SUB(ARG3, DELAY(ARG3, 1))), 6, 1)), 100.0)",
    'alpha064': "NEG(MAX2(CS_RANK(DECAYLINEAR(CORR(CS_RANK(ARG6), CS_RANK(ARG5), 4), 4)), CS_RANK(DECAYLINEAR(TSMAX(CORR(CS_RANK(ARG3), CS_RANK(MEAN(ARG5, 60)), 4), 14), 14))))",
    'alpha065': "DIV(MEAN(ARG3, 6), ARG3)",
    'alpha066': "MUL(DIV(SUB(ARG3, MEAN(ARG3, 6)), MEAN(ARG3, 6)), 100.0)",
    'alpha067': "MUL(DIV(SMA(MAX2(SUB(ARG3, DELAY(ARG3, 1)), 0.0), 24, 1), SMA(ABS(SUB(ARG3, DELAY(ARG3, 1))), 24, 1)), 100.0)",
    'alpha068': "SMA(MUL(SUB(DIV(ADD(ARG1, ARG2), 2), DIV(ADD(DELAY(ARG1, 1), DELAY(ARG2, 1)), 2)), DIV(SUB(ARG1, ARG2), ARG5)), 15, 2)",
    # alpha069: DTM/DBM ratio — if SUM(DTM,20)>SUM(DBM,20): diff/DTM else: diff/DBM
    'alpha069': "IF_POS(SUB(SUM(DTM(ARG0,ARG1),20),SUM(DBM(ARG0,ARG2),20)),DIV(SUB(SUM(DTM(ARG0,ARG1),20),SUM(DBM(ARG0,ARG2),20)),SUM(DTM(ARG0,ARG1),20)),IF_POS(SUB(SUM(DBM(ARG0,ARG2),20),SUM(DTM(ARG0,ARG1),20)),DIV(SUB(SUM(DTM(ARG0,ARG1),20),SUM(DBM(ARG0,ARG2),20)),SUM(DBM(ARG0,ARG2),20)),0.0))",
    'alpha070': "STD(MUL(ARG5, ARG6), 6)",  # AMOUNT ≈ VOLUME*VWAP
    'alpha071': "MUL(DIV(SUB(ARG3, MEAN(ARG3, 24)), MEAN(ARG3, 24)), 100.0)",
    'alpha072': "SMA(DIV(SUB(TSMAX(ARG1, 6), ARG3), SUB(TSMAX(ARG1, 6), TSMIN(ARG2, 6))), 15, 1)",
    'alpha073': "NEG(SUB(TSRANK(DECAYLINEAR(DECAYLINEAR(CORR(ARG3, ARG5, 10), 16), 4), 5), CS_RANK(DECAYLINEAR(CORR(ARG6, MEAN(ARG5, 30), 4), 3))))",
    'alpha074': "ADD(CS_RANK(CORR(SUM(ADD(MUL(ARG2, 0.618), MUL(ARG6, 0.618)), 20), SUM(MEAN(ARG5, 40), 20), 7)), CS_RANK(CORR(CS_RANK(ARG6), CS_RANK(ARG5), 6)))",
    'alpha075': None,  # BANCHMARKINDEX
    'alpha076': None,  # complex nested division
    'alpha077': "MIN2(CS_RANK(DECAYLINEAR(SUB(ADD(DIV(ADD(ARG1, ARG2), 2), ARG1), ADD(ARG6, ARG1)), 20)), CS_RANK(DECAYLINEAR(CORR(DIV(ADD(ARG1, ARG2), 2), MEAN(ARG5, 40), 3), 6)))",
    'alpha078': None,  # complex MA + ABS nesting
    'alpha079': "MUL(DIV(SMA(MAX2(SUB(ARG3, DELAY(ARG3, 1)), 0.0), 12, 1), SMA(ABS(SUB(ARG3, DELAY(ARG3, 1))), 12, 1)), 100.0)",
    'alpha080': "MUL(DIV(SUB(ARG5, DELAY(ARG5, 5)), DELAY(ARG5, 5)), 100.0)",

    # ===== Alpha 081-100 =====
    'alpha081': "SMA(ARG5, 21, 2)",
    'alpha082': "SMA(DIV(SUB(TSMAX(ARG1, 6), ARG3), SUB(TSMAX(ARG1, 6), TSMIN(ARG2, 6))), 20, 1)",
    'alpha083': "NEG(CS_RANK(COVIANCE(CS_RANK(ARG1), CS_RANK(ARG5), 5)))",
    'alpha084': "SUM(IF_POS(SUB(ARG3, DELAY(ARG3, 1)), ARG5, IF_POS(SUB(DELAY(ARG3, 1), ARG3), NEG(ARG5), 0.0)), 20)",
    'alpha085': "MUL(TSRANK(DIV(ARG5, MEAN(ARG5, 20)), 20), TSRANK(NEG(DELTA(ARG3, 7)), 8))",
    'alpha086': None,  # ternary with comparison
    'alpha087': None,  # complex nested
    'alpha088': "MUL(DIV(SUB(ARG3, DELAY(ARG3, 20)), DELAY(ARG3, 20)), 100.0)",
    'alpha089': None,  # multi SMA nesting
    'alpha090': "NEG(CS_RANK(CORR(CS_RANK(ARG6), CS_RANK(ARG5), 5)))",
    'alpha091': "NEG(MUL(CS_RANK(SUB(ARG3, TSMAX(ARG3, 5))), CS_RANK(CORR(MEAN(ARG5, 40), ARG2, 5))))",
    'alpha092': "NEG(MAX2(CS_RANK(DECAYLINEAR(DELTA(ADD(MUL(ARG3, 0.618), MUL(ARG6, 0.618)), 2), 3)), TSRANK(DECAYLINEAR(ABS(CORR(MEAN(ARG5, 180), ARG3, 13)), 5), 15)))",
    'alpha093': None,  # ternary + nested MAX
    'alpha094': "SUM(IF_POS(SUB(ARG3, DELAY(ARG3, 1)), ARG5, IF_POS(SUB(DELAY(ARG3, 1), ARG3), NEG(ARG5), 0.0)), 30)",
    'alpha095': "STD(MUL(ARG5, ARG6), 20)",
    'alpha096': "SMA(SMA(DIV(SUB(ARG3, TSMIN(ARG2, 9)), SUB(TSMAX(ARG1, 9), TSMIN(ARG2, 9))), 3, 1), 3, 1)",
    'alpha097': "STD(ARG5, 10)",
    'alpha098': None,  # complex comparison + ternary
    'alpha099': "NEG(CS_RANK(COVIANCE(CS_RANK(ARG3), CS_RANK(ARG5), 5)))",
    'alpha100': "STD(ARG5, 20)",

    # ===== Alpha 101-120 =====
    'alpha101': None,  # comparison chain (GT/LT available, exact formula TBD)
    'alpha102': "MUL(DIV(SMA(MAX2(SUB(ARG5, DELAY(ARG5, 1)), 0.0), 6, 1), SMA(ABS(SUB(ARG5, DELAY(ARG5, 1))), 6, 1)), 100.0)",
    'alpha103': "TSARGMIN(ARG2, 20)",  # LOWDAY → TSARGMIN
    'alpha104': "NEG(MUL(DELTA(CORR(ARG1, ARG5, 5), 5), CS_RANK(STD(ARG3, 20))))",
    'alpha105': "NEG(CORR(CS_RANK(ARG0), CS_RANK(ARG5), 10))",
    'alpha106': "SUB(ARG3, DELAY(ARG3, 20))",
    'alpha107': "MUL(MUL(NEG(CS_RANK(SUB(ARG0, DELAY(ARG1, 1)))), CS_RANK(SUB(ARG0, DELAY(ARG3, 1)))), CS_RANK(SUB(ARG0, DELAY(ARG2, 1))))",
    'alpha108': "NEG(POWER(CS_RANK(SUB(ARG1, TSMIN(ARG1, 2))), CS_RANK(CORR(ARG6, MEAN(ARG5, 120), 6))))",
    'alpha109': "DIV(SMA(SUB(ARG1, ARG2), 10, 2), SMA(SMA(SUB(ARG1, ARG2), 10, 2), 10, 2))",
    'alpha110': None,  # ternary + nested MAX (IF_POS + MAX2 available, exact formula TBD)
    'alpha111': None,  # SMA nesting with VOL (SMA available, exact formula TBD)
    'alpha112': None,  # complex SUM with comparisons (GT/LT + SUM available, exact formula TBD)
    'alpha113': None,  # complex nested RANK+CORR
    'alpha114': None,  # complex division cascade
    'alpha115': None,  # complex RANK^RANK (POWER available, exact formula TBD)
    'alpha116': "REGBETA(MEAN(ARG3, 6), SEQUENCE(6))",  # REGBETA + SEQUENCE
    'alpha117': None,  # complex TSRANK nesting
    'alpha118': "MUL(DIV(SUM(SUB(ARG1, ARG0), 20), SUM(SUB(ARG0, ARG2), 20)), 100.0)",
    'alpha119': None,  # complex RANK nesting
    'alpha120': "DIV(CS_RANK(SUB(ARG6, ARG3)), CS_RANK(ADD(ARG6, ARG3)))",

    # ===== Alpha 121-140 =====
    'alpha121': "NEG(POWER(CS_RANK(SUB(ARG6, TSMIN(ARG6, 12))), TSRANK(CORR(TSRANK(ARG6, 20), TSRANK(MEAN(ARG5, 60), 2), 18), 3)))",
    'alpha122': None,  # multi-layer SMA-LOG nesting (SMA+LOG available, exact formula TBD)
    'alpha123': None,  # comparison chain (GT/LT available, exact formula TBD)
    'alpha124': "DIV(SUB(ARG3, ARG6), DECAYLINEAR(CS_RANK(TSMAX(ARG3, 30)), 2))",
    'alpha125': None,  # complex RANK/DECAY nesting
    'alpha126': "DIV(ADD(ADD(ARG3, ARG1), ARG2), 3)",
    'alpha127': None,  # complex POWER + SQRT (POWER+SQRT available, exact formula TBD)
    'alpha128': None,  # complex SUMIF-like (GT/LT+SUM+MUL available, exact formula TBD)
    'alpha129': "SUM(IF_POS(SUB(DELAY(ARG3, 1), ARG3), ABS(SUB(ARG3, DELAY(ARG3, 1))), 0.0), 12)",
    'alpha130': None,  # complex RANK/DECAY
    'alpha131': None,  # RANK^TSRANK (POWER available, exact formula TBD)
    'alpha132': "MEAN(MUL(ARG5, ARG6), 20)",
    'alpha133': "DIV(SUB(TSARGMAX(ARG1, 20), TSARGMIN(ARG2, 20)), 20)",  # HIGHDAY-LOWDAY
    'alpha134': "MUL(DIV(SUB(ARG3, DELAY(ARG3, 12)), DELAY(ARG3, 12)), ARG5)",
    'alpha135': "SMA(DELAY(DIV(ARG3, DELAY(ARG3, 20)), 1), 20, 1)",
    'alpha136': "MUL(NEG(CS_RANK(DELTA(ARG4, 3))), CORR(ARG0, ARG5, 10))",
    'alpha137': None,  # same as alpha055 (complex ternary)
    'alpha138': None,  # complex RANK/TSRANK
    'alpha139': "NEG(CORR(ARG0, ARG5, 10))",
    'alpha140': None,  # complex RANK/TSRANK

    # ===== Alpha 141-160 =====
    'alpha141': "NEG(CS_RANK(CORR(CS_RANK(ARG1), CS_RANK(MEAN(ARG5, 15)), 9)))",
    'alpha142': None,  # complex RANK nesting
    'alpha143': None,  # SELF recursive (SELF primitive now available, exact formula TBD)
    'alpha144': "DIV(SUM(MUL(ARG3, GT(ARG3, DELAY(ARG3, 1))), 20), SUM(GT(ARG3, DELAY(ARG3, 1)), 20))",  # SUMIF/COUNT → MUL+GT+SUM
    'alpha145': "MUL(DIV(SUB(MEAN(ARG5, 9), MEAN(ARG5, 26)), MEAN(ARG5, 12)), 100.0)",
    'alpha146': None,  # complex MEAN/SMA nesting
    'alpha147': "REGBETA(MEAN(ARG3, 12), SEQUENCE(12))",  # REGBETA + SEQUENCE
    'alpha148': None,  # comparison chain (GT/LT available, exact formula TBD)
    'alpha149': None,  # REGBETA available, but needs benchmark index for FILTER
    'alpha150': "MUL(DIV(ADD(ADD(ARG3, ARG1), ARG2), 3), ARG5)",
    'alpha151': "SMA(SUB(ARG3, DELAY(ARG3, 20)), 20, 1)",
    'alpha152': None,  # complex SMA/MEAN/DELAY nesting (all primitives available)
    'alpha153': "DIV(ADD(ADD(MEAN(ARG3, 3), MEAN(ARG3, 6)), ADD(MEAN(ARG3, 12), MEAN(ARG3, 24))), 4)",
    'alpha154': None,  # comparison (GT/LT available, exact formula TBD)
    'alpha155': None,  # SMA multi-layer (SMA available, exact formula TBD)
    'alpha156': "NEG(MAX2(CS_RANK(DECAYLINEAR(DELTA(ARG6, 5), 3)), CS_RANK(DECAYLINEAR(NEG(DIV(DELTA(ADD(MUL(ARG0, 0.618), MUL(ARG2, 0.618)), 2), ADD(MUL(ARG0, 0.618), MUL(ARG2, 0.618)))), 3))))",
    'alpha157': None,  # PROD + MIN + TSRANK (PROD/TSMIN/TSRANK available)
    'alpha158': None,  # SMA + complex (SMA available, exact formula TBD)
    'alpha159': None,  # has HGIH typo, complex nested
    'alpha160': "SMA(IF_POS(SUB(DELAY(ARG3, 1), ARG3), STD(ARG3, 20), 0.0), 20, 1)",

    # ===== Alpha 161-180 =====
    'alpha161': None,  # MEAN of MAX of ABS (all primitives available, exact formula TBD)
    'alpha162': None,  # SMA(RSI-style) (SMA available, exact formula TBD)
    'alpha163': "CS_RANK(MUL(MUL(MUL(NEG(ARG4), MEAN(ARG5, 20)), ARG6), SUB(ARG1, ARG3)))",
    'alpha164': None,  # SMA + complex ternary (IF_POS+SMA available, exact formula TBD)
    'alpha165': None,  # SUMAC + MIN/MAX (SUMAC/TSMIN/TSMAX available, exact formula TBD)
    'alpha166': None,  # complex statistics formula
    'alpha167': "SUM(IF_POS(SUB(ARG3, DELAY(ARG3, 1)), SUB(ARG3, DELAY(ARG3, 1)), 0.0), 12)",
    'alpha168': "NEG(DIV(ARG5, MEAN(ARG5, 20)))",
    'alpha169': None,  # SMA/MEAN/DELAY multi-layer (all primitives available)
    'alpha170': None,  # complex RANK nesting
    'alpha171': None,  # complex POWER division (POWER available, exact formula TBD)
    'alpha172': None,  # HD/LD/TR-based (HD/LD/TR primitives available, exact formula TBD)
    'alpha173': None,  # SMA + LOG multi-layer (SMA+LOG available, exact formula TBD)
    'alpha174': "SMA(IF_POS(SUB(ARG3, DELAY(ARG3, 1)), STD(ARG3, 20), 0.0), 20, 1)",
    'alpha175': None,  # MEAN of MAX of ABS (ts) (MEAN/TSMAX/ABS available, exact formula TBD)
    'alpha176': None,  # CORR of RANK (CORR+CS_RANK available, exact formula TBD)
    'alpha177': "TSARGMAX(ARG3, 20)",  # HIGHDAY → TSARGMAX
    'alpha178': "MUL(DIV(SUB(ARG3, DELAY(ARG3, 1)), DELAY(ARG3, 1)), ARG5)",
    'alpha179': "MUL(CS_RANK(CORR(ARG6, ARG5, 4)), CS_RANK(CORR(CS_RANK(ARG2), CS_RANK(MEAN(ARG5, 50)), 12)))",
    'alpha180': "IF_POS(SUB(ARG5, MEAN(ARG5, 20)), MUL(NEG(TSRANK(ABS(DELTA(ARG3, 7)), 60)), SIGN(DELTA(ARG3, 7))), NEG(ARG5))",

    # ===== Alpha 181-191 =====
    'alpha181': None,  # benchmark + complex SUM (needs external benchmark index)
    'alpha182': None,  # benchmark + COUNT/OR (GT+SUM available for COUNT, needs benchmark)
    'alpha183': None,  # SUMAC + MIN/MAX/STD (SUMAC/TSMIN/TSMAX/STD available, exact formula TBD)
    'alpha184': "ADD(CS_RANK(CORR(DELAY(SUB(ARG0, ARG3), 1), ARG3, 200)), CS_RANK(SUB(ARG0, ARG3)))",
    'alpha185': "CS_RANK(SQUARE(SUB(1.0, DIV(ARG0, ARG3))))",
    'alpha186': None,  # HD/LD/TR-based (HD/LD/TR primitives available, exact formula TBD)
    'alpha187': None,  # ternary + MAX nesting (IF_POS+MAX2 available, exact formula TBD)
    'alpha188': "SMA(SUB(ARG1, ARG2), 5, 1)",  # SMA of HIGH-LOW
    'alpha189': "MEAN(ABS(SUB(ARG3, MEAN(ARG3, 6))), 6)",
    'alpha190': None,  # COUNT + SUMIF + LOG + benchmark (GT/SUM/MUL available, needs benchmark)
    'alpha191': "SUB(ADD(CORR(MEAN(ARG5, 20), ARG2, 5), DIV(ADD(ARG1, ARG2), 2)), ARG3)",
}

# ============================================================
# Statistics
# ============================================================

def stats():
    """Print summary: how many factors are expressible."""
    total = len(FACTOR_EXPRS)
    ok = sum(1 for v in FACTOR_EXPRS.values() if v is not None)
    print(f"GTJA 191: {ok}/{total} expressible as DEAP GP trees ({ok*100//total}%)")
    return ok, total


if __name__ == '__main__':
    stats()
