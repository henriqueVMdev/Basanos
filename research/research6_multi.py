# -*- coding: utf-8 -*-
"""
Estrategia MM9 pullback maker, janela 19-00h BRT, em 13 perps Bybit.
Slope 1h SEM lookahead. Exporta trades stress (ano todo) por simbolo e
a tabela treino/forward. A SELECAO de cesta sera feita so com TREINO.
"""
import numpy as np
import pandas as pd

import os
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAKER, TAKER, SLIP = 0.0002, 0.00055, 0.0005
WINDOW_UTC = {22, 23, 0, 1, 2}
SYMS = ["btc", "eth", "sol", "xrp", "doge", "bnb", "ada", "link",
        "avax", "sui", "ltc", "near", "ton"]

def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values

def prep(df):
    O, H, L, C = (df[c].values for c in ["Open", "High", "Low", "Close"])
    TS = (df.index - pd.Timestamp(0)).total_seconds().values * 1000
    HOUR = df.index.hour.values
    h, l, c = pd.Series(H), pd.Series(L), pd.Series(C)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    ATR = tr.ewm(span=14, adjust=False).mean().values
    E9, E50, E200 = ema(C, 9), ema(C, 50), ema(C, 200)
    VOLRANK = pd.Series(ATR / C * 100).rolling(384).rank(pct=True).values
    c1h = pd.Series(C, index=df.index).resample("1h").last()
    e1h = c1h.ewm(span=9, adjust=False).mean()
    slope_raw = e1h.diff() > 0
    slope_raw.index = slope_raw.index + pd.Timedelta(hours=1)
    slope1h = slope_raw.reindex(df.index, method="ffill").fillna(False).values.astype(bool)
    prev = lambda a: np.roll(a, 1)
    up = prev((E50 > E200) & (C > E200)) & prev(slope1h) & prev(C > E9)
    dn = prev((E50 < E200) & (C < E200)) & prev(np.logical_not(slope1h)) & prev(C < E9)
    hivol = prev(VOLRANK >= 0.5)
    return dict(O=O, H=H, L=L, C=C, TS=TS, HOUR=HOUR, E9=E9,
                setup=(up | dn) & hivol, sides=np.where(up, 1, -1), N=len(df))

def run(d, fund_ts, fund_rate, tp_pct=0.5, sl_pct=1.5, max_bars=48,
        pess=True, mask=None):
    O, H, L, C, TS, E9 = d["O"], d["H"], d["L"], d["C"], d["TS"], d["E9"]
    setup, sides, N, HOUR = d["setup"], d["sides"], d["N"], d["HOUR"]
    fmult = 1.5 if pess else 1.0
    slip = SLIP if pess else 0.0
    trades = []
    i = 1
    while i < N - 1:
        ok = setup[i] and HOUR[i] in WINDOW_UTC and (mask is None or mask[i])
        if not ok:
            i += 1; continue
        side = sides[i]
        limit = E9[i-1]
        filled = (L[i] < limit) if side == 1 else (H[i] > limit)
        if not filled:
            i += 1; continue
        entry = limit
        tp = entry * (1 + tp_pct/100*side)
        sl = entry * (1 - sl_pct/100*side)
        exit_price, exit_fee, exit_slip = None, TAKER, slip
        j = i
        while j < N:
            hit_sl = (L[j] <= sl) if side == 1 else (H[j] >= sl)
            hit_tp = (H[j] > tp) if side == 1 else (L[j] < tp)
            if j == i and hit_tp:
                # TP no candle do fill so vale se a trajetoria intrabar
                # negocia o alvo DEPOIS do fill (long+verde/short+vermelho)
                hit_tp = (C[j] >= O[j]) if side == 1 else (C[j] < O[j])
            if hit_sl:
                exit_price, exit_fee, exit_slip = sl, TAKER, slip; break
            if hit_tp:
                exit_price, exit_fee, exit_slip = tp, MAKER, 0.0; break
            if j - i + 1 >= max_bars:
                exit_price, exit_fee, exit_slip = C[j], TAKER, slip; break
            j += 1
        if exit_price is None:
            j = N - 1; exit_price = C[j]
        gross = (exit_price/entry - 1) * side * 100
        i0 = np.searchsorted(fund_ts, TS[i], side="right")
        i1 = np.searchsorted(fund_ts, TS[j], side="right")
        f = fund_rate[i0:i1].sum() * fmult if i1 > i0 else 0.0
        net = gross - (MAKER + exit_fee)*100 - exit_slip*100 - (f if side == 1 else -f)*100
        trades.append((net, TS[i], TS[j], side))
        i = j + 1
    return trades

def stats(trades):
    if not trades: return None
    p = np.array([t[0] for t in trades])
    loss = p[p <= 0].sum()
    pf = p[p > 0].sum()/abs(loss) if loss != 0 else float("inf")
    days = (trades[-1][2]-trades[0][1])/86400000
    return dict(n=len(p), wr=round((p > 0).mean()*100, 1), exp=round(p.mean(), 4),
                pf=round(pf, 2), tot=round(p.sum(), 1), tpd=round(len(p)/max(days, 1), 2))

fmt = lambda s: (f"{s['n']:4d} {s['wr']:5.1f}% {s['exp']:+8.4f} {s['pf']:5.2f} "
                 f"{s['tot']:+7.1f} {s['tpd']:4.2f}/d") if s else "sem trades"

if __name__ == "__main__":
    rows = []
    frames = []
    print(f"{'sym':5s} | {'TREINO (stress) n/wr/exp/pf/tot/freq':42s} | {'FORWARD (stress) n/wr/exp/pf/tot/freq':42s}")
    print("-"*100)
    for sym in SYMS:
        try:
            df = pd.read_csv(SCRATCH + fr"\{sym}_15m_bybit.csv", index_col=0, parse_dates=True)
            fund = pd.read_csv(SCRATCH + fr"\funding_{sym}.csv")
        except FileNotFoundError:
            print(f"{sym.upper():5s} | dados ausentes — pulado"); continue
        d = prep(df)
        N = d["N"]; SPLIT = int(N * 0.7)
        tr_mask = np.arange(N) < SPLIT
        fts, fr = fund["timestamp"].values, fund["rate"].values
        s1 = stats(run(d, fts, fr, mask=tr_mask))
        s2 = stats(run(d, fts, fr, mask=~tr_mask))
        print(f"{sym.upper():5s} | {fmt(s1):42s} | {fmt(s2):42s}")
        rows.append({"symbol": sym,
                     "train_exp": s1["exp"] if s1 else np.nan,
                     "train_pf": s1["pf"] if s1 else np.nan,
                     "train_n": s1["n"] if s1 else 0})
        tr = run(d, fts, fr)          # ano todo, stress
        w = pd.DataFrame(tr, columns=["net_pct", "ts_entry", "ts_exit", "side"])
        w["symbol"] = sym
        # split por simbolo (70% dos candles daquele simbolo)
        split_ms = (df.index[SPLIT] - pd.Timestamp(0)).total_seconds() * 1000
        w["is_forward"] = w["ts_entry"] >= split_ms
        frames.append(w)

    pd.concat(frames).sort_values("ts_entry").to_csv(SCRATCH + r"\trades_multi_stress.csv", index=False)
    pd.DataFrame(rows).to_csv(SCRATCH + r"\train_ranking.csv", index=False)
    print("\nRanking TREINO (base da selecao — forward intocado):")
    rk = pd.DataFrame(rows).sort_values("train_exp", ascending=False)
    print(rk.to_string(index=False))

