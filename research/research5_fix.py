# -*- coding: utf-8 -*-
"""
CORRECAO DE LOOKAHEAD: o slope da EMA9 1h era calculado com
resample('1h').last() + reindex(ffill) â€” o bucket 10:00 contem o close
de 10:45, visto pelo candle 15m de 10:00/10:15 (ate 45min de futuro).

Fix: o valor do bucket 1h so fica disponivel em bucket_start + 1h
(apos o fechamento da hora). Re-roda TODAS as variantes com o filtro
honesto: 24h e janela 19-00h BRT, 5 simbolos, realista + stress.
"""
import numpy as np
import pandas as pd

import os
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAKER, TAKER, SLIP = 0.0002, 0.00055, 0.0005
WINDOW_UTC = {22, 23, 0, 1, 2}       # 19:00-23:59 BRT (UTC-3)

def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values

def load(sym):
    df = pd.read_csv(SCRATCH + fr"\{sym}_15m_bybit.csv", index_col=0, parse_dates=True)
    fund = pd.read_csv(SCRATCH + fr"\funding_{sym}.csv")
    return df, fund["timestamp"].values, fund["rate"].values

def prep(df):
    O, H, L, C = (df[c].values for c in ["Open", "High", "Low", "Close"])
    TS = (df.index - pd.Timestamp(0)).total_seconds().values * 1000
    assert TS[0] > 1e12
    HOUR = df.index.hour.values
    h, l, c = pd.Series(H), pd.Series(L), pd.Series(C)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    ATR = tr.ewm(span=14, adjust=False).mean().values
    E9, E50, E200 = ema(C, 9), ema(C, 50), ema(C, 200)
    VOLRANK = pd.Series(ATR / C * 100).rolling(384).rank(pct=True).values

    # â”€â”€ SLOPE 1h SEM LOOKAHEAD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # bucket 1h so e conhecido apos fechar: disponivel em start + 1h
    c1h = pd.Series(C, index=df.index).resample("1h").last()
    e1h = c1h.ewm(span=9, adjust=False).mean()
    slope_raw = e1h.diff() > 0
    slope_raw.index = slope_raw.index + pd.Timedelta(hours=1)   # disponibilidade
    slope1h = slope_raw.reindex(df.index, method="ffill").fillna(False).values

    prev = lambda a: np.roll(a, 1)
    up = prev((E50 > E200) & (C > E200)) & prev(slope1h) & prev(C > E9)
    dn = prev((E50 < E200) & (C < E200)) & prev(~slope1h) & prev(C < E9)
    hivol = prev(VOLRANK >= 0.5)
    setup = (up | dn) & hivol
    sides = np.where(up, 1, -1)
    return dict(O=O, H=H, L=L, C=C, TS=TS, HOUR=HOUR, E9=E9,
                setup=setup, sides=sides, N=len(df))

def run(d, fund_ts, fund_rate, tp_pct, sl_pct, max_bars=48,
        hour_filter=False, pess=False, mask=None):
    H, L, C, TS, E9 = d["H"], d["L"], d["C"], d["TS"], d["E9"]
    setup, sides, N, HOUR = d["setup"], d["sides"], d["N"], d["HOUR"]
    fmult = 1.5 if pess else 1.0
    slip = SLIP if pess else 0.0
    trades = []
    i = 1
    while i < N - 1:
        ok = setup[i] and (mask is None or mask[i])
        if hour_filter and HOUR[i] not in WINDOW_UTC:
            ok = False
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
        fc = (f if side == 1 else -f) * 100
        net = gross - (MAKER + exit_fee)*100 - exit_slip*100 - fc
        trades.append((net, TS[i], TS[j], side))
        i = j + 1
    return trades

def stats(trades):
    if not trades: return None
    p = np.array([t[0] for t in trades])
    wr = (p > 0).mean()*100
    loss = p[p <= 0].sum()
    pf = p[p > 0].sum()/abs(loss) if loss != 0 else float("inf")
    days = (trades[-1][2]-trades[0][1])/86400000
    return dict(n=len(p), wr=round(wr,1), exp=round(p.mean(),4), pf=round(pf,2),
                tot=round(p.sum(),1), tpd=round(len(p)/max(days,1), 2))

fmt = lambda s: (f"{s['n']:4d} {s['wr']:5.1f}% {s['exp']:+8.4f} {s['pf']:5.2f} "
                 f"{s['tot']:+7.1f} {s['tpd']:4.2f}/d") if s else "     sem trades"

print(f"{'symbol/variante':34s} | {'TREINO n/wr/exp/pf/tot/freq':40s} | {'FORWARD n/wr/exp/pf/tot/freq':40s}")
print("-"*122)

all_win = {}
for sym in ["btc", "eth", "sol", "xrp", "doge"]:
    df, fts, fr = load(sym)
    d = prep(df)
    N = d["N"]; SPLIT = int(N*0.7)
    tr_mask = np.arange(N) < SPLIT
    te_mask = ~tr_mask
    for label, hf in [("24h   ", False), ("19-00h", True)]:
        s1 = stats(run(d, fts, fr, 0.5, 1.5, hour_filter=hf, mask=tr_mask))
        s2 = stats(run(d, fts, fr, 0.5, 1.5, hour_filter=hf, mask=te_mask))
        print(f"{sym.upper():5s} {label} tp0.5/sl1.5        | {fmt(s1):40s} | {fmt(s2):40s}")
    s2s = stats(run(d, fts, fr, 0.5, 1.5, hour_filter=True, mask=te_mask, pess=True))
    print(f"{sym.upper():5s} 19-00h stress forward     | {'':40s} | {fmt(s2s):40s}")
    tr = run(d, fts, fr, 0.5, 1.5, hour_filter=True, pess=True)
    w = pd.DataFrame(tr, columns=["net_pct", "ts_entry", "ts_exit", "side"])
    w["symbol"] = sym
    all_win[sym] = w
    print()

combo = pd.concat(all_win.values()).sort_values("ts_entry").reset_index(drop=True)
combo.to_csv(SCRATCH + r"\trades_horario_stress.csv", index=False)
print(f"export portfolio 19-00h stress (SEM lookahead): {len(combo)} trades")

