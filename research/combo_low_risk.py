# -*- coding: utf-8 -*-
"""Cestas maiores x risco menor: diversificacao vs regra diaria."""
import numpy as np
import pandas as pd

import os
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SL = 1.5
df = pd.read_csv(SCRATCH + r"\trades_multi_stress.csv")
df["day"] = pd.to_datetime(df["ts_entry"], unit="ms").dt.date
fwd = df[df.is_forward]
rank = pd.read_csv(SCRATCH + r"\train_ranking.csv").sort_values("train_exp", ascending=False)
elig = rank[(rank.train_exp > 0) & (rank.train_n >= 80)].symbol.tolist()

def dg(d):
    return [g["net_pct"].values for _, g in d.groupby("day", sort=True)]

def sim(dp, risk, rng, target=.05, ml=-.10, dl=-.05, md=120):
    bal = 1.0
    for d in range(1, md + 1):
        day = dp[int(rng.integers(len(dp)))]
        s = bal
        for t in day:
            bal += bal * risk * (t / SL)
            if bal - 1 <= ml: return 0, d
            if bal / s - 1 <= dl: return 0, d
            if bal - 1 >= target: return 1, d
    return 0, md

def mc(dp, risk, n=3000, seed=7):
    rng = np.random.default_rng(seed)
    w, dd = 0, []
    for _ in range(n):
        ok, d = sim(dp, risk, rng)
        w += ok
        if ok: dd.append(d)
    return w / n * 100, float(np.median(dd)) if dd else None

pools = {f"top{k}": fwd[fwd.symbol.isin(elig[:k])] for k in (3, 4, 5, 8, 12)}
print(f"{'pool':8s} {'risco%':>7s} {'aprova%':>8s} {'med_dias':>9s} {'tr/dia':>7s}")
best = []
for name, d in pools.items():
    dp = dg(d)
    tpd = len(d) / d.day.nunique()
    for risk in (0.004, 0.005, 0.006, 0.0075, 0.01):
        ap, med = mc(dp, risk)
        best.append((name, risk, ap, med))
        print(f"{name:8s} {risk*100:7.2f} {ap:8.1f} {str(med):>9s} {tpd:7.2f}")

print("\n=== FINALISTAS 10k sims (aprova>=95, mais rapidos) ===")
ok = sorted([b for b in best if b[2] >= 95 and b[3]], key=lambda x: (x[3], -x[2]))[:3]
for name, risk, _, _ in ok:
    d = pools[name]
    ap, med = mc(dg(d), risk, n=10000, seed=99)
    bal, peak, mdd, worst = 1.0, 1.0, 0.0, 0.0
    for _, g in d.groupby("day", sort=True):
        s0 = bal
        for t in g["net_pct"].values:
            bal += bal * risk * (t / SL)
            peak = max(peak, bal)
            mdd = min(mdd, bal / peak - 1)
        worst = min(worst, bal / s0 - 1)
    print(f"{name} risco {risk*100:.2f}%: aprova {ap:.1f}% | mediana {med} d | "
          f"hist ret {(bal-1)*100:+.1f}%, maxDD {mdd*100:.1f}%, pior dia {worst*100:.1f}%")

