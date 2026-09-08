# -*- coding: utf-8 -*-
"""
Fronteira aprovacao x velocidade do prop challenge (alvo +5%, total -10%,
diaria -5% = reprova) sobre os trades STRESS na janela 19-00h BRT.

Metodologia anti-vies:
- Cestas definidas pelo ranking de TREINO (train_ranking.csv).
- O Monte Carlo avalia APENAS os trades FORWARD (out-of-sample).
- Bootstrap de dias inteiros (trades do mesmo dia juntos -> correlacao).

Varre: solos (top treino) e cestas top2..topN x risco {0.75..3}%.
Criterio: mais rapido (mediana de dias) sujeito a aprovacao >= 95%;
reporta tambem o mais rapido com >= 90%.
"""
import numpy as np
import pandas as pd

import os
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SL_PCT = 1.5
RISKS = [0.0075, 0.01, 0.015, 0.02, 0.025, 0.03]
N_SIMS_GRID, N_SIMS_FINAL = 2000, 10000

df = pd.read_csv(SCRATCH + r"\trades_multi_stress.csv")
df["day"] = pd.to_datetime(df["ts_entry"], unit="ms").dt.date
fwd = df[df["is_forward"]]

rank = pd.read_csv(SCRATCH + r"\train_ranking.csv").sort_values("train_exp", ascending=False)
# elegiveis: expectancia de treino positiva e amostra minima
elig = rank[(rank["train_exp"] > 0) & (rank["train_n"] >= 80)]["symbol"].tolist()
print("Elegiveis (treino exp>0, n>=80), em ordem:", elig)

def day_groups(d):
    return [g["net_pct"].values for _, g in d.groupby("day", sort=True)]

def simulate(days_pool, risk, rng, target=0.05, max_loss=-0.10, daily=-0.05, max_days=120):
    bal = 1.0
    for d in range(1, max_days + 1):
        day = days_pool[int(rng.integers(len(days_pool)))]
        start = bal
        for t in day:
            bal += bal * risk * (t / SL_PCT)
            if (bal - 1.0) <= max_loss:
                return 0, d
            if (bal / start - 1.0) <= daily:
                return 0, d
            if (bal - 1.0) >= target:
                return 1, d
    return 0, max_days          # timeout = reprova

def mc(days_pool, risk, n, seed=23):
    rng = np.random.default_rng(seed)
    wins, days_to_pass = 0, []
    for _ in range(n):
        ok, d = simulate(days_pool, risk, rng)
        wins += ok
        if ok:
            days_to_pass.append(d)
    med = float(np.median(days_to_pass)) if days_to_pass else None
    return wins / n * 100, med

# monta os pools: solos elegiveis + cestas topK
pools = {}
for s in elig:
    pools[f"solo {s.upper()}"] = fwd[fwd["symbol"] == s]
for k in range(2, len(elig) + 1):
    basket = elig[:k]
    pools[f"top{k} ({'+'.join(x.upper() for x in basket)})"] = fwd[fwd["symbol"].isin(basket)]

results = []
for name, d in pools.items():
    if d["day"].nunique() < 30:
        continue
    dp = day_groups(d)
    tpd = len(d) / d["day"].nunique()
    for risk in RISKS:
        ap, med = mc(dp, risk, N_SIMS_GRID)
        results.append(dict(pool=name, risk=risk*100, aprova=ap, med_dias=med,
                            trades_dia=round(tpd, 2)))

res = pd.DataFrame(results)
res.to_csv(SCRATCH + r"\combo_grid.csv", index=False)

def frontier(res, min_ap):
    ok = res[(res["aprova"] >= min_ap) & res["med_dias"].notna()]
    return ok.sort_values(["med_dias", "aprova"], ascending=[True, False]).head(8)

print("\n=== FRONTEIRA: mais rapido com aprovacao >= 95% (grid 2000 sims) ===")
print(frontier(res, 95).to_string(index=False))
print("\n=== FRONTEIRA: mais rapido com aprovacao >= 90% ===")
print(frontier(res, 90).to_string(index=False))
print("\n=== Top aprovacao absoluta ===")
print(res.sort_values('aprova', ascending=False).head(8).to_string(index=False))

# refina os 3 melhores da fronteira 95% com 10k sims
print("\n=== FINALISTAS re-avaliados com 10.000 sims ===")
top = frontier(res, 95).head(3)
for _, r in top.iterrows():
    d = pools[r["pool"]]
    ap, med = mc(day_groups(d), r["risk"]/100, N_SIMS_FINAL, seed=99)
    # historico real da sequencia forward
    bal, peak, mdd, worst = 1.0, 1.0, 0.0, 0.0
    for _, g in d.groupby("day", sort=True):
        s0 = bal
        for t in g["net_pct"].values:
            bal += bal * (r["risk"]/100) * (t / SL_PCT)
            peak = max(peak, bal); mdd = min(mdd, bal/peak - 1)
        worst = min(worst, bal/s0 - 1)
    print(f"{r['pool']:34s} risco {r['risk']:.2f}%: aprova {ap:.1f}% | mediana {med} dias | "
          f"hist: ret {(bal-1)*100:+.1f}%, maxDD {mdd*100:.1f}%, pior dia {worst*100:.1f}%")

