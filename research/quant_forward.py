# -*- coding: utf-8 -*-
"""Avaliacao FORWARD (unica) da config escolhida no treino + Monte Carlo
do prop challenge (+5%, -10%, diaria -5%).

Config selecionada pelo plato do treino (quant_lab3):
  carry+momentum: mom_n=96 (24h), funding percentil 70, stop 3 ATR,
  time-stop 192 barras (48h). Execucao honesta (quant_lab.run_engine).

Sizing no MC: risco fixo por trade; exposicao = risco% / stop_dist%,
cap de alavancagem 3x. Bootstrap de dias inteiros.

Uso: py research/quant_forward.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_lab import SCRATCH, SYMS, run_engine, stats
from quant_lab2 import sig_carry_mom

CFG = dict(stop_atr=3.0, max_bars=192)
MOM_N, F_PCT = 96, 70
LEV_CAP = 3.0

data = {}
for sym in SYMS:
    try:
        df = pd.read_csv(os.path.join(SCRATCH, f"{sym}_15m_bybit.csv"),
                         index_col=0, parse_dates=True)
        fund = pd.read_csv(os.path.join(SCRATCH, f"funding_{sym}.csv"))
    except FileNotFoundError:
        continue
    data[sym] = (df, fund)

print("FORWARD (30% finais) — config fixa do treino: mom 24h, pct70, s3, t48h")
print(f"{'sym':5s} {'n':>4s} {'WR':>6s} {'exp':>8s} {'PF':>5s}")
frames = []
for sym, (df, fund) in data.items():
    split_ms = (df.index[int(len(df) * 0.7)] - pd.Timestamp(0)).total_seconds() * 1000
    # sinal na serie completa (warm-up legitimo vindo do treino);
    # só contam trades com ENTRADA no forward
    tr = run_engine(df, fund, sig_carry_mom(df, fund, MOM_N, F_PCT), CFG)
    tr = [t for t in tr if t[1] >= split_ms]
    s = stats(tr)
    if s:
        print(f"{sym.upper():5s} {s['n']:4d} {s['wr']:5.1f}% {s['exp']:+8.4f} {s['pf']:5.2f}")
    else:
        print(f"{sym.upper():5s}  sem trades")
        continue
    w = pd.DataFrame(tr, columns=["net_pct", "ts_entry", "ts_exit", "side",
                                  "stop_dist_pct"])
    w["symbol"] = sym
    frames.append(w)

allf = pd.concat(frames).sort_values("ts_entry")
p = allf["net_pct"].values
loss = abs(p[p <= 0].sum())
pf = p[p > 0].sum() / loss if loss > 0 else float("inf")
print(f"\nAGREGADO FORWARD: n={len(p)} WR={(p>0).mean()*100:.1f}% "
      f"exp={p.mean():+.4f} PF={pf:.2f}")
allf.to_csv(os.path.join(SCRATCH, "trades_carrymom_fwd.csv"), index=False)

# Monte Carlo do challenge
allf["day"] = pd.to_datetime(allf["ts_entry"], unit="ms").dt.date
days_pool = [g[["net_pct", "stop_dist_pct"]].values
             for _, g in allf.groupby("day", sort=True)]
print(f"\ndias no pool forward: {len(days_pool)} | trades/dia: "
      f"{len(allf)/len(days_pool):.1f}")


def simulate(risk, rng, target=0.05, max_loss=-0.10, daily=-0.05, max_days=120):
    bal = 1.0
    for d in range(1, max_days + 1):
        day = days_pool[int(rng.integers(len(days_pool)))]
        start = bal
        for net, sd in day:
            expo = min(risk * 100 / sd, LEV_CAP) if sd > 0 else 0.0
            bal += bal * (net / 100) * expo
            if (bal - 1.0) <= max_loss:
                return 0, d
            if (bal / start - 1.0) <= daily:
                return 0, d
            if (bal - 1.0) >= target:
                return 1, d
    return 0, max_days


print(f"\n{'risco':>6s} {'aprovacao':>10s} {'mediana dias':>13s}")
for risk in (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03):
    rng = np.random.default_rng(7)
    wins, dtp = 0, []
    for _ in range(10000):
        ok, d = simulate(risk, rng)
        wins += ok
        if ok:
            dtp.append(d)
    med = np.median(dtp) if dtp else float("nan")
    print(f"{risk*100:5.2f}% {wins/100:9.1f}% {med:13.0f}")

# historico real da sequencia forward (sem bootstrap) com risco 1%
bal, peak, mdd, worst = 1.0, 1.0, 0.0, 0.0
for _, g in allf.groupby("day", sort=True):
    s0 = bal
    for net, sd in g[["net_pct", "stop_dist_pct"]].values:
        expo = min(1.0 / sd, LEV_CAP) if sd > 0 else 0.0
        bal += bal * (net / 100) * expo
        peak = max(peak, bal)
        mdd = min(mdd, bal / peak - 1)
    worst = min(worst, bal / s0 - 1)
print(f"\nsequencia real fwd @1%: retorno {(bal-1)*100:+.1f}% | "
      f"maxDD {mdd*100:.1f}% | pior dia {worst*100:.1f}%")
