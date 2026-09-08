"""Portfolio risk analytics using adjusted daily returns."""
from __future__ import annotations
import math
def _annualization_factor(returns):
    import pandas as pd
    trailing_start=returns.index.max()-pd.Timedelta(days=365)
    value=float(len(returns[returns.index>trailing_start]))
    return max(60.0,min(366.0,value))

def analyze(symbols, years=3):
    import numpy as np, pandas as pd, yfinance as yf
    syms=[str(x).strip().upper() for x in symbols if str(x).strip()][:20]
    if len(syms)<2:raise ValueError('informe pelo menos dois ativos')
    raw=yf.download(syms,period=f'{int(years)}y',auto_adjust=True,progress=False,group_by='column')
    close=raw['Close'] if isinstance(raw.columns,pd.MultiIndex) else raw[['Close']].rename(columns={'Close':syms[0]})
    # Use only timestamps shared by every asset. This avoids turning equity
    # weekends into artificial zero returns when the portfolio also has crypto.
    close=close.dropna(axis=1,how='all').dropna(how='any')
    returns=close.pct_change(fill_method=None).dropna(how='any')
    if len(returns)<60: raise ValueError('histórico comum insuficiente entre os ativos')
    # Mixed portfolios naturally resolve to ~252 shared dates; all-crypto
    # portfolios resolve to ~365. Infer instead of imposing an incorrect basis.
    annual_factor=_annualization_factor(returns)
    labels=list(returns.columns);n=len(labels);w=np.repeat(1/n,n);cov=returns.cov().values*annual_factor
    port_vol=float(np.sqrt(w@cov@w));marg=cov@w;contrib=w*marg/(port_vol**2)
    curve=(1+returns@w).cumprod();dd=curve/curve.cummax()-1
    metrics=[]
    for s in labels:
        r=returns[s];eq=(1+r).cumprod();d=eq/eq.cummax()-1
        metrics.append({'symbol':s,'return_ann_pct':float(r.mean()*annual_factor*100),'vol_ann_pct':float(r.std()*math.sqrt(annual_factor)*100),'sharpe':float(r.mean()/r.std()*math.sqrt(annual_factor)) if r.std() else None,'max_drawdown_pct':float(d.min()*100)})
    return {'symbols':labels,'correlation':returns.corr().round(4).values.tolist(),'metrics':metrics,'portfolio':{'vol_ann_pct':port_vol*100,'return_ann_pct':float((returns@w).mean()*annual_factor*100),'sharpe':float((returns@w).mean()/(returns@w).std()*math.sqrt(annual_factor)),'max_drawdown_pct':float(dd.min()*100),'risk_contribution_pct':[float(x*100) for x in contrib]},'observations':len(returns),'annualization_factor':annual_factor,'methodology':f'Pesos iguais; somente datas comuns; retornos ajustados; anualização inferida em {annual_factor:.0f} observações/ano.'}
